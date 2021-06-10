// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "clustering/clusterers/parallel-correlation-clusterer.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/parallel-correlation-clusterer-internal.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

#include "external/gbbs/benchmarks/KCore/JulienneDBS17/KCore.h"
#include "external/gbbs/gbbs/pbbslib/sparse_additive_map.h"
#include "external/gbbs/pbbslib/random_shuffle.h"

namespace research_graph {
namespace in_memory {

namespace {

// This struct is necessary to perform an edge map with GBBS over a vertex
// set. Essentially, all neighbors are valid in this edge map, and this
// map does not do anything except allow for neighbors to be aggregated
// into the next frontier.
struct CorrelationClustererEdgeMap {
  inline bool cond(gbbs::uintE d) { return true; }
  inline bool update(const gbbs::uintE& s, const gbbs::uintE& d, float wgh) {
    return true;
  }
  inline bool updateAtomic(const gbbs::uintE& s, const gbbs::uintE& d,
                           float wgh) {
    return true;
  }
};

struct CorrelationClustererRefine {
  using H = std::unique_ptr<ClusteringHelper>;
  using G = std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>;
  gbbs::sequence<H> recurse_helpers = gbbs::sequence<H>(0, [](std::size_t i){return H(nullptr);});
  gbbs::sequence<G> recurse_graphs = gbbs::sequence<G>(0, [](std::size_t i){return G(nullptr);});
  bool use_refine = false;
};


// Takes a vertexSubsetData (with some non-trivial Data) and applies a map
// function f : (uintE x Data) -> void over each vertex in the vertexSubset, in
// parallel.
template <class F, class VS,
          typename std::enable_if<!std::is_same<VS, gbbs::vertexSubset>::value,
                                  int>::type = 0>
inline void vertexMapPermute(VS& V, F f, size_t granularity=pbbslib::kSequentialForThreshold) {
  size_t n = V.numRows(), m = V.numNonzeros();
  if (V.dense()) {
    auto P = pbbslib::random_permutation<gbbs::uintE>(n);
    pbbs::parallel_for(0, n, [&] (size_t x) {
      size_t i = P[x]; 
      if (V.isIn(i)) {
        f(i, V.ithData(i));
      }
    }, granularity);
  } else {
    auto P = pbbslib::random_permutation<gbbs::uintE>(m);
    pbbs::parallel_for(0, m, [&] (size_t x) { size_t i = P[x]; f(V.vtx(i), V.vtxData(i)); }, granularity);
  }
}

// Takes a vertexSubset (with no extra data per-vertex) and applies a map
// function f : uintE -> void over each vertex in the vertexSubset, in
// parallel.
template <class VS, class F,
          typename std::enable_if<std::is_same<VS, gbbs::vertexSubset>::value,
                                  int>::type = 0>
inline void vertexMapPermute(VS& V, F f, size_t granularity=pbbslib::kSequentialForThreshold) {
  size_t n = V.numRows(), m = V.numNonzeros();
  if (V.dense()) {
    auto P = pbbslib::random_permutation<gbbs::uintE>(n);
    pbbs::parallel_for(0, n, [&] (size_t x) {
      size_t i = P[x];
      if (V.isIn(i)) {
        f(i);
      }
    }, granularity);
  } else {
    auto P = pbbslib::random_permutation<gbbs::uintE>(m);
    pbbs::parallel_for(0, m, [&] (size_t x)
                    { size_t i = P[x]; f(V.vtx(i)); }, granularity);
  }
}


// Given a vertex subset moved_subset, computes best moves for all vertices
// and performs the moves. Returns a vertex subset consisting of all vertices
// adjacent to modified clusters.
std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>
BestMovesForVertexSubset(
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
    std::size_t num_nodes, gbbs::vertexSubset* moved_subset,
    ClusteringHelper* helper, const ClustererConfig& clusterer_config,
    CorrelationClustererSubclustering& subclustering) {
  bool permute = clusterer_config.correlation_clusterer_config().permute();
  bool async = clusterer_config.correlation_clusterer_config().async();
  std::vector<absl::optional<ClusteringHelper::ClusterId>> moves(num_nodes,
                                                                 absl::nullopt);
  std::vector<char> moved_vertex(num_nodes, 0);

  // Find best moves per vertex in moved_subset
  gbbs::sequence<bool> async_mark = gbbs::sequence<bool>(current_graph->n, false);
  auto moved_clusters = absl::make_unique<bool[]>(current_graph->n);
  pbbs::parallel_for(0, current_graph->n,
                     [&](std::size_t i) { moved_clusters[i] = false; });
  auto vertex_map_func = [&](std::size_t i) {
    if (async) {
      auto move = helper->AsyncMove(*current_graph, i);
      if (move) {
        pbbslib::CAS<bool>(&moved_clusters[helper->ClusterIds()[i]], false, true);
        moved_vertex[i] = 1;
      }
    } else {
    std::tuple<ClusteringHelper::ClusterId, double> best_move =
        helper->EfficientBestMove(*current_graph, i);
    // If a singleton cluster wishes to move to another singleton cluster,
    // only move if the id of the moving cluster is lower than the id
    // of the cluster it wishes to move to
    auto move_cluster_id = std::get<0>(best_move);
    auto current_cluster_id = helper->ClusterIds()[i];
    if (move_cluster_id < current_graph->n &&
        helper->ClusterSizes()[move_cluster_id] == 1 &&
        helper->ClusterSizes()[current_cluster_id] == 1 &&
        current_cluster_id >= move_cluster_id) {
      best_move = std::make_tuple(current_cluster_id, 0);
    }
    if (std::get<1>(best_move) > 0) {
      moves[i] = std::get<0>(best_move);
      moved_vertex[i] = 1;
    }
    }
  };
  if (permute) vertexMapPermute(*moved_subset, vertex_map_func);
  else gbbs::vertexMap(*moved_subset, vertex_map_func);

  // Compute modified clusters
  if (!async) {
    moved_clusters = helper->MoveNodesToCluster(moves);
  }

  // Perform cluster moves
  if (clusterer_config.correlation_clusterer_config()
          .clustering_moves_method() ==
      CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES) {
    // Reset moves
    if (!async) {
      pbbs::parallel_for(0, num_nodes,
                       [&](std::size_t i) { moves[i] = absl::nullopt; });
    }

    // Aggregate clusters
    auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
    std::vector<std::vector<gbbs::uintE>> curr_clustering =
        parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
            helper->ClusterIds(), get_clusters, helper->ClusterIds().size());

    // Compute sub-clusters by taking existing clusters and running
    // connected components on each cluster with triangle reweighting
    if (clusterer_config.correlation_clusterer_config()
          .subclustering_method() != CorrelationClustererConfig::NONE_SUBCLUSTERING) {
      std::vector<ClusteringHelper::ClusterId> subcluster_ids(num_nodes);
  
      auto next_id_seq = gbbs::sequence<std::size_t>(curr_clustering.size(), [](std::size_t i){return 0;});
      pbbs::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
        next_id_seq[i] = ComputeSubcluster(subcluster_ids, 0, curr_clustering[i], current_graph,
          subclustering, helper->ClusterIds(), clusterer_config);
      });
      auto total_next_id = pbbslib::scan_add_inplace(next_id_seq);

      pbbs::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
        pbbs::parallel_for (0, curr_clustering[i].size(), [&](std::size_t j){
          subcluster_ids[curr_clustering[i][j]] += next_id_seq[i];
        });
      });

      std::vector<std::vector<gbbs::uintE>> curr_subclustering =
          parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
              subcluster_ids, get_clusters, num_nodes);

      // Compute best move per subcluster
      auto additional_moved_subclusters = absl::make_unique<bool[]>(current_graph->n);
      if (async) {
      pbbs::parallel_for(0, current_graph->n,
                     [&](std::size_t i) { additional_moved_subclusters[i] = false; });
      }
      pbbs::parallel_for(0, curr_subclustering.size(), [&](std::size_t i) {
        if (!curr_subclustering[i].empty()) {
          if (async) {
            bool move_flag = helper->AsyncMove(*current_graph, curr_subclustering[i]);
            if (move_flag) {
              pbbs::parallel_for(0, curr_subclustering[i].size(), [&](std::size_t j) {
                additional_moved_subclusters[curr_clustering[i][j]] = true;
              });
            }
          }
          else {
          std::tuple<ClusteringHelper::ClusterId, double> best_move =
              helper->BestMove(*current_graph, curr_subclustering[i]);
          // If a cluster wishes to move to another cluster,
          // only move if the id of the moving cluster is lower than the id
          // of the cluster it wishes to move to
          auto move_cluster_id = std::get<0>(best_move);
          auto current_cluster_id =
              helper->ClusterIds()[curr_subclustering[i].front()];
          if (move_cluster_id < current_graph->n &&
              current_cluster_id >= move_cluster_id) {
            best_move = std::make_tuple(current_cluster_id, 0);
          }
          if (std::get<1>(best_move) > 0) {
            for (size_t j = 0; j < curr_subclustering[i].size(); j++) {
              moves[curr_subclustering[i][j]] = std::get<0>(best_move);
            }
          }
          }
        }
      });
      // Compute modified subclusters
      if (!async) additional_moved_subclusters = helper->MoveNodesToCluster(moves);
      pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
        moved_clusters[i] |= additional_moved_subclusters[i];
      });
      // Reset moves
      if (!async) {
        pbbs::parallel_for(0, num_nodes,
                         [&](std::size_t i) { moves[i] = absl::nullopt; });
      }
    }

    // Compute best move per cluster
    auto additional_moved_clusters = absl::make_unique<bool[]>(current_graph->n);
    if (async) {
    pbbs::parallel_for(0, current_graph->n,
                     [&](std::size_t i) { additional_moved_clusters[i] = false; });
    }
    pbbs::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
      if (!curr_clustering[i].empty()) {
        if (async) {
          bool move_flag = helper->AsyncMove(*current_graph, curr_clustering[i]);
          if (move_flag) {
            pbbs::parallel_for(0, curr_clustering[i].size(), [&](std::size_t j) {
              additional_moved_clusters[curr_clustering[i][j]] = true;
            });
          }
        } else {
          std::tuple<ClusteringHelper::ClusterId, double> best_move =
            helper->BestMove(*current_graph, curr_clustering[i]);
          // If a cluster wishes to move to another cluster,
          // only move if the id of the moving cluster is lower than the id
          // of the cluster it wishes to move to
          auto move_cluster_id = std::get<0>(best_move);
          auto current_cluster_id =
            helper->ClusterIds()[curr_clustering[i].front()];
          if (move_cluster_id < current_graph->n &&
            current_cluster_id >= move_cluster_id) {
            best_move = std::make_tuple(current_cluster_id, 0);
          }
          if (std::get<1>(best_move) > 0) {
            for (size_t j = 0; j < curr_clustering[i].size(); j++) {
              moves[curr_clustering[i][j]] = std::get<0>(best_move);
            }
          }
        }
      }
    });

    // Compute modified clusters
    if(!async) additional_moved_clusters = helper->MoveNodesToCluster(moves);
    pbbs::parallel_for(0, num_nodes, [&](std::size_t i) {
      moved_clusters[i] |= additional_moved_clusters[i];
    });
  }

  if (clusterer_config.correlation_clusterer_config().move_method() == CorrelationClustererConfig::ALL_MOVE) {
    return std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
    new gbbs::vertexSubset(num_nodes, num_nodes,
    gbbs::sequence<bool>(num_nodes, true).to_array()),
    [](gbbs::vertexSubset* subset) {
      subset->del();
      delete subset;
    });
  }

  bool default_move = clusterer_config.correlation_clusterer_config().move_method() == CorrelationClustererConfig::NBHR_CLUSTER_MOVE;
  // Mark vertices adjacent to clusters that have moved; these are
  // the vertices whose best moves must be recomputed.
  auto local_moved_subset =
      std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
          new gbbs::vertexSubset(
              num_nodes, num_nodes,
              gbbs::sequence<bool>(
                  num_nodes,
                  [&](std::size_t i) {
                    if (default_move)
                      return moved_clusters[helper->ClusterIds()[i]];
                    else
                      return (bool) moved_vertex[i];
                  })
                  .to_array()),
          [](gbbs::vertexSubset* subset) {
            subset->del();
            delete subset;
          });
  auto edge_map = CorrelationClustererEdgeMap{};
  auto new_moved_subset =
      gbbs::edgeMap(*current_graph, *(local_moved_subset.get()), edge_map);
  return std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
      new gbbs::vertexSubset(std::move(new_moved_subset)),
      [](gbbs::vertexSubset* subset) {
        subset->del();
        delete subset;
      });
}

bool IterateBestMoves(int num_inner_iterations, const ClustererConfig& clusterer_config,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
  ClusteringHelper* helper, CorrelationClustererSubclustering& subclustering) {
  const auto num_nodes = current_graph->n;
  bool moved = false;
  bool local_moved = true;
  auto moved_subset = std::unique_ptr<gbbs::vertexSubset, void (*)(gbbs::vertexSubset*)>(
    new gbbs::vertexSubset(num_nodes, num_nodes,
    gbbs::sequence<bool>(num_nodes, true).to_array()),
    [](gbbs::vertexSubset* subset) {
      subset->del();
      delete subset;
    });

  // Iterate over best moves
  int local_iter = 0;
  for (local_iter = 0; local_iter < num_inner_iterations && local_moved; ++local_iter) {
    auto new_moved_subset =
      BestMovesForVertexSubset(current_graph, num_nodes, moved_subset.get(),
                              helper, clusterer_config, subclustering);
    moved_subset.swap(new_moved_subset);
    local_moved = !moved_subset->isEmpty();
    moved |= local_moved;
  }
  return moved;
}

}  // namespace

template <template <class inner_wgh> class vtx_type, class wgh_type,
          typename P,
          typename std::enable_if<
              std::is_same<vtx_type<wgh_type>, gbbs::symmetric_vertex<wgh_type>>::value,
              int>::type = 0>
static inline gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, wgh_type> filterGraph(
    gbbs::symmetric_ptr_graph<vtx_type, wgh_type>& G, P& pred) {
  auto[newN, newM, newVData, newEdges] = gbbs::filter_graph<vtx_type, wgh_type>(G, pred);
  assert(newN == G.num_vertices());
  auto out_vdata = pbbs::new_array_no_init<gbbs::symmetric_vertex<float>>(newN);
  pbbs::parallel_for(0, newN, [&] (size_t i) {
    auto offset = (i == newN - 1) ? newM : newVData[i+1].offset;
    out_vdata[i].degree = offset-newVData[i].offset;
    out_vdata[i].neighbors = newEdges + newVData[i].offset;
  });
  pbbslib::free_arrays(newVData);
  return gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, wgh_type>(
      newN, newM, out_vdata,
      [newVData = out_vdata, newEdges = newEdges]() {
        pbbslib::free_arrays(newVData, newEdges);
      });
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty);
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph) const {
  std::vector<double> empty;
  return RefineClusters(clusterer_config, initial_clustering, empty, graph);
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights, double resolution) const {
    return RefineClusters(clusterer_config, initial_clustering, node_weights, graph_.Graph(), resolution);
}

absl::Status ParallelCorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering) const {
  std::vector<double> empty;
  return RefineClusters_subroutine(clusterer_config, initial_clustering, empty);
}

absl::Status ParallelCorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph) const {
  std::vector<double> empty;
  return RefineClusters_subroutine(clusterer_config, initial_clustering, empty, graph);
}

absl::Status ParallelCorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights) const {
    return RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph_.Graph());
}

absl::StatusOr<GraphWithWeights> CompressSubclusters(const ClustererConfig& clusterer_config, 
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph,
  std::vector<gbbs::uintE>& local_cluster_ids, 
  ClusteringHelper* helper,
  CorrelationClustererSubclustering& subclustering,
  InMemoryClusterer::Clustering& new_clustering) {

  auto get_clusters = [&](gbbs::uintE i) -> gbbs::uintE { return i; };
  std::vector<std::vector<gbbs::uintE>> curr_clustering =
    parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
      local_cluster_ids, get_clusters, current_graph->n);
  std::vector<ClusteringHelper::ClusterId> subcluster_ids(current_graph->n, 0);
 
  auto next_ids = gbbs::sequence<std::size_t>(curr_clustering.size() + 1, [](std::size_t i){ return 0; });


  pbbs::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
    next_ids[i] = ComputeSubcluster(subcluster_ids, 0, curr_clustering[i], current_graph, 
      subclustering, local_cluster_ids, clusterer_config);
  });

  auto total_next_id = pbbslib::scan_add_inplace(next_ids);
  pbbs::parallel_for(0, curr_clustering.size(), [&](std::size_t i) {
    pbbs::parallel_for (0, curr_clustering[i].size(), [&](std::size_t j){
      subcluster_ids[curr_clustering[i][j]] += next_ids[i];
    });
  });

  // Create new local clusters (subcluster)
  pbbs::parallel_for(1, curr_clustering.size() + 1, [&](std::size_t i) {
    for (std::size_t j = next_ids[i-1]; j < next_ids[i]; j++) {
      local_cluster_ids[j] = i-1;
    }
  });
  new_clustering = parallel::OutputIndicesById<ClusteringHelper::ClusterId, gbbs::uintE>(
    local_cluster_ids, get_clusters, total_next_id);

  return CompressGraph(*current_graph, subcluster_ids, helper);
} 

absl::Status ParallelCorrelationClusterer::RefineClusters_subroutine(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights,
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph) const {
  pbbs::timer t; t.start();

  const auto& config = clusterer_config.correlation_clusterer_config();
  // Set number of iterations based on clustering method
  int num_iterations = 0;
  int num_inner_iterations = 0;
  switch (config.clustering_moves_method()) {
    case CorrelationClustererConfig::DEFAULT_CLUSTER_MOVES:
      num_iterations = 1;
      num_inner_iterations =
          config.num_iterations() > 0 ? config.num_iterations() : 20;
      break;
    case CorrelationClustererConfig::LOUVAIN:
      num_iterations = config.louvain_config().num_iterations() > 0
                           ? config.louvain_config().num_iterations()
                           : 10;
      num_inner_iterations =
          config.louvain_config().num_inner_iterations() > 0
              ? config.louvain_config().num_inner_iterations()
              : 10;
      break;
    default:
      return absl::UnimplementedError(
          "Correlation clustering moves must be DEFAULT_CLUSTER_MOVES or "
          "LOUVAIN.");
  }

  // Initialize refinement data structure
  CorrelationClustererRefine refine{};
  if (config.refine()) {
    using H = std::unique_ptr<ClusteringHelper>;
    using G = std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>;
    refine.recurse_helpers = gbbs::sequence<H>(num_iterations, [](std::size_t i){return H(nullptr);});
    refine.recurse_graphs = gbbs::sequence<G>(num_iterations, [](std::size_t i){return G(nullptr);});
  }

  std::unique_ptr<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>
      compressed_graph;

  // Initialize clustering helper
  auto helper = node_weights.empty() ? absl::make_unique<ClusteringHelper>(
      graph->n, clusterer_config, *initial_clustering) :
      absl::make_unique<ClusteringHelper>(
      graph->n, clusterer_config, node_weights, *initial_clustering);

  std::vector<gbbs::uintE> cluster_ids(graph->n);
  std::vector<gbbs::uintE> local_cluster_ids(graph->n);
  pbbs::parallel_for(0, graph->n, [&](std::size_t i) {
    cluster_ids[i] = i;
  });

  int iter = 0;
  for (iter = 0; iter < num_iterations; ++iter) {
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
        (iter == 0) ? graph : compressed_graph.get();

    // Initialize subclustering data structure
    CorrelationClustererSubclustering subclustering(clusterer_config, current_graph);

    bool moved = IterateBestMoves(num_inner_iterations, clusterer_config, current_graph,
      helper.get(), subclustering);

    // If no moves can be made at all, exit
    if (!moved) {
      iter--;
      break;
    }

    // Compress cluster ids in initial_helper based on helper
    if (!config.refine()) cluster_ids = FlattenClustering(cluster_ids, helper->ClusterIds());
    else if (config.refine() && iter == num_iterations - 1) {
      refine.recurse_helpers[iter] = std::move(helper);
      refine.recurse_graphs[iter] = (iter == 0) ? nullptr : std::move(compressed_graph);
    } 

    if (iter == num_iterations - 1) break;

    // Compress graph
    GraphWithWeights new_compressed_graph;
    InMemoryClusterer::Clustering new_clustering{};
    pbbs::parallel_for(0, current_graph->n, [&](std::size_t i) {
        local_cluster_ids[i] = helper->ClusterIds()[i];
    });
    if (config.subclustering_method() != CorrelationClustererConfig::NONE_SUBCLUSTERING) {
      ASSIGN_OR_RETURN(new_compressed_graph,
        CompressSubclusters(clusterer_config, current_graph, local_cluster_ids, helper.get(),
        subclustering, new_clustering));
    } else {
      ASSIGN_OR_RETURN(
          new_compressed_graph,
          CompressGraph(*current_graph, local_cluster_ids, helper.get()));
      // Create new local clusters
      pbbs::parallel_for(0, new_compressed_graph.graph->n,
                         [&](std::size_t i) { local_cluster_ids[i] = i; });
    }

    compressed_graph.swap(new_compressed_graph.graph);
    if (config.refine()) {
      refine.recurse_helpers[iter] = std::move(helper);
      refine.recurse_graphs[iter] = std::move(new_compressed_graph.graph);
    } else if (new_compressed_graph.graph) new_compressed_graph.graph->del();

    helper = absl::make_unique<ClusteringHelper>(
        compressed_graph->n, clusterer_config,
        new_compressed_graph.node_weights, new_clustering);
  }

  // Refine clusters up the stack
  if (config.refine() && iter > 0) {
    auto get_clusters = [&](NodeId i) -> NodeId { return i; };
    for (int i = iter - 1; i >= 0; i--) {
      gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* current_graph =
        (i == 0) ? graph : refine.recurse_graphs[i].get();
      auto flatten_cluster_ids = FlattenClustering(refine.recurse_helpers[i]->ClusterIds(),
          refine.recurse_helpers[i+1]->ClusterIds());
      auto flatten_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
          flatten_cluster_ids,
          get_clusters, 
          flatten_cluster_ids.size());
      refine.recurse_helpers[i]->ResetClustering(flatten_clustering);

      CorrelationClustererSubclustering subclustering(clusterer_config, current_graph);
      IterateBestMoves(num_inner_iterations, clusterer_config, current_graph,
        refine.recurse_helpers[i].get(), subclustering);
    }
    cluster_ids = refine.recurse_helpers[0]->ClusterIds();
  }

  t.stop(); t.reportTotal("Cluster Time: ");

  if (compressed_graph) compressed_graph->del();

  auto get_clusters = [&](NodeId i) -> NodeId { return i; };

  *initial_clustering = parallel::OutputIndicesById<ClusterId, NodeId>(
      cluster_ids, get_clusters, cluster_ids.size());

  // Compute final objective
  auto helper2 = node_weights.empty() ? absl::make_unique<ClusteringHelper>(
      graph->n, clusterer_config, *initial_clustering) :
      absl::make_unique<ClusteringHelper>(
      graph->n, clusterer_config, node_weights, *initial_clustering);
  double max_objective = helper2->ComputeObjective(*graph);
  std::cout << "Objective: " << std::setprecision(17) << max_objective << std::endl;

  std::cout << "Number of Clusters: " << initial_clustering->size() << std::endl;

  return absl::OkStatus();
}

absl::Status ParallelCorrelationClusterer::RefineClusters(
    const ClustererConfig& clusterer_config,
    InMemoryClusterer::Clustering* initial_clustering,
    std::vector<double> node_weights,
    gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>* graph, double original_resolution) const {
  const auto& config = clusterer_config.correlation_clusterer_config();
  RETURN_IF_ERROR(RefineClusters_subroutine(clusterer_config, initial_clustering, node_weights, graph));
  return absl::OkStatus();
}

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelCorrelationClusterer::Cluster(
    const ClustererConfig& clusterer_config) const {
  InMemoryClusterer::Clustering clustering(graph_.Graph()->n);
  // Create all-singletons initial clustering
  pbbs::parallel_for(0, graph_.Graph()->n, [&](std::size_t i) {
    clustering[i] = {static_cast<gbbs::uintE>(i)};
  });

  RETURN_IF_ERROR(RefineClusters(clusterer_config, &clustering));

  return clustering;
}

}  // namespace in_memory
}  // namespace research_graph
