#include "clustering/clusterers/parallel-modularity-clusterer.h"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
#include <iomanip>
#include <iostream>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/parallel-correlation-clusterer.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

namespace research_graph {
namespace in_memory {

double ComputeModularity(
  InMemoryClusterer::Clustering& initial_clustering,
  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>& graph,
  double total_edge_weight, std::vector<gbbs::uintE>& cluster_ids,
  double resolution){
  total_edge_weight = 0;
  double modularity = 0;
  for (std::size_t i = 0; i < graph.n; i++) {
    auto vtx = graph.get_vertex(i);
    auto nbhrs = vtx.getOutNeighbors();
    double deg_i = vtx.getOutDegree();
    for (std::size_t j = 0; j < deg_i; j++) {
      total_edge_weight++;
      auto nbhr = std::get<0>(nbhrs[j]);
      if (cluster_ids[i] == cluster_ids[nbhr]) {
        modularity++;
      }
    }
  }
  for (std::size_t i = 0; i < initial_clustering.size(); i++) {
    double degree = 0;
    for (std::size_t j = 0; j < initial_clustering[i].size(); j++) {
      auto vtx_id = initial_clustering[i][j];
      auto vtx = graph.get_vertex(vtx_id);
      degree += vtx.getOutDegree();
    }
    modularity -= (resolution * degree * degree) / (total_edge_weight);
  }
  modularity = modularity / (total_edge_weight);
  return modularity;
}

absl::Status ParallelModularityClusterer::RefineClusters(
    const ClustererConfig& clusterer_config2,
    InMemoryClusterer::Clustering* initial_clustering) const {

  pbbs::timer t; t.start();
  const auto& config = clusterer_config2.correlation_clusterer_config();
  auto modularity_config= ComputeModularityConfig(graph_.Graph(), config.resolution());

  ClustererConfig clusterer_config;
  clusterer_config.CopyFrom(clusterer_config2);
  clusterer_config.mutable_correlation_clusterer_config()->set_resolution(std::get<1>(modularity_config));
  clusterer_config.mutable_correlation_clusterer_config()->set_edge_weight_offset(0);
  t.stop(); t.reportTotal("Modularity Config Time: ");
  auto status = ParallelCorrelationClusterer::RefineClusters(clusterer_config, initial_clustering,
    std::get<0>(modularity_config), config.resolution());

  std::vector<gbbs::uintE> cluster_ids(graph_.Graph()->n);
  for (std::size_t i = 0; i < initial_clustering->size(); i++) {
    for (std::size_t j = 0; j < ((*initial_clustering)[i]).size(); j++) {
      cluster_ids[(*initial_clustering)[i][j]] = i;
    }
  }

  double modularity = ComputeModularity(*initial_clustering,
    *graph_.Graph(), std::get<2>(modularity_config), cluster_ids, config.resolution());
  std::cout << "Modularity: " << std::setprecision(17) << modularity << std::endl;

  return absl::OkStatus();
}

absl::StatusOr<InMemoryClusterer::Clustering>
ParallelModularityClusterer::Cluster(
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