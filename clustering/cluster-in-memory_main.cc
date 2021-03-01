#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <iomanip>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

#include "clustering/clusterers/parallel-affinity.h"
#include "clustering/clusterers/parallel-correlation-clusterer.h"
#include "clustering/clusterers/parallel-modularity-clusterer.h"
#include "clustering/clusterers/correlation-clusterer.h"
#include "clustering/clusterers/modularity-clusterer.h"
#include "clustering/config.pb.h"
#include "clustering/in-memory-clusterer.h"
#include "clustering/status_macros.h"
#include "external/gbbs/gbbs/edge_map_blocked.h"
#include "google/protobuf/text_format.h"

ABSL_FLAG(std::string, clusterer_name, "",
          "Name of a clusterer (ParallelAffinityClusterer or "
          "ParallelCorrelationClusterer or ParallelModularityClusterer.");

ABSL_FLAG(std::string, clusterer_config, "",
          "Text-format research_graph.in_memory.ClustererConfig proto.");

ABSL_FLAG(std::string, input_graph, "",
          "Input file pattern of a graph. Should be in edge list format "
          "(SNAP format).");

ABSL_FLAG(std::string, output_clustering, "",
          "Output filename of a clustering.");

ABSL_FLAG(bool, is_symmetric_graph, true,
          "Without this flag, the program expects the edge list to represent "
          "an undirected graph (each edge needs to be given in both "
          "directions). With this flag, the program symmetrizes the graph.");

ABSL_FLAG(bool, float_weighted, false,
          "Use this flag if the edge list is weighted with 32-bit floats. If "
          "this flag is not set, then the graph is assumed to be unweighted, "
          "and edge weights are automatically set to 1.");

ABSL_FLAG(std::string, input_communities, "",
          "Input file pattern of a list of communities; tab separated nodes, lines separating communities.");

namespace research_graph {
namespace in_memory {
namespace {

void PrintTime(std::chrono::steady_clock::time_point begin,
  std::chrono::steady_clock::time_point end, const std::string& input){
  std::cout << input << " Time: " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

double DoubleFromWeight(pbbslib::empty weight) { return 1; }
double DoubleFromWeight(double weight) { return weight; }

float FloatFromWeight(float weight) { return weight; }
float FloatFromWeight(pbbslib::empty weight) { return 1; }

template <class Graph>
absl::Status GbbsGraphToInMemoryClustererGraph(InMemoryClusterer::Graph* graph,
                                               Graph& gbbs_graph) {
  using weight_type = typename Graph::weight_type;
  for (std::size_t i = 0; i < gbbs_graph.n; i++) {
    auto vertex = gbbs_graph.get_vertex(i);
    std::vector<std::pair<gbbs::uintE, double>> outgoing_edges(
        vertex.getOutDegree());
    gbbs::uintE index = 0;
    auto add_outgoing_edge = [&](gbbs::uintE, const gbbs::uintE neighbor,
                                 weight_type wgh) {
      outgoing_edges[index] = std::make_pair(
        static_cast<gbbs::uintE>(neighbor), DoubleFromWeight(wgh));
      index++;
    };
    vertex.mapOutNgh(i, add_outgoing_edge, false);
    InMemoryClusterer::Graph::AdjacencyList adjacency_list{
        static_cast<InMemoryClusterer::NodeId>(i), 1,
        std::move(outgoing_edges)};
    RETURN_IF_ERROR(graph->Import(adjacency_list));
  }
  RETURN_IF_ERROR(graph->FinishImport());
  return absl::OkStatus();
}

template <typename Weight>
absl::StatusOr<std::size_t> WriteEdgeListAsGraph(
    InMemoryClusterer::Graph* graph,
    const std::vector<gbbs::gbbs_io::Edge<Weight>>& edge_list,
    bool is_symmetric_graph) {
  std::size_t n = 0;
  if (is_symmetric_graph) {
    auto gbbs_graph{gbbs::gbbs_io::edge_list_to_symmetric_graph(edge_list)};
    n = gbbs_graph.n;
    auto status = GbbsGraphToInMemoryClustererGraph<
        gbbs::symmetric_graph<gbbs::symmetric_vertex, Weight>>(graph,
                                                               gbbs_graph);
    RETURN_IF_ERROR(status);
    gbbs_graph.del();
  } else {
    auto gbbs_graph{gbbs::gbbs_io::edge_list_to_asymmetric_graph(edge_list)};
    n = gbbs_graph.n;
    auto status = GbbsGraphToInMemoryClustererGraph<
        gbbs::asymmetric_graph<gbbs::asymmetric_vertex, Weight>>(graph,
                                                                 gbbs_graph);
    RETURN_IF_ERROR(status);
    gbbs_graph.del();
  }
  return n;
}

absl::Status WriteClustering(const char* filename,
                             InMemoryClusterer::Clustering clustering) {
  std::ofstream file{filename};
  if (!file.is_open()) {
    return absl::NotFoundError("Unable to open file.");
  }
  for (gbbs::uintE i = 0; i < clustering.size(); i++) {
    for (auto node_id : clustering[i]) {
      file << node_id << "\t";
    }
    file << std::endl;
  }
  return absl::OkStatus();
}

void split(const std::string &s, char delim, std::vector<gbbs::uintE> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(std::stoi(item));
  }
}

absl::Status ReadCommunities(const char* filename,
  std::vector<std::vector<gbbs::uintE>>& communities) {
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    return absl::NotFoundError("Unable to open file.");
  }
  std::string line;
  while (std::getline(infile, line)) {
    std::vector<gbbs::uintE> row_values;
    split(line, '\t', row_values);
    std::sort(row_values.begin(), row_values.end());
    communities.push_back(row_values);
  }
  return absl::OkStatus();
}

absl::Status CompareCommunities(const char* filename, InMemoryClusterer::Clustering clustering) {
  std::vector<std::vector<gbbs::uintE>> communities;
  ReadCommunities(filename, communities);

  gbbs::sequence<double> precision_vec(communities.size(), [](std::size_t i){return 0;});
  gbbs::sequence<double> recall_vec(communities.size(), [](std::size_t i){return 0;});
  pbbs::parallel_for(0, clustering.size(), [&](std::size_t i) {
    auto cluster = clustering[i];
    std::sort(cluster.begin(), cluster.end());
  });
  pbbs::parallel_for(0, communities.size(), [&](std::size_t j) {
    auto community = communities[j];
    std::vector<gbbs::uintE> intersect(community.size());
    std::size_t max_intersect = 0;
    std::size_t max_idx = 0;

    for (std::size_t i = 0; i < clustering.size(); i++) {
      auto cluster = clustering[i];
      auto it = std::set_intersection(cluster.begin(), cluster.end(), 
        community.begin(), community.end(), intersect.begin());
      std::size_t it_size = it - intersect.begin();
      if (it_size > max_intersect) {
        max_intersect = it_size;
        max_idx = i;
      }
    }
    precision_vec[j] = (double) max_intersect / (double) clustering[max_idx].size();
    recall_vec[j] = (communities[j].size() == 0) ? 0 : 
      (double) max_intersect / (double) communities[j].size();
  });
  double avg_precision = pbbslib::reduce_add(precision_vec);
  double avg_recall = pbbslib::reduce_add(recall_vec);
  avg_precision /= communities.size();
  avg_recall /= communities.size();
  std::cout << "Avg precision: " << std::setprecision(17) << avg_precision << std::endl;
  std::cout << "Avg recall: " << std::setprecision(17) << avg_recall << std::endl;

  std::size_t min_size = UINT_E_MAX;
  std::size_t max_size = 0;
  double avg_size = 0;
  for (std::size_t i = 0; i < clustering.size(); i++) {
    auto sz = clustering[i].size();
    assert(sz > 0);
    if (sz < min_size) min_size = sz;
    if (sz > max_size) max_size = sz;
    avg_size += sz;
  }
  avg_size /= clustering.size();
  std::cout << "Num communities: " << clustering.size() << std::endl;
  std::cout << "Min size: " << min_size << std::endl;
  std::cout << "Max size: " << max_size << std::endl;
  std::cout << "Avg size: " << avg_size << std::endl;
  return absl::OkStatus();
}

struct FakeGraph {
  std::size_t n;
};

template <class Graph>
gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float> CopyGraph(Graph& graph){
  using vertex_data = gbbs::symmetric_vertex<float>;
  using edge_type = std::tuple<gbbs::uintE, float>;
  auto vd = pbbs::new_array_no_init<vertex_data>(graph.n);
  auto ed = pbbs::new_array_no_init<edge_type>(graph.m);
  pbbs::parallel_for(0, graph.n, [&] (size_t i) {
    vd[i].degree = graph.v_data[i].degree;
    vd[i].neighbors = ed + graph.v_data[i].offset;
  });
  pbbs::parallel_for(0, graph.m, [&] (size_t i) {
    ed[i] = std::make_tuple(
        std::get<0>(graph.e0[i]), FloatFromWeight(std::get<1>(graph.e0[i])));
  });
  auto g = gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>(graph.n, graph.m, vd, [vd, ed] () {
      pbbs::free_array(vd);
      pbbs::free_array(ed);
  });
  return g;
}

absl::Status Main() {
  gbbs::pcm_init();

  std::string clusterer_name = absl::GetFlag(FLAGS_clusterer_name);

  ClustererConfig config;
  std::string clusterer_config = absl::GetFlag(FLAGS_clusterer_config);
  if (!google::protobuf::TextFormat::ParseFromString(clusterer_config,
                                                     &config)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot parse --clusterer_config as a text-format "
                        "research_graph.in_memory.ClustererConfig proto: %s",
                        clusterer_config));
  }

  std::unique_ptr<InMemoryClusterer> clusterer;
  if (clusterer_name == "ParallelAffinityClusterer") {
    clusterer.reset(new ParallelAffinityClusterer);
  } else if (clusterer_name == "ParallelCorrelationClusterer") {
    clusterer.reset(new ParallelCorrelationClusterer);
  } else if (clusterer_name == "ParallelModularityClusterer") {
    clusterer.reset(new ParallelModularityClusterer);
  } else if (clusterer_name == "CorrelationClusterer") {
    clusterer.reset(new CorrelationClusterer);
  } else if (clusterer_name == "ModularityClusterer") {
    clusterer.reset(new ModularityClusterer);
  }
  else {
    return absl::UnimplementedError(
      "ParallelAffinityClusterer, ParallelCorrelationClusterer, "
      "ParallelModularityClusterer, CorrelationClusterer, and "
      "ModularityClusterer are the only supported clusterers.");
  }
  auto begin_read = std::chrono::steady_clock::now();
  std::string input_file = absl::GetFlag(FLAGS_input_graph);

  bool float_weighted = absl::GetFlag(FLAGS_float_weighted);
  std::size_t n = 0;

  gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float> g;
  if (float_weighted) {
    auto G = gbbs::gbbs_io::read_weighted_symmetric_graph<float>(
            input_file.c_str(), false);
    gbbs::alloc_init(G);
    // Transform to pointer graph
    n = G.n;
    g = CopyGraph(G);
  } else {
    auto G = gbbs::gbbs_io::read_unweighted_symmetric_graph(input_file.c_str(), false);
    gbbs::alloc_init(G);
    // Transform to pointer graph
    n = G.n;
    g = CopyGraph(G);
  }
  clusterer->MutableGraph()->graph_ = absl::make_unique<gbbs::symmetric_ptr_graph<gbbs::symmetric_vertex, float>>(g);

  auto end_read = std::chrono::steady_clock::now();
  PrintTime(begin_read, end_read, "Read graph");

  // Must initialize the list allocator for GBBS, to support parallelism.
  // The list allocator seeds using the number of vertices in the input graph.

  InMemoryClusterer::Clustering clustering;
  ASSIGN_OR_RETURN(clustering, clusterer->Cluster(config));

  std::string input_communities = absl::GetFlag(FLAGS_input_communities);
  if (!input_communities.empty())
    CompareCommunities(input_communities.c_str(), clustering);

  gbbs::alloc_finish();

  std::string output_file = absl::GetFlag(FLAGS_output_clustering);
  if (!output_file.empty()) WriteClustering(output_file.c_str(), clustering);

  return absl::OkStatus();
}

}  // namespace
}  // namespace in_memory
}  // namespace research_graph

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  auto status = research_graph::in_memory::Main();
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return EXIT_FAILURE;
  }
}
