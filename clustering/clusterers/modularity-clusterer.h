#ifndef PARALLEL_CLUSTERING_CLUSTERERS_MODULARITY_CLUSTERER_H_
#define PARALLEL_CLUSTERING_CLUSTERERS_MODULARITY_CLUSTERER_H_

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "clustering/clusterers/correlation-clusterer.h"
#include "clustering/config.pb.h"
#include "clustering/gbbs-graph.h"
#include "clustering/in-memory-clusterer.h"
#include "parallel/parallel-graph-utils.h"
#include "clustering/status_macros.h"

namespace research_graph {
namespace in_memory {

// A local-search based clusterer optimizing the correlation clustering
// objective. See comment above CorrelationClustererConfig in
// ../config.proto for more. This uses the CorrelationClustererConfig proto.
// Also, note that the input graph is required to be undirected.
class ModularityClusterer : public CorrelationClusterer {
 public:
  using ClusterId = gbbs::uintE;

  absl::StatusOr<Clustering> Cluster(
      const ClustererConfig& config) const override;

  // initial_clustering must include every node in the range
  // [0, MutableGraph().NumNodes()) exactly once.
  absl::Status RefineClusters(const ClustererConfig& clusterer_config,
                              Clustering* initial_clustering) const override;
};

}  // namespace in_memory
}  // namespace research_graph

#endif  // PARALLEL_CLUSTERING_CLUSTERERS_MODULARITY_CLUSTERER_H_