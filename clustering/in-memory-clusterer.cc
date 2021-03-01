#include "clustering/in-memory-clusterer.h"

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"

namespace research_graph {
namespace in_memory {

absl::Status InMemoryClusterer::Graph::FinishImport() {
  return absl::OkStatus();
}

std::string InMemoryClusterer::StringId(NodeId id) const {
  if (node_id_map_ == nullptr) {
    return absl::StrCat(id);
  } else if (id >= 0 && id < node_id_map_->size()) {
    return (*node_id_map_)[id];
  } else {
    return absl::StrCat("missing-id-", id);
  }
}

}  // namespace in_memory
}  // namespace research_graph
