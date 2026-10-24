#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "Config.h"

using NodeIndex = std::uint32_t;

inline constexpr NodeIndex kNoNode = 0xFFFFFFFFu;

struct Edge {
  NodeIndex to = kNoNode;
  double meters = 0.0;
  std::int64_t travelMs = 0;
};

class CityGraph {
 public:
  explicit CityGraph(const Config& cfg);

  std::size_t nodeCount() const { return nodes_.size(); }
  std::size_t edgeCount() const { return edgeCount_; }

  NodeIndex indexOf(const std::string& id) const;

  const std::string& idOf(NodeIndex i) const { return nodes_[i].id; }
  double latOf(NodeIndex i) const { return nodes_[i].lat; }
  double lonOf(NodeIndex i) const { return nodes_[i].lon; }

  const std::vector<Edge>& edgesFrom(NodeIndex i) const { return adjacency_[i]; }

 private:
  struct Node {
    std::string id;
    double lat = 0.0;
    double lon = 0.0;
  };

  std::vector<Node> nodes_;
  std::unordered_map<std::string, NodeIndex> ids_;
  std::vector<std::vector<Edge>> adjacency_;
  std::size_t edgeCount_ = 0;
};
