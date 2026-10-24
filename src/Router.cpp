#include "Router.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>

namespace {
constexpr std::int64_t kInfinity = std::numeric_limits<std::int64_t>::max();
}

Router::Router(const CityGraph &graph)
    : graph_(graph), dist_(graph.nodeCount(), kInfinity),
      prev_(graph.nodeCount(), kNoNode) {}

Route Router::shortestPath(NodeIndex from, NodeIndex to) {
  Route route;
  if (from >= graph_.nodeCount() || to >= graph_.nodeCount()) {
    return route;
  }

  std::fill(dist_.begin(), dist_.end(), kInfinity);
  std::fill(prev_.begin(), prev_.end(), kNoNode);

  using Entry = std::pair<std::int64_t, NodeIndex>;
  std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;

  dist_[from] = 0;
  queue.emplace(0, from);

  while (!queue.empty()) {
    const Entry top = queue.top();
    queue.pop();
    const std::int64_t d = top.first;
    const NodeIndex u = top.second;

    if (d > dist_[u]) {
      continue;
    }
    if (u == to) {
      break;
    }

    for (const Edge &e : graph_.edgesFrom(u)) {
      const std::int64_t candidate = d + e.travelMs;
      if (candidate < dist_[e.to]) {
        dist_[e.to] = candidate;
        prev_[e.to] = u;
        queue.emplace(candidate, e.to);
      }
    }
  }

  if (dist_[to] == kInfinity) {
    return route;
  }

  route.found = true;
  route.travelMs = dist_[to];
  for (NodeIndex at = to; at != kNoNode; at = prev_[at]) {
    route.nodes.push_back(at);
    if (at == from) {
      break;
    }
  }
  std::reverse(route.nodes.begin(), route.nodes.end());
  return route;
}
