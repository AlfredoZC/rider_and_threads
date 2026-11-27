#include "Router.h"

#include <algorithm>
#include <queue>
#include <utility>

Router::Router(const CityGraph &graph)
    : graph_(graph), dist_(graph.nodeCount(), kInfinityMs),
      prev_(graph.nodeCount(), kNoNode) {}

void Router::runDijkstra(NodeIndex from, NodeIndex stopAt) {
  std::fill(dist_.begin(), dist_.end(), kInfinityMs);
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
    if (u == stopAt) {
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
}

Route Router::shortestPath(NodeIndex from, NodeIndex to) {
  Route route;
  if (from >= graph_.nodeCount() || to >= graph_.nodeCount()) {
    return route;
  }

  runDijkstra(from, to);

  if (dist_[to] == kInfinityMs) {
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

const std::vector<std::int64_t> &Router::travelTimesFrom(NodeIndex from) {
  if (from >= graph_.nodeCount()) {
    static const std::vector<std::int64_t> kEmpty;
    return kEmpty;
  }
  runDijkstra(from, kNoNode);
  return dist_;
}
