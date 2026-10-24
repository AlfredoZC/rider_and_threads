#include "Router.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr std::int64_t kInf = std::numeric_limits<std::int64_t>::max();

std::vector<std::int64_t> bellmanFord(const CityGraph& g, NodeIndex source) {
    std::vector<std::int64_t> dist(g.nodeCount(), kInf);
    dist[source] = 0;
    for (std::size_t pass = 0; pass + 1 < g.nodeCount(); ++pass) {
        bool changed = false;
        for (NodeIndex u = 0; u < g.nodeCount(); ++u) {
            if (dist[u] == kInf) { continue; }
            for (const Edge& e : g.edgesFrom(u)) {
                if (dist[u] + e.travelMs < dist[e.to]) {
                    dist[e.to] = dist[u] + e.travelMs;
                    changed = true;
                }
            }
        }
        if (!changed) { break; }
    }
    return dist;
}

void checkRouteIsWalkable(const CityGraph& g, const Route& r, NodeIndex from, NodeIndex to) {
    assert(r.found);
    assert(!r.nodes.empty());
    assert(r.nodes.front() == from);
    assert(r.nodes.back() == to);

    std::int64_t sum = 0;
    for (std::size_t i = 0; i + 1 < r.nodes.size(); ++i) {
        bool edgeExists = false;
        for (const Edge& e : g.edgesFrom(r.nodes[i])) {
            if (e.to == r.nodes[i + 1]) {
                edgeExists = true;
                sum += e.travelMs;
                break;
            }
        }
        assert(edgeExists);
    }
    assert(sum == r.travelMs);
}

}

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph g(cfg);
    Router router(g);

    const NodeIndex start = g.indexOf(cfg.startNode);
    assert(start != kNoNode);

    const Route self = router.shortestPath(start, start);
    assert(self.found);
    assert(self.travelMs == 0);
    assert(self.nodes.size() == 1 && self.nodes[0] == start);

    const Route bad = router.shortestPath(start, static_cast<NodeIndex>(g.nodeCount()));
    assert(!bad.found);
    assert(!router.shortestPath(kNoNode, start).found);

    std::size_t reachable = 0, unreachable = 0;
    std::int64_t worst = 0;
    for (NodeIndex to = 0; to < g.nodeCount(); ++to) {
        const Route r = router.shortestPath(start, to);
        if (!r.found) { ++unreachable; continue; }
        ++reachable;
        worst = std::max(worst, r.travelMs);
        checkRouteIsWalkable(g, r, start, to);
    }
    std::cout << "  desde " << cfg.startNode << ": alcanzables=" << reachable
              << " inalcanzables=" << unreachable << "\n";
    std::cout << "  viaje mas largo=" << worst / 1000 << "s simulados\n";

    for (const RestaurantConfig& rest : cfg.restaurants) {
        const Route r = router.shortestPath(start, g.indexOf(rest.node));
        assert(r.found);
        checkRouteIsWalkable(g, r, start, g.indexOf(rest.node));
    }
    std::cout << "  los " << cfg.restaurants.size() << " restaurantes son alcanzables\n";

    const NodeIndex sources[] = {start, g.indexOf(cfg.restaurants.front().node),
                                 g.indexOf(cfg.restaurants.back().node)};
    for (NodeIndex src : sources) {
        const std::vector<std::int64_t> reference = bellmanFord(g, src);
        for (NodeIndex to = 0; to < g.nodeCount(); ++to) {
            const Route r = router.shortestPath(src, to);
            if (reference[to] == kInf) {
                assert(!r.found);
            } else {
                assert(r.found);
                assert(r.travelMs == reference[to]);
            }
        }
    }
    std::cout << "  distancias identicas a Bellman-Ford desde 3 origenes\n";

    const Route a = router.shortestPath(start, g.indexOf(cfg.restaurants.front().node));
    for (int i = 0; i < 50; ++i) { router.shortestPath(start, static_cast<NodeIndex>(i)); }
    const Route b = router.shortestPath(start, g.indexOf(cfg.restaurants.front().node));
    assert(a.travelMs == b.travelMs && a.nodes == b.nodes);
    std::cout << "  los buffers reutilizados no contaminan consultas posteriores\n";

    std::cout << "  router ok\n";
}
