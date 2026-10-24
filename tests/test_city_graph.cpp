#include "CityGraph.h"

#include <algorithm>
#include <cassert>
#include <iostream>

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph g(cfg);

    std::cout << "  nodos=" << g.nodeCount() << " aristas=" << g.edgeCount() << "\n";
    assert(g.nodeCount() == cfg.nodes.size());

    std::size_t oneWay = 0;
    for (const StreetConfig& s : cfg.streets) {
        if (s.oneWay) { ++oneWay; }
    }
    assert(g.edgeCount() == (cfg.streets.size() - oneWay) * 2 + oneWay);

    assert(g.indexOf("no_existe") == kNoNode);
    assert(g.indexOf(cfg.startNode) != kNoNode);
    assert(g.idOf(g.indexOf(cfg.startNode)) == cfg.startNode);
    for (const RestaurantConfig& r : cfg.restaurants) {
        assert(g.indexOf(r.node) != kNoNode);
    }

    std::size_t checkedOneWay = 0;
    for (const StreetConfig& s : cfg.streets) {
        if (!s.oneWay) { continue; }
        const NodeIndex a = g.indexOf(s.from);
        const NodeIndex b = g.indexOf(s.to);
        bool aToB = false, bToA = false;
        for (const Edge& e : g.edgesFrom(a)) { if (e.to == b) { aToB = true; } }
        for (const Edge& e : g.edgesFrom(b)) { if (e.to == a) { bToA = true; } }
        assert(aToB);
        bool reverseInConfig = false;
        for (const StreetConfig& t : cfg.streets) {
            if (t.from == s.to && t.to == s.from) { reverseInConfig = true; }
            if (!t.oneWay && ((t.from == s.to && t.to == s.from) ||
                              (t.from == s.from && t.to == s.to))) { reverseInConfig = true; }
        }
        if (!reverseInConfig) {
            assert(!bToA);
            ++checkedOneWay;
        }
    }
    std::cout << "  sentidos unicos verificados=" << checkedOneWay << "\n";
    assert(checkedOneWay > 0);

    double minM = 1e9, maxM = 0.0;
    std::size_t isolated = 0, total = 0;
    const double msPerMeter = 3600.0 / cfg.speedKmh;
    for (NodeIndex i = 0; i < g.nodeCount(); ++i) {
        if (g.edgesFrom(i).empty()) { ++isolated; }
        for (const Edge& e : g.edgesFrom(i)) {
            assert(e.to != i);
            assert(e.to < g.nodeCount());
            assert(e.meters > 0.0);
            assert(e.travelMs > 0);
            assert(e.travelMs == static_cast<std::int64_t>(e.meters * msPerMeter));
            minM = std::min(minM, e.meters);
            maxM = std::max(maxM, e.meters);
            ++total;
        }
    }
    assert(total == g.edgeCount());

    std::cout << "  arista mas corta=" << minM << "m  mas larga=" << maxM << "m\n";
    std::cout << "  nodos sin salida=" << isolated << "\n";
    std::cout << "  graph ok\n";
}
