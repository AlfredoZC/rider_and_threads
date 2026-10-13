// Pruebas de Router (Dijkstra).
//
// Lo importante no es que devuelva "un" camino, sino que devuelva EL camino minimo y
// que ese camino sea transitable de verdad. Por eso:
//   1. se comprueba que cada paso del camino exista como arista en el sentido correcto
//      (si no, el courier estaria cruzando una manzana: defecto del §3),
//   2. se comprueba que la suma de los tramos sea exactamente el travelMs devuelto,
//   3. se comparan las distancias con Bellman-Ford, un algoritmo distinto y mas lento,
//      para verificar que de verdad son minimas y no solo plausibles.
#include "Router.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr std::int64_t kInf = std::numeric_limits<std::int64_t>::max();

// Distancias minimas desde "source" calculadas con Bellman-Ford: relaja todas las
// aristas V-1 veces. Es O(V*E) y mucho mas lento que Dijkstra, pero es una
// implementacion independiente, y por eso sirve como referencia.
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

// Verifica que el camino sea realmente recorrible y que los tiempos cuadren.
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
        // si falla, el camino salta entre nodos sin calle: el courier cruzaria una manzana
        assert(edgeExists);
    }
    assert(sum == r.travelMs);
}

}  // namespace

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph g(cfg);
    Router router(g);

    const NodeIndex start = g.indexOf(cfg.startNode);
    assert(start != kNoNode);

    // caso degenerado: de un nodo a si mismo
    const Route self = router.shortestPath(start, start);
    assert(self.found);
    assert(self.travelMs == 0);
    assert(self.nodes.size() == 1 && self.nodes[0] == start);

    // indices invalidos: no se cae, devuelve "no encontrado"
    const Route bad = router.shortestPath(start, static_cast<NodeIndex>(g.nodeCount()));
    assert(!bad.found);
    assert(!router.shortestPath(kNoNode, start).found);

    // caminos reales desde el nodo de inicio hacia todos los demas
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

    // los 6 restaurantes tienen que ser alcanzables desde el inicio
    for (const RestaurantConfig& rest : cfg.restaurants) {
        const Route r = router.shortestPath(start, g.indexOf(rest.node));
        assert(r.found);
        checkRouteIsWalkable(g, r, start, g.indexOf(rest.node));
    }
    std::cout << "  los " << cfg.restaurants.size() << " restaurantes son alcanzables\n";

    // minimalidad: contrastar con Bellman-Ford desde 3 origenes distintos
    const NodeIndex sources[] = {start, g.indexOf(cfg.restaurants.front().node),
                                 g.indexOf(cfg.restaurants.back().node)};
    for (NodeIndex src : sources) {
        const std::vector<std::int64_t> reference = bellmanFord(g, src);
        for (NodeIndex to = 0; to < g.nodeCount(); ++to) {
            const Route r = router.shortestPath(src, to);
            if (reference[to] == kInf) {
                assert(!r.found);              // si no hay camino, Dijkstra tampoco debe hallarlo
            } else {
                assert(r.found);
                assert(r.travelMs == reference[to]);   // y si lo hay, tiene que ser EL minimo
            }
        }
    }
    std::cout << "  distancias identicas a Bellman-Ford desde 3 origenes\n";

    // el mismo Router reutilizado muchas veces sigue dando lo mismo
    const Route a = router.shortestPath(start, g.indexOf(cfg.restaurants.front().node));
    for (int i = 0; i < 50; ++i) { router.shortestPath(start, static_cast<NodeIndex>(i)); }
    const Route b = router.shortestPath(start, g.indexOf(cfg.restaurants.front().node));
    assert(a.travelMs == b.travelMs && a.nodes == b.nodes);
    std::cout << "  los buffers reutilizados no contaminan consultas posteriores\n";

    std::cout << "  router ok\n";
}
