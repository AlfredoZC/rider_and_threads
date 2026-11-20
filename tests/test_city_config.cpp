#include "CityGraph.h"
#include "Router.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

// No usa assert: con -DNDEBUG desapareceria y el test pasaria sin comprobar nada.
#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

// Cuantos nodos alcanza `src` siguiendo las aristas del grafo (BFS), o llegan a `src` si reverse.
std::size_t bfsCount(const std::vector<std::vector<NodeIndex>>& adj, NodeIndex src) {
    std::vector<char> seen(adj.size(), 0);
    std::queue<NodeIndex> q;
    seen[src] = 1;
    q.push(src);
    std::size_t count = 1;
    while (!q.empty()) {
        const NodeIndex u = q.front();
        q.pop();
        for (NodeIndex v : adj[u]) {
            if (!seen[v]) {
                seen[v] = 1;
                ++count;
                q.push(v);
            }
        }
    }
    return count;
}

}

int main() {
    const Config cfg = loadConfig("config/ciudad.json");
    const CityGraph g(cfg);

    std::cout << "  nodos=" << cfg.nodes.size() << " calles=" << cfg.streets.size()
              << " aristas=" << g.edgeCount() << " restaurantes=" << cfg.restaurants.size()
              << " couriers=" << cfg.couriers << "\n";

    CHECK(cfg.nodes.size() >= 3000 && cfg.nodes.size() <= 5000);
    CHECK(g.nodeCount() == cfg.nodes.size());
    CHECK(cfg.couriers >= 2000 && cfg.couriers <= 5000);
    CHECK(cfg.restaurants.size() >= 20);

    // los ids son unicos y todo lo que se referencia existe
    std::unordered_set<std::string> ids;
    for (const NodeConfig& n : cfg.nodes) {
        CHECK(ids.insert(n.id).second);
        CHECK(n.lat <= cfg.bounds.north && n.lat >= cfg.bounds.south);
        CHECK(n.lon <= cfg.bounds.east && n.lon >= cfg.bounds.west);
    }
    for (const StreetConfig& s : cfg.streets) {
        CHECK(ids.count(s.from) == 1 && ids.count(s.to) == 1);
        CHECK(s.from != s.to);
    }
    CHECK(ids.count(cfg.startNode) == 1);
    std::unordered_set<std::string> restaurantIds, restaurantNodes;
    for (const RestaurantConfig& r : cfg.restaurants) {
        CHECK(ids.count(r.node) == 1);
        CHECK(restaurantIds.insert(r.id).second);
        restaurantNodes.insert(r.node);
    }
    CHECK(restaurantNodes.size() == cfg.restaurants.size());
    CHECK(std::filesystem::exists(cfg.imagePath));
    std::cout << "  ids unicos, calles/restaurantes/startNode apuntan a nodos existentes, imagen presente\n";

    // pedidos: la cola admite cientos de pendientes
    CHECK(cfg.maxPending >= 100);
    CHECK(cfg.burstMax >= 2);
    CHECK(cfg.meanIntervalMs >= 1);

    // alcanzabilidad con el Router real (Dijkstra) desde el startNode y desde restaurantes
    Router router(g);
    const std::size_t n = g.nodeCount();
    // Dijkstra punto a punto desde `src` hacia los nodos 0, stride, 2*stride...
    auto reachableFrom = [&](NodeIndex src, std::size_t stride, std::size_t& asked) {
        std::size_t ok = 0;
        for (std::size_t to = 0; to < n; to += stride) {
            ++asked;
            if (router.shortestPath(src, static_cast<NodeIndex>(to)).found) { ++ok; }
        }
        return ok;
    };

    // startNode: contra TODOS los nodos
    const NodeIndex start = g.indexOf(cfg.startNode);
    std::size_t asked = 0;
    const std::size_t fromStart = reachableFrom(start, 1, asked);
    std::cout << "  Router desde startNode " << cfg.startNode << ": " << fromStart << "/" << asked
              << " nodos alcanzables (" << 100.0 * fromStart / asked << "%)\n";
    CHECK(fromStart * 100 >= asked * 99);

    // restaurantes: primero, ultimo y 2 intermedios, contra 1 de cada 5 nodos (el Router en -O0 es
    // lento; la cobertura completa de los 150 restaurantes la da el BFS de mas abajo)
    const std::size_t rn = cfg.restaurants.size();
    const std::size_t sample[] = {0, rn / 3, 2 * rn / 3, rn - 1};
    std::size_t pairsOk = 0, pairsTotal = 0, worst = n;
    for (std::size_t i : sample) {
        std::size_t askedHere = 0;
        const std::size_t ok = reachableFrom(g.indexOf(cfg.restaurants[i].node), 5, askedHere);
        pairsOk += ok;
        pairsTotal += askedHere;
        worst = std::min(worst, ok * 100 / askedHere);
    }
    std::cout << "  Router desde 4 restaurantes muestreados: " << pairsOk << "/" << pairsTotal
              << " pares (" << 100.0 * pairsOk / pairsTotal << "%), peor restaurante " << worst << "%\n";
    CHECK(pairsOk * 100 >= pairsTotal * 99);

    // los 150 restaurantes (BFS directo e inverso) y el startNode, sobre el grafo dirigido
    std::vector<std::vector<NodeIndex>> fwd(n), rev(n);
    for (NodeIndex u = 0; u < n; ++u) {
        for (const Edge& e : g.edgesFrom(u)) {
            fwd[u].push_back(e.to);
            rev[e.to].push_back(u);
        }
    }
    std::size_t allOut = 0, allIn = 0, allPairs = 0;
    std::vector<NodeIndex> sources{start};
    for (const RestaurantConfig& r : cfg.restaurants) { sources.push_back(g.indexOf(r.node)); }
    for (NodeIndex s : sources) {
        allOut += bfsCount(fwd, s);   // desde s se llega a ...
        allIn += bfsCount(rev, s);    // a s se llega desde ...
        allPairs += n;
    }
    std::cout << "  todos los restaurantes + startNode (" << sources.size() << " orígenes), BFS: ida "
              << allOut << "/" << allPairs << " (" << 100.0 * allOut / allPairs << "%), vuelta " << allIn
              << "/" << allPairs << " (" << 100.0 * allIn / allPairs << "%)\n";
    CHECK(allOut * 100 >= allPairs * 99);
    CHECK(allIn * 100 >= allPairs * 99);

    std::cout << "  city_config ok\n";
}
