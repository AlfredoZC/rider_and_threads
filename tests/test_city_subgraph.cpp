#include "Config.h"
#include "bench/CitySubgraph.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

// Conexidad debil (calles tratadas como no dirigidas) con union-find.
bool weaklyConnected(const Config& config) {
    std::map<std::string, std::size_t> nodeIndexById;
    for (std::size_t nodeIndex = 0; nodeIndex < config.nodes.size(); ++nodeIndex) {
        nodeIndexById[config.nodes[nodeIndex].id] = nodeIndex;
    }
    std::vector<std::size_t> parent(config.nodes.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto findRoot = [&](std::size_t node) {
        while (parent[node] != node) node = parent[node] = parent[parent[node]];
        return node;
    };
    for (const auto& street : config.streets) {
        parent[findRoot(nodeIndexById.at(street.from))] = findRoot(nodeIndexById.at(street.to));
    }
    std::set<std::size_t> roots;
    for (std::size_t nodeIndex = 0; nodeIndex < parent.size(); ++nodeIndex) roots.insert(findRoot(nodeIndex));
    return roots.size() == 1;
}

bool sameStreet(const StreetConfig& left, const StreetConfig& right) {
    return left.id == right.id && left.from == right.from && left.to == right.to && left.oneWay == right.oneWay;
}

}  // namespace

int main() {
    const Config city = loadConfig("config/ciudad.json");
    CHECK(city.nodes.size() == 3741);

    for (std::size_t nodeCount : {25u, 100u, 400u, 1000u, 2000u, 3741u}) {
        const Config region = bench::extractRegion(city, city.startNode, nodeCount);
        CHECK(region.nodes.size() == nodeCount);
        CHECK(weaklyConnected(region));
        // Contiene el nodo de partida.
        CHECK(std::any_of(region.nodes.begin(), region.nodes.end(),
                          [&](const NodeConfig& node) { return node.id == city.startNode; }));
        CHECK(region.startNode == city.startNode);

        // Calles: exactamente las del original con ambos extremos dentro, mismo sentido e id.
        std::set<std::string> regionNodeIds;
        for (const auto& node : region.nodes) regionNodeIds.insert(node.id);
        std::vector<StreetConfig> expectedStreets;
        for (const auto& street : city.streets) {
            if (regionNodeIds.count(street.from) && regionNodeIds.count(street.to)) expectedStreets.push_back(street);
        }
        CHECK(region.streets.size() == expectedStreets.size());
        for (std::size_t streetIndex = 0; streetIndex < expectedStreets.size(); ++streetIndex) {
            CHECK(sameStreet(region.streets[streetIndex], expectedStreets[streetIndex]));
        }

        // Restaurantes: solo los que quedan sobre nodos conservados.
        for (const auto& restaurant : region.restaurants) CHECK(regionNodeIds.count(restaurant.node) == 1);

        // Determinista: dos corridas dan lo mismo.
        const Config repeated = bench::extractRegion(city, city.startNode, nodeCount);
        CHECK(repeated.nodes.size() == region.nodes.size());
        for (std::size_t nodeIndex = 0; nodeIndex < region.nodes.size(); ++nodeIndex) {
            CHECK(repeated.nodes[nodeIndex].id == region.nodes[nodeIndex].id);
        }
        CHECK(repeated.streets.size() == region.streets.size());
    }

    // Anidados: la region de 100 esta contenida en la de 400 (misma BFS).
    {
        const Config smallRegion = bench::extractRegion(city, city.startNode, 100);
        const Config largeRegion = bench::extractRegion(city, city.startNode, 400);
        std::set<std::string> largeRegionIds;
        for (const auto& node : largeRegion.nodes) largeRegionIds.insert(node.id);
        for (const auto& node : smallRegion.nodes) CHECK(largeRegionIds.count(node.id) == 1);
    }

    // El sentido se conserva: una subregion grande contiene calles de un solo sentido y de doble sentido.
    {
        const Config region = bench::extractRegion(city, city.startNode, 1000);
        const auto oneWayCount = std::count_if(region.streets.begin(), region.streets.end(),
                                               [](const StreetConfig& street) { return street.oneWay; });
        CHECK(oneWayCount > 0);
        CHECK(static_cast<std::size_t>(oneWayCount) < region.streets.size());
    }

    // Errores: nodeCount = 0, nodeCount mayor que la ciudad, nodo de partida inexistente.
    {
        bool threw = false;
        try { bench::extractRegion(city, city.startNode, 3742); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        threw = false;
        try { bench::extractRegion(city, city.startNode, 0); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        threw = false;
        try { bench::extractRegion(city, "no_existe", 10); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }

    // Grafo desconectado: pedir mas nodos que los de la componente del inicio es un error.
    {
        Config disconnected;
        disconnected.startNode = "a";
        for (const char* nodeId : {"a", "b", "c", "d"}) disconnected.nodes.push_back(NodeConfig{nodeId, 0.0, 0.0});
        disconnected.streets.push_back(StreetConfig{"s0", "a", "b", false});
        disconnected.streets.push_back(StreetConfig{"s1", "c", "d", true});
        const Config firstComponent = bench::extractRegion(disconnected, "a", 2);
        CHECK(firstComponent.nodes.size() == 2 && firstComponent.streets.size() == 1);
        bool threw = false;
        try { bench::extractRegion(disconnected, "a", 3); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        // Un sentido unico se respeta: solo a->b, pedir desde b sigue alcanzando a (conexidad debil).
        Config oneWayConfig = disconnected;
        oneWayConfig.streets[0].oneWay = true;
        const Config fromNodeB = bench::extractRegion(oneWayConfig, "b", 2);
        CHECK(fromNodeB.streets.size() == 1 && fromNodeB.streets[0].from == "a" && fromNodeB.streets[0].to == "b" &&
              fromNodeB.streets[0].oneWay);
    }

    std::cout << "test_city_subgraph OK\n";
    return 0;
}
