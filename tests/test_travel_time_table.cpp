// Aplicacion 03: tabla de tiempos entre todos los pares. Floyd-Warshall con 1 hilo, con N hilos y
// Dijkstra desde cada nodo (1 y N hilos) tienen que dar la MISMA tabla, y esa tabla tiene que coincidir con
// Router. La equivalencia con la GPU la prueba el bench (columna correct) con el build CUDA.
#include "accel/TravelTimeTable.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "CityGraph.h"
#include "Config.h"
#include "Router.h"
#include "bench/CitySubgraph.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

using accel::kUnreachableMs;
using accel::TravelTimeTable;

// Ciudad sintetica: nodeCount nodos en una cuadricula chica, calles dadas por (desde, hasta, un sentido).
struct StreetSpec {
    std::size_t fromNode;
    std::size_t toNode;
    bool oneWay;
};

Config makeConfig(std::size_t nodeCount, const std::vector<StreetSpec>& streets) {
    Config config;
    config.speedKmh = 30.0;
    for (std::size_t node = 0; node < nodeCount; ++node) {
        NodeConfig nodeConfig;
        nodeConfig.id = "n" + std::to_string(node);
        nodeConfig.lat = -17.76 + 0.0007 * static_cast<double>(node / 10);
        nodeConfig.lon = -63.19 + 0.0009 * static_cast<double>(node % 10) + 0.00003 * static_cast<double>(node);
        config.nodes.push_back(nodeConfig);
    }
    std::size_t streetIndex = 0;
    for (const StreetSpec& spec : streets) {
        StreetConfig street;
        street.id = "s" + std::to_string(streetIndex++);
        street.from = "n" + std::to_string(spec.fromNode);
        street.to = "n" + std::to_string(spec.toNode);
        street.oneWay = spec.oneWay;
        config.streets.push_back(street);
    }
    return config;
}

std::vector<StreetSpec> randomStreets(std::size_t nodeCount, std::size_t streetCount, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<std::size_t> pickNode(0, nodeCount - 1);
    std::vector<StreetSpec> streets;
    for (std::size_t i = 0; i < streetCount; ++i) {
        streets.push_back({pickNode(generator), pickNode(generator), generator() % 3 == 0});
    }
    return streets;
}

// Referencia independiente: Bellman-Ford desde cada nodo sobre las aristas del grafo (saturado a kUnreachableMs).
std::vector<std::int32_t> bellmanFordAllPairs(const CityGraph& graph) {
    const std::size_t nodeCount = graph.nodeCount();
    std::vector<std::int32_t> expected(nodeCount * nodeCount, kUnreachableMs);
    for (std::size_t source = 0; source < nodeCount; ++source) {
        std::vector<std::int64_t> best(nodeCount, std::int64_t{kUnreachableMs});
        best[source] = 0;
        for (std::size_t pass = 0; pass < nodeCount; ++pass) {
            bool changed = false;
            for (std::size_t node = 0; node < nodeCount; ++node) {
                for (const Edge& edge : graph.edgesFrom(static_cast<NodeIndex>(node))) {
                    if (best[node] + edge.travelMs < best[edge.to]) {
                        best[edge.to] = best[node] + edge.travelMs;
                        changed = true;
                    }
                }
            }
            if (!changed) break;
        }
        for (std::size_t target = 0; target < nodeCount; ++target) {
            expected[source * nodeCount + target] = static_cast<std::int32_t>(best[target]);
        }
    }
    return expected;
}

// Las cuatro versiones de CPU dan lo mismo y coinciden con la referencia.
void checkAllCpuVersionsAgree(const CityGraph& graph, const std::string& label) {
    const std::size_t nodeCount = graph.nodeCount();
    const std::vector<std::int32_t> initialCells = accel::buildInitialCells(graph);
    CHECK(initialCells.size() == nodeCount * nodeCount);

    std::vector<std::int32_t> floydSingle = initialCells;
    accel::floydWarshallSingleThread(floydSingle, nodeCount);

    const std::vector<std::int32_t> expected = bellmanFordAllPairs(graph);
    if (floydSingle != expected) std::cerr << label << ": Floyd-Warshall 1 hilo != Bellman-Ford\n";
    CHECK(floydSingle == expected);

    for (const unsigned threads : {1u, 2u, 3u, 8u}) {
        accel::ThreadPool pool(threads);
        std::vector<std::int32_t> floydMulti = initialCells;
        accel::floydWarshallMultiThread(floydMulti, nodeCount, pool);
        if (floydMulti != floydSingle) std::cerr << label << ": Floyd-Warshall " << threads << " hilos != 1 hilo\n";
        CHECK(floydMulti == floydSingle);
        if (accel::dijkstraAllPairsMultiThread(graph, pool) != floydSingle) {
            std::cerr << label << ": Dijkstra " << threads << " hilos != Floyd-Warshall\n";
        }
        CHECK(accel::dijkstraAllPairsMultiThread(graph, pool) == floydSingle);
    }
    CHECK(accel::dijkstraAllPairsSingleThread(graph) == floydSingle);
}

void testEdgeCases() {
    // Sin nodos y con un nodo.
    {
        const CityGraph empty(makeConfig(0, {}));
        checkAllCpuVersionsAgree(empty, "vacio");
        CHECK(accel::buildInitialCells(empty).empty());
        const CityGraph single(makeConfig(1, {}));
        checkAllCpuVersionsAgree(single, "1 nodo");
        accel::ThreadPool pool(2);
        const auto table = accel::buildTravelTimeTableCpu(single, pool);
        CHECK(table->nodeCount() == 1);
        CHECK(table->travelTimeMs(0, 0) == 0);
    }
    // Un nodo inalcanzable (el 3 no tiene calles), calles de un sentido, calles paralelas y un lazo.
    {
        const std::vector<StreetSpec> streets = {
            {0, 1, true},   // 0 -> 1 solo ida
            {1, 2, true},   // 1 -> 2 solo ida
            {0, 2, false},  // 0 <-> 2 doble sentido, mas larga que pasar por 1? depende de la geometria
            {0, 1, true},   // paralela exacta de la primera
            {1, 0, true},   // 1 -> 0 solo ida (vuelta)
            {2, 2, false},  // lazo: CityGraph lo ignora
            {4, 5, false},
        };
        const CityGraph graph(makeConfig(6, streets));
        checkAllCpuVersionsAgree(graph, "borde");
        accel::ThreadPool pool(3);
        const auto table = accel::buildTravelTimeTableCpu(graph, pool);
        CHECK(table->nodeCount() == 6);
        // El nodo 3 no tiene calles: no se llega a el ni se sale de el, salvo a si mismo.
        for (NodeIndex node = 0; node < 6; ++node) {
            if (node == 3) continue;
            CHECK(table->travelTimeMs(node, 3) == TravelTimeTable::kNoRoute);
            CHECK(table->travelTimeMs(3, node) == TravelTimeTable::kNoRoute);
        }
        CHECK(table->travelTimeMs(3, 3) == 0);
        // Dos componentes: {0,1,2} y {4,5}.
        CHECK(table->travelTimeMs(0, 4) == TravelTimeTable::kNoRoute);
        CHECK(table->travelTimeMs(4, 5) > 0 && table->travelTimeMs(5, 4) == table->travelTimeMs(4, 5));
        // Sentido: a 2 se llega desde 0 y desde 1; a 1 solo desde 0 (y 1 -> 0 existe por la calle de vuelta).
        CHECK(table->travelTimeMs(0, 1) > 0);
        CHECK(table->travelTimeMs(1, 0) > 0);
        // Fuera de rango.
        CHECK(table->travelTimeMs(6, 0) == TravelTimeTable::kNoRoute);
        CHECK(table->travelTimeMs(0, 99) == TravelTimeTable::kNoRoute);
    }
    // Un solo sentido de verdad: una cadena 0 -> 1 -> 2 sin vuelta.
    {
        const CityGraph chain(makeConfig(3, {{0, 1, true}, {1, 2, true}}));
        checkAllCpuVersionsAgree(chain, "cadena");
        accel::ThreadPool pool(2);
        const auto table = accel::buildTravelTimeTableCpu(chain, pool);
        CHECK(table->travelTimeMs(0, 2) == table->travelTimeMs(0, 1) + table->travelTimeMs(1, 2));
        CHECK(table->travelTimeMs(2, 0) == TravelTimeTable::kNoRoute);
        CHECK(table->travelTimeMs(1, 0) == TravelTimeTable::kNoRoute);
    }
}

void testInitialCellsFromEdges() {
    // Aristas paralelas con distinto peso: gana la menor, en su sentido; el resto kUnreachableMs.
    const std::vector<accel::WeightedEdge> edges = {
        {0, 1, 500}, {0, 1, 300}, {0, 1, 900},  // paralelas: 300
        {1, 0, 700},                             // el otro sentido tiene su propio peso
        {2, 3, -5},                              // negativo -> 0
        {3, 2, std::int64_t{1} << 40},           // enorme -> kUnreachableMs - 1 (satura, no desborda)
        {0, 9, 10},                              // nodo fuera de rango: se ignora
        {9, 0, 10},
    };
    const std::vector<std::int32_t> cells = accel::buildInitialCells(4, edges);
    CHECK(cells.size() == 16);
    CHECK(cells[0 * 4 + 1] == 300);
    CHECK(cells[1 * 4 + 0] == 700);
    CHECK(cells[2 * 4 + 3] == 0);
    CHECK(cells[3 * 4 + 2] == kUnreachableMs - 1);
    CHECK(cells[0 * 4 + 2] == kUnreachableMs);
    for (std::size_t node = 0; node < 4; ++node) CHECK(cells[node * 4 + node] == 0);
    CHECK(accel::saturateEdgeMs(-1) == 0);
    CHECK(accel::saturateEdgeMs(123) == 123);
    CHECK(accel::saturateEdgeMs(std::int64_t{kUnreachableMs}) == kUnreachableMs - 1);
    CHECK(accel::saturateEdgeMs(INT64_MAX) == kUnreachableMs - 1);

    // Una arista a mas de 12 dias no crea un camino mas corto que "no hay camino" y la suma no desborda.
    std::vector<std::int32_t> longChain = accel::buildInitialCells(
        3, {{0, 1, std::int64_t{kUnreachableMs}}, {1, 2, std::int64_t{kUnreachableMs}}});
    accel::floydWarshallSingleThread(longChain, 3);
    CHECK(longChain[0 * 3 + 2] == kUnreachableMs);
    CHECK(longChain[0 * 3 + 1] == kUnreachableMs - 1);
}

void testTableObject() {
    bool threw = false;
    try {
        TravelTimeTable bad(3, std::vector<std::int32_t>(8, 0));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);

    const TravelTimeTable table(2, {0, 70, kUnreachableMs, 0});
    CHECK(table.travelTimeMs(0, 1) == 70);
    CHECK(table.travelTimeMs(1, 0) == TravelTimeTable::kNoRoute);
    CHECK(table.travelTimeMs(kNoNode, 0) == TravelTimeTable::kNoRoute);

    accel::TravelTimeTableSlot slot;
    CHECK(slot.get() == nullptr);
    auto published = std::make_shared<const TravelTimeTable>(1, std::vector<std::int32_t>{0});
    slot.publish(published);
    CHECK(slot.get() == published);
}

void testPadding() {
    CHECK(accel::paddedNodeCount(0, 32) == 0);
    CHECK(accel::paddedNodeCount(1, 32) == 32);
    CHECK(accel::paddedNodeCount(32, 32) == 32);
    CHECK(accel::paddedNodeCount(33, 32) == 64);
    CHECK(accel::paddedNodeCount(3741, 32) == 3744);
    bool threw = false;
    try {
        (void)accel::paddedNodeCount(SIZE_MAX - 3, 32);
    } catch (const std::length_error&) {
        threw = true;
    }
    CHECK(threw);

    const std::vector<std::int32_t> cells = {0, 5, 9, 0};
    const std::vector<std::int32_t> padded = accel::padCells(cells, 2, 4);
    CHECK(padded.size() == 16);
    CHECK(padded[0 * 4 + 1] == 5 && padded[1 * 4 + 0] == 9);
    CHECK(padded[2 * 4 + 2] == 0 && padded[3 * 4 + 3] == 0);          // diagonal del relleno
    CHECK(padded[0 * 4 + 2] == kUnreachableMs && padded[3 * 4 + 1] == kUnreachableMs);
    CHECK(accel::unpadCells(padded, 4, 2) == cells);
}

void testRandomGraphs() {
    // 33 y 65 no son multiplos de 32: sirven despues para la GPU; aqui solo cubren tamanos raros.
    for (const std::size_t nodeCount : {5u, 33u, 65u, 120u}) {
        const CityGraph graph(makeConfig(nodeCount, randomStreets(nodeCount, nodeCount * 2, 1000u + static_cast<std::uint32_t>(nodeCount))));
        checkAllCpuVersionsAgree(graph, "aleatorio n=" + std::to_string(nodeCount));
    }
}

void testRealCity() {
    const Config city = loadConfig("config/equipetrol.json");
    const CityGraph graph(city);
    const std::size_t nodeCount = graph.nodeCount();
    accel::ThreadPool pool(std::max(2u, std::min(8u, std::thread::hardware_concurrency())));
    const auto table = accel::buildTravelTimeTableCpu(graph, pool);
    CHECK(table->nodeCount() == nodeCount);

    // Muestra de pares contra Router.
    Router router(graph);
    std::mt19937 generator(7);
    std::uniform_int_distribution<std::size_t> pickNode(0, nodeCount - 1);
    for (int pair = 0; pair < 400; ++pair) {
        const NodeIndex from = static_cast<NodeIndex>(pickNode(generator));
        const NodeIndex to = static_cast<NodeIndex>(pickNode(generator));
        const Route route = router.shortestPath(from, to);
        if (route.found) {
            CHECK(table->travelTimeMs(from, to) == route.travelMs);
        } else {
            CHECK(table->travelTimeMs(from, to) == TravelTimeTable::kNoRoute);
        }
    }

    // Floyd-Warshall (1 y N hilos) sobre un subgrafo de la ciudad real == Dijkstra.
    const Config region = bench::extractRegion(city, city.startNode, std::min<std::size_t>(nodeCount, 150));
    const CityGraph regionGraph(region);
    checkAllCpuVersionsAgree(regionGraph, "subgrafo real");
}

#ifndef USE_CUDA
void testGpuUnavailableWithoutCuda() {
    std::vector<std::int32_t> result;
    CHECK(!accel::floydWarshallGpu({0}, 1, result, nullptr).empty());
    bool threw = false;
    try {
        (void)accel::buildTravelTimeTableGpuTask({0}, 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}
#endif

}  // namespace

int main() {
    testEdgeCases();
    testInitialCellsFromEdges();
    testTableObject();
    testPadding();
    testRandomGraphs();
    testRealCity();
#ifndef USE_CUDA
    testGpuUnavailableWithoutCuda();
#endif
    std::cout << "TODO OK\n";
    return 0;
}
