// Tabla de tiempos entre todos los pares de intersecciones.
// Sin CUDA: prueba la tabla, Floyd-Warshall de 1 y N hilos, Dijkstra desde cada nodo, el relleno a
// multiplos de 32, la estimacion con tabla y el armado de la tabla al arrancar (con respaldo CPU).
// La equivalencia de la GPU con la CPU la prueba el modo --bench (columna `correct`).
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "CityGraph.h"
#include "Config.h"
#include "Courier.h"
#include "Order.h"
#include "Quote.h"
#include "Router.h"
#include "accel/GpuWorker.h"
#include "accel/ThreadPool.h"
#include "accel/TravelTimeTable.h"
#include "bench/BenchGraph.h"
#include "bench/CitySubgraph.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

using accel::DirectedStreet;
using accel::kNoPathMs;

// Config minima con nodos "n0".."n<count-1>" y las calles dadas; la velocidad hace que los ms salgan
// de la distancia real (los tests comparan siempre contra el Router, no contra numeros a mano).
Config smallConfig(std::size_t nodeCount, const std::vector<StreetConfig>& streets) {
    Config config;
    config.speedKmh = 20.0;
    for (std::size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        config.nodes.push_back(NodeConfig{"n" + std::to_string(nodeIndex), -17.77 + 0.001 * static_cast<double>(nodeIndex),
                                          -63.19 + 0.0005 * static_cast<double>(nodeIndex * nodeIndex % 7)});
    }
    config.streets = streets;
    return config;
}

StreetConfig street(const std::string& id, std::size_t from, std::size_t to, bool oneWay) {
    return StreetConfig{id, "n" + std::to_string(from), "n" + std::to_string(to), oneWay};
}

// Compara celda por celda una tabla contra el Router, par a par.
void checkAgainstRouter(const CityGraph& graph, const accel::TravelTimeTable& table) {
    Router router(graph);
    const std::size_t nodeCount = graph.nodeCount();
    CHECK(table.nodeCount() == nodeCount);
    for (std::size_t fromNode = 0; fromNode < nodeCount; ++fromNode) {
        for (std::size_t toNode = 0; toNode < nodeCount; ++toNode) {
            const Route route = router.shortestPath(static_cast<NodeIndex>(fromNode), static_cast<NodeIndex>(toNode));
            const std::int64_t expected = route.found ? route.travelMs : -1;
            if (table.travelTimeMs(fromNode, toNode) != expected) {
                std::cerr << "FALLO celda " << fromNode << " -> " << toNode << ": tabla "
                          << table.travelTimeMs(fromNode, toNode) << ", Router " << expected << "\n";
                std::exit(1);
            }
        }
    }
}

// FW 1 hilo, FW con varios hilos y Dijkstra (1 y N hilos) dan la misma tabla.
accel::TravelTimeTable checkAllVersionsAgree(const CityGraph& graph) {
    const std::size_t nodeCount = graph.nodeCount();
    std::vector<std::int32_t> dijkstraCells;
    accel::dijkstraFromEveryNode(graph, dijkstraCells);

    std::vector<std::int32_t> floydCells = accel::buildInitialCells(nodeCount, accel::streetsOf(graph));
    accel::floydWarshallSingleThread(floydCells, nodeCount);
    CHECK(floydCells == dijkstraCells);

    for (unsigned threads : {1u, 2u, 3u, 4u, 16u}) {
        accel::ThreadPool pool(threads);
        std::vector<std::int32_t> floydParallel = accel::buildInitialCells(nodeCount, accel::streetsOf(graph));
        accel::floydWarshallMultiThread(floydParallel, nodeCount, pool);
        CHECK(floydParallel == dijkstraCells);

        std::vector<std::int32_t> dijkstraParallel;
        accel::dijkstraFromEveryNodeParallel(graph, dijkstraParallel, pool);
        CHECK(dijkstraParallel == dijkstraCells);
    }
    return accel::TravelTimeTable(nodeCount, std::move(dijkstraCells));
}

Order orderAt(const std::string& pickup, const std::string& dropoff, std::int64_t prepMs) {
    Order order;
    order.id = 1;
    order.restaurantId = "r0";
    order.pickupNode = pickup;
    order.dropoffNode = dropoff;
    order.prepMs = prepMs;
    return order;
}

// El ejemplo de 4 nodos de un solo sentido de estudio/partes/bloque_A.md.
void testFourNodeExample() {
    const std::vector<DirectedStreet> streets{{0, 1, 5}, {1, 2, 3}, {2, 3, 1}, {3, 0, 2}, {0, 3, 10}};
    const std::vector<std::int32_t> expected{0, 5, 8, 9,   //
                                             6, 0, 3, 4,   //
                                             3, 8, 0, 1,   //
                                             2, 7, 10, 0};
    std::vector<std::int32_t> cells = accel::buildInitialCells(4, streets);
    CHECK(cells[0 * 4 + 2] == kNoPathMs);  // 0 -> 2 no tiene calle directa
    CHECK(cells[1 * 4 + 0] == kNoPathMs);  // y 1 -> 0 tampoco: es de un solo sentido
    CHECK(cells[2 * 4 + 2] == 0);
    accel::floydWarshallSingleThread(cells, 4);
    CHECK(cells == expected);

    for (unsigned threads : {1u, 2u, 3u, 4u, 16u}) {
        accel::ThreadPool pool(threads);
        std::vector<std::int32_t> parallelCells = accel::buildInitialCells(4, streets);
        accel::floydWarshallMultiThread(parallelCells, 4, pool);
        CHECK(parallelCells == expected);
    }

    const accel::TravelTimeTable table(4, expected);
    CHECK(table.travelTimeMs(0, 3) == 9);
    CHECK(table.travelTimeMs(3, 0) == 2);  // respeta el sentido
    CHECK(table.travelTimeMs(0, 1) == 5);
    CHECK(table.travelTimeMs(1, 0) == 6);
}

void testParallelEdgesKeepTheFastest() {
    const std::vector<DirectedStreet> streets{{0, 1, 9}, {0, 1, 4}, {0, 1, 7}};
    const std::vector<std::int32_t> cells = accel::buildInitialCells(2, streets);
    CHECK(cells[0 * 2 + 1] == 4);
}

void testUnreachablePairsAreMinusOne() {
    // Dos islas {0,1} y {2,3}, y una calle de un solo sentido 1 -> 2.
    const std::vector<DirectedStreet> streets{{0, 1, 5}, {1, 0, 5}, {2, 3, 2}, {3, 2, 2}, {1, 2, 8}};
    std::vector<std::int32_t> cells = accel::buildInitialCells(4, streets);
    accel::floydWarshallSingleThread(cells, 4);
    const accel::TravelTimeTable table(4, cells);
    CHECK(table.travelTimeMs(0, 3) == 15);  // 0 -> 1 -> 2 -> 3
    CHECK(table.travelTimeMs(3, 0) == -1);  // la vuelta no existe
    CHECK(table.travelTimeMs(2, 1) == -1);
    CHECK(table.travelTimeMs(2, 2) == 0);
    CHECK(table.travelTimeMs(9, 0) == -1);  // fuera de rango: como un Router sin ruta
    CHECK(table.travelTimeMs(0, 9) == -1);
}

void testOneWayStreetsInARealGraph() {
    // n0 -> n1 de un solo sentido, n1 <-> n2 de doble sentido, n3 aislado.
    const Config config = smallConfig(4, {street("a", 0, 1, true), street("b", 1, 2, false)});
    const CityGraph graph(config);
    const accel::TravelTimeTable table = checkAllVersionsAgree(graph);
    CHECK(table.travelTimeMs(0, 1) > 0);
    CHECK(table.travelTimeMs(1, 0) == -1);
    CHECK(table.travelTimeMs(0, 2) == table.travelTimeMs(0, 1) + table.travelTimeMs(1, 2));
    CHECK(table.travelTimeMs(2, 0) == -1);
    CHECK(table.travelTimeMs(0, 3) == -1);
    CHECK(table.travelTimeMs(3, 3) == 0);
    checkAgainstRouter(graph, table);
}

void testEmptyAndSingleNodeGraphs() {
    const CityGraph empty(smallConfig(0, {}));
    const accel::TravelTimeTable emptyTable = checkAllVersionsAgree(empty);
    CHECK(emptyTable.nodeCount() == 0);
    CHECK(emptyTable.cells().empty());
    CHECK(emptyTable.travelTimeMs(0, 0) == -1);

    const CityGraph single(smallConfig(1, {}));
    const accel::TravelTimeTable singleTable = checkAllVersionsAgree(single);
    CHECK(singleTable.nodeCount() == 1);
    CHECK(singleTable.travelTimeMs(0, 0) == 0);
}

// Tamanos que no son multiplo de 32, con calles de un sentido y pares sin camino.
void testSyntheticGraphsOfAwkwardSizes() {
    for (std::size_t nodeCount : {1u, 2u, 5u, 31u, 32u, 33u, 64u, 100u}) {
        bool usedCity = true;
        const Config region = bench::cityRegionOrSynthetic(Config{}, nodeCount, 42, &usedCity);
        CHECK(!usedCity);
        CHECK(region.nodes.size() == nodeCount);
        const CityGraph graph(region);
        const accel::TravelTimeTable table = checkAllVersionsAgree(graph);
        if (nodeCount <= 64) {
            checkAgainstRouter(graph, table);
        }
        // Determinista: la misma semilla da el mismo grafo.
        const Config again = bench::cityRegionOrSynthetic(Config{}, nodeCount, 42, nullptr);
        CHECK(again.streets.size() == region.streets.size());
    }
    // En un grafo sintetico grande hay pares sin camino (calles de un sentido): -1 aparece.
    const CityGraph graph(bench::cityRegionOrSynthetic(Config{}, 100, 7, nullptr));
    const accel::TravelTimeTable table = checkAllVersionsAgree(graph);
    std::size_t unreachablePairs = 0;
    for (std::size_t fromNode = 0; fromNode < 100; ++fromNode) {
        for (std::size_t toNode = 0; toNode < 100; ++toNode) {
            if (table.travelTimeMs(fromNode, toNode) < 0) ++unreachablePairs;
        }
    }
    CHECK(unreachablePairs > 0);
}

void testPaddingToTileMultiple() {
    CHECK(accel::paddedNodeCount(0) == 0);
    CHECK(accel::paddedNodeCount(1) == 32);
    CHECK(accel::paddedNodeCount(32) == 32);
    CHECK(accel::paddedNodeCount(33) == 64);
    CHECK(accel::paddedNodeCount(3741) == 3744);

    const std::vector<DirectedStreet> streets{{0, 1, 5}, {1, 2, 3}};
    const std::vector<std::int32_t> cells = accel::buildInitialCells(3, streets);
    const std::vector<std::int32_t> padded = accel::padCells(cells, 3);
    CHECK(padded.size() == 32u * 32u);
    for (std::size_t fromNode = 0; fromNode < 32; ++fromNode) {
        for (std::size_t toNode = 0; toNode < 32; ++toNode) {
            const std::int32_t value = padded[fromNode * 32 + toNode];
            if (fromNode < 3 && toNode < 3) {
                CHECK(value == cells[fromNode * 3 + toNode]);
            } else if (fromNode == toNode) {
                CHECK(value == 0);  // los nodos de relleno: 0 en la diagonal
            } else {
                CHECK(value == kNoPathMs);  // y sin calles
            }
        }
    }
    CHECK(accel::unpadCells(padded, 3) == cells);

    // Floyd-Warshall sobre la matriz rellena da lo mismo que sin relleno (los nodos de relleno no crean caminos).
    std::vector<std::int32_t> plain = cells;
    accel::floydWarshallSingleThread(plain, 3);
    std::vector<std::int32_t> paddedFloyd = padded;
    accel::floydWarshallSingleThread(paddedFloyd, 32);
    CHECK(accel::unpadCells(paddedFloyd, 3) == plain);
}

void testSizeAndWeightLimits() {
    bool threw = false;
    try {
        accel::tableCellCount(accel::kMaxTableNodes + 1);
    } catch (const accel::TravelTimeTableError&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        accel::tableCellCount(std::numeric_limits<std::size_t>::max());  // n * n desbordaria size_t
    } catch (const accel::TravelTimeTableError&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(accel::tableCellCount(3741) == 3741u * 3741u);
    CHECK(accel::tableCellCount(0) == 0);

    // Un tiempo de calle que no cabe, o un camino posible que podria alcanzar el "infinito": error claro.
    for (const std::vector<DirectedStreet>& streets :
         {std::vector<DirectedStreet>{{0, 1, kNoPathMs}}, std::vector<DirectedStreet>{{0, 1, -3}},
          std::vector<DirectedStreet>{{0, 5, 1}},  // nodo fuera de rango
          std::vector<DirectedStreet>{{0, 1, 600000000}, {1, 2, 600000000}}}) {
        threw = false;
        try {
            accel::buildInitialCells(3, streets);
        } catch (const accel::TravelTimeTableError&) {
            threw = true;
        }
        CHECK(threw);
    }
    // Un tiempo grande pero seguro si pasa.
    CHECK(accel::buildInitialCells(3, {{0, 1, 400000000}, {1, 2, 400000000}}).size() == 9);
}

void testRealCityTables() {
    // Equipetrol completo y un subgrafo de la ciudad: todas las versiones == Router par a par.
    const Config equipetrol = loadConfig("config/equipetrol.json");
    const CityGraph equipetrolGraph(equipetrol);
    checkAgainstRouter(equipetrolGraph, checkAllVersionsAgree(equipetrolGraph));

    const Config city = loadConfig("config/ciudad.json");
    const Config region = bench::extractRegion(city, city.startNode, 100);
    const CityGraph regionGraph(region);
    checkAgainstRouter(regionGraph, checkAllVersionsAgree(regionGraph));

    // Region mas grande: las versiones coinciden entre si (el Router par a par seria lento).
    const Config bigRegion = bench::extractRegion(city, city.startNode, 400);
    const CityGraph bigGraph(bigRegion);
    checkAllVersionsAgree(bigGraph);

    // cityRegionOrSynthetic usa la ciudad cuando alcanza y el respaldo sintetico cuando no.
    bool usedCity = false;
    const Config fromCity = bench::cityRegionOrSynthetic(city, 100, 1, &usedCity);
    CHECK(usedCity);
    CHECK(fromCity.nodes.size() == 100);
    const Config tooBig = bench::cityRegionOrSynthetic(equipetrol, equipetrol.nodes.size() + 50, 1, &usedCity);
    CHECK(!usedCity);
    CHECK(tooBig.nodes.size() == equipetrol.nodes.size() + 50);
}

// Estimacion con tabla == estimacion con Router, ms exactos, tambien sin camino y con nodos desconocidos.
void testEstimateWithTableEqualsRouter() {
    const Config config = loadConfig("config/equipetrol.json");
    const CityGraph graph(config);
    const accel::TravelTimeTable table = checkAllVersionsAgree(graph);
    Router router(graph);

    std::size_t compared = 0;
    const std::size_t step = config.nodes.size() / 12 + 1;
    for (std::size_t atIndex = 0; atIndex < config.nodes.size(); atIndex += step) {
        const Courier courier(1, config.nodes[atIndex].id, config.bagCapacity);
        for (std::size_t pickupIndex = 1; pickupIndex < config.nodes.size(); pickupIndex += step) {
            for (std::size_t dropoffIndex = 2; dropoffIndex < config.nodes.size(); dropoffIndex += step) {
                for (std::int64_t prepMs : {std::int64_t{0}, std::int64_t{5000}, std::int64_t{10000000}}) {
                    const Order order = orderAt(config.nodes[pickupIndex].id, config.nodes[dropoffIndex].id, prepMs);
                    CHECK(estimateDeliveryMs(table, graph, courier, order) == estimateDeliveryMs(router, graph, courier, order));
                    ++compared;
                }
            }
        }
    }
    CHECK(compared > 100);

    // Nodo desconocido: kUnreachable, igual que con el Router.
    const Courier courier(1, config.startNode, config.bagCapacity);
    const Order unknownPickup = orderAt("no-existe", config.nodes[0].id, 0);
    CHECK(estimateDeliveryMs(table, graph, courier, unknownPickup) == kUnreachable);
    CHECK(estimateDeliveryMs(router, graph, courier, unknownPickup) == kUnreachable);
    const Courier lost(2, "no-existe", config.bagCapacity);
    const Order normal = orderAt(config.nodes[1].id, config.nodes[2].id, 0);
    CHECK(estimateDeliveryMs(table, graph, lost, normal) == kUnreachable);

    // Sin camino: grafo con un nodo aislado y calles de un sentido. Mismo resultado (-1) con tabla y con Router.
    const Config island = smallConfig(4, {street("a", 0, 1, true), street("b", 1, 2, false)});
    const CityGraph islandGraph(island);
    const accel::TravelTimeTable islandTable = checkAllVersionsAgree(islandGraph);
    Router islandRouter(islandGraph);
    for (std::size_t atIndex = 0; atIndex < 4; ++atIndex) {
        for (std::size_t pickupIndex = 0; pickupIndex < 4; ++pickupIndex) {
            for (std::size_t dropoffIndex = 0; dropoffIndex < 4; ++dropoffIndex) {
                const Courier islandCourier(1, "n" + std::to_string(atIndex), 2);
                const Order order = orderAt("n" + std::to_string(pickupIndex), "n" + std::to_string(dropoffIndex), 100);
                CHECK(estimateDeliveryMs(islandTable, islandGraph, islandCourier, order) ==
                      estimateDeliveryMs(islandRouter, islandGraph, islandCourier, order));
            }
        }
    }
    const Courier fromIsland(1, "n3", 2);
    CHECK(estimateDeliveryMs(islandTable, islandGraph, fromIsland, orderAt("n0", "n1", 0)) == kUnreachable);
    CHECK(estimateDeliveryMs(islandTable, islandGraph, Courier(1, "n1", 2), orderAt("n0", "n2", 0)) == kUnreachable);
}

// La tabla del arranque: sin GPU la arma la CPU; con un worker cuya tarea falla (sin CUDA) vuelve al respaldo CPU.
void testBuildAtStartup() {
    const Config config = loadConfig("config/equipetrol.json");
    const CityGraph graph(config);
    std::vector<std::int32_t> expectedCells;
    accel::dijkstraFromEveryNode(graph, expectedCells);

    const accel::TravelTimeBuild withoutGpu = accel::buildTravelTimeTable(graph, nullptr, 3);
    CHECK(withoutGpu.table != nullptr);
    CHECK(withoutGpu.table->cells() == expectedCells);
    CHECK(withoutGpu.line.find("CPU") != std::string::npos);

    accel::GpuWorker worker([] {});  // sano, pero esta compilacion no tiene CUDA: la tarea de la GPU falla
    const accel::TravelTimeBuild withBrokenGpu = accel::buildTravelTimeTable(graph, &worker, 2);
    CHECK(withBrokenGpu.table != nullptr);
    CHECK(withBrokenGpu.table->cells() == expectedCells);
    CHECK(withBrokenGpu.line.find("CPU") != std::string::npos);
    worker.stop();

    // Mas nodos que el tope: no se arma tabla (el programa sigue con el Router) y la linea lo dice.
    const CityGraph hundredNodeGraph(bench::cityRegionOrSynthetic(Config{}, 100, 1, nullptr));
    const accel::TravelTimeBuild tooBig = accel::buildTravelTimeTable(hundredNodeGraph, nullptr, 2, /*maxNodes=*/50);
    CHECK(tooBig.table == nullptr);
    CHECK(tooBig.line.find("tope") != std::string::npos);
}


// Plazo de arranque: si el hilo de la GPU esta ocupado y no responde a tiempo, la CPU arma la tabla y la
// tarea de la GPU se descarta (captura por valor, asi que puede seguir viva sin dangling).
void testStartupDeadlineFallsBackToCpu() {
    const Config config = loadConfig("config/equipetrol.json");
    const CityGraph graph(config);
    std::vector<std::int32_t> expectedCells;
    accel::dijkstraFromEveryNode(graph, expectedCells);

    accel::GpuWorker worker([] {});
    std::promise<void> release;
    std::shared_future<void> gate = release.get_future().share();
    std::future<void> blocker = worker.submit([gate] { gate.wait(); });  // ocupa el hilo de la GPU

    const accel::TravelTimeBuild build = accel::buildTravelTimeTable(graph, &worker, 2, accel::kMaxTableNodes, /*gpuTimeoutMs=*/100);
    CHECK(build.table != nullptr);
    CHECK(build.table->cells() == expectedCells);
    CHECK(build.line.find("CPU") != std::string::npos);
    CHECK(build.line.find("plazo") != std::string::npos);

    release.set_value();  // la tarea descartada puede correr ahora (la GPU falla sin CUDA) sin tocar nada del llamador
    blocker.get();
    worker.stop();
}
}  // namespace

int main() {
    testFourNodeExample();
    testParallelEdgesKeepTheFastest();
    testUnreachablePairsAreMinusOne();
    testOneWayStreetsInARealGraph();
    testEmptyAndSingleNodeGraphs();
    testSyntheticGraphsOfAwkwardSizes();
    testPaddingToTileMultiple();
    testSizeAndWeightLimits();
    testRealCityTables();
    testEstimateWithTableEqualsRouter();
    testBuildAtStartup();
    testStartupDeadlineFallsBackToCpu();
    std::cout << "test_travel_time_table OK\n";
    return 0;
}
