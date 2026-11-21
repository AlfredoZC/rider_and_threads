// Aplicacion 10: filtro de candidatos. 1 hilo y N hilos dan la MISMA lista (mismo orden); el ejemplo de
// 8 couriers del material da {1, 3, 4, 7}; casos borde: radio que no deja a nadie, todos disponibles, flota
// vacia, entradas invalidas. El servicio CandidateFilter (GPU con respaldo CPU) se prueba sin CUDA, donde
// la GPU "falla" y siempre responde la CPU. La equivalencia GPU == CPU la prueba el bench (columna correct).
#include "accel/CandidateFilter.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "accel/GpuWorker.h"
#include "accel/TravelTimeTable.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

using accel::FleetColumns;
using accel::TravelTimeTable;
using Candidates = std::vector<std::int32_t>;

// Ciudad del material: 0 -> 1 (5), 1 -> 2 (3), 2 -> 3 (1), 3 -> 0 (2), 0 -> 3 (10), calculada con Floyd-Warshall.
std::shared_ptr<const TravelTimeTable> exampleTable() {
    const std::vector<accel::WeightedEdge> streets = {{0, 1, 5}, {1, 2, 3}, {2, 3, 1}, {3, 0, 2}, {0, 3, 10}};
    std::vector<std::int32_t> cells = accel::buildInitialCells(4, streets);
    accel::floydWarshallSingleThread(cells, 4);
    return std::make_shared<const TravelTimeTable>(4, std::move(cells));
}

FleetColumns exampleFleet() {
    FleetColumns fleet;
    fleet.node = {3, 1, 0, 2, 0, 3, 1, 2};
    fleet.available = {1, 1, 0, 1, 1, 1, 0, 1};
    return fleet;
}

FleetColumns randomFleet(std::size_t courierCount, std::size_t nodeCount, std::uint32_t seed) {
    std::mt19937 generator(seed);
    FleetColumns fleet;
    for (std::size_t courier = 0; courier < courierCount; ++courier) {
        fleet.node.push_back(static_cast<std::int32_t>(generator() % nodeCount));
        fleet.available.push_back(generator() % 10 < 7 ? 1 : 0);  // ~70 % disponibles
    }
    return fleet;
}

void testExample() {
    const auto table = exampleTable();
    const FleetColumns fleet = exampleFleet();
    // Tiempos al restaurante (nodo 2): 0 -> 8, 1 -> 3, 2 -> 0, 3 -> 10. Con R = 8 pasan {1, 3, 4, 7}.
    CHECK(table->travelTimeMs(0, 2) == 8 && table->travelTimeMs(3, 2) == 10);
    const Candidates expected = {1, 3, 4, 7};
    CHECK(accel::filterCandidatesSingleThread(fleet, *table, 2, 8) == expected);
    for (const unsigned threads : {1u, 2u, 3u, 4u, 8u, 16u}) {
        accel::ThreadPool pool(threads);
        CHECK(accel::filterCandidatesMultiThread(fleet, *table, 2, 8, pool) == expected);
    }
    // Un radio mas chico deja menos; el limite es inclusivo (<=).
    CHECK(accel::filterCandidatesSingleThread(fleet, *table, 2, 3) == (Candidates{1, 3, 7}));
    CHECK(accel::filterCandidatesSingleThread(fleet, *table, 2, 2) == (Candidates{3, 7}));
}

void testEdgeCases() {
    const auto table = exampleTable();
    accel::ThreadPool pool(3);
    const FleetColumns fleet = exampleFleet();

    // Radio que no deja a nadie (negativo) y radio de "ninguno llega" (< 0).
    CHECK(accel::filterCandidatesSingleThread(fleet, *table, 2, -1).empty());
    CHECK(accel::filterCandidatesMultiThread(fleet, *table, 2, -1, pool).empty());

    // Todos disponibles y radio enorme: pasan todos, en orden.
    FleetColumns allAvailable = fleet;
    allAvailable.available.assign(fleet.size(), 1);
    Candidates everyone;
    for (std::int32_t courier = 0; courier < 8; ++courier) everyone.push_back(courier);
    CHECK(accel::filterCandidatesSingleThread(allAvailable, *table, 2, accel::kUnreachableMs - 1) == everyone);
    CHECK(accel::filterCandidatesMultiThread(allAvailable, *table, 2, accel::kUnreachableMs - 1, pool) == everyone);

    // Nadie disponible: nadie pasa, aunque el radio sea enorme (y no se lee la tabla).
    FleetColumns noneAvailable = fleet;
    noneAvailable.available.assign(fleet.size(), 0);
    CHECK(accel::filterCandidatesSingleThread(noneAvailable, *table, 2, accel::kUnreachableMs - 1).empty());

    // Flota vacia y flota de un courier.
    const FleetColumns empty;
    CHECK(accel::filterCandidatesSingleThread(empty, *table, 2, 100).empty());
    CHECK(accel::filterCandidatesMultiThread(empty, *table, 2, 100, pool).empty());
    FleetColumns single;
    single.node = {1};
    single.available = {1};
    CHECK(accel::filterCandidatesMultiThread(single, *table, 2, 3, pool) == (Candidates{0}));

    // Un courier que no puede llegar (tabla con "no hay camino") no pasa con ningun radio razonable.
    const TravelTimeTable islands(2, {0, accel::kUnreachableMs, accel::kUnreachableMs, 0});
    FleetColumns apart;
    apart.node = {0, 1};
    apart.available = {1, 1};
    CHECK(accel::filterCandidatesSingleThread(apart, islands, 1, 1000000) == (Candidates{1}));

    // Entradas invalidas.
    bool threw = false;
    try {
        FleetColumns broken;
        broken.node = {0, 1};
        broken.available = {1};
        (void)accel::filterCandidatesSingleThread(broken, *table, 2, 5);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)accel::filterCandidatesSingleThread(fleet, *table, 4, 5);  // restaurante fuera de la tabla
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        FleetColumns outside;
        outside.node = {9};
        outside.available = {1};
        (void)accel::filterCandidatesMultiThread(outside, *table, 2, 5, pool);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    // Un nodo invalido en un courier NO disponible no importa: no se mira.
    FleetColumns unavailableOutside;
    unavailableOutside.node = {9};
    unavailableOutside.available = {0};
    CHECK(accel::filterCandidatesSingleThread(unavailableOutside, *table, 2, 5).empty());
}

void testRandomFleets() {
    const std::size_t nodeCount = 300;
    std::mt19937 generator(7);
    std::vector<accel::WeightedEdge> streets;
    for (std::size_t street = 0; street < nodeCount * 3; ++street) {
        streets.push_back({generator() % nodeCount, generator() % nodeCount, static_cast<std::int64_t>(1 + generator() % 20)});
    }
    std::vector<std::int32_t> cells = accel::buildInitialCells(nodeCount, streets);
    accel::floydWarshallSingleThread(cells, nodeCount);
    const TravelTimeTable table(nodeCount, std::move(cells));
    for (const std::size_t courierCount : {1u, 7u, 100u, 5000u}) {
        const FleetColumns fleet = randomFleet(courierCount, nodeCount, 11u + static_cast<std::uint32_t>(courierCount));
        for (const std::int32_t radius : {-1, 5, 15, accel::kUnreachableMs - 1}) {
            const Candidates reference = accel::filterCandidatesSingleThread(fleet, table, 42, radius);
            for (const unsigned threads : {2u, 4u, 7u}) {
                accel::ThreadPool pool(threads);
                CHECK(accel::filterCandidatesMultiThread(fleet, table, 42, radius, pool) == reference);
            }
        }
    }
}

void testServiceFallsBackToCpu() {
    const auto table = exampleTable();
    const Candidates expected = {1, 3, 4, 7};
    // Sin GpuWorker: siempre CPU.
    accel::CandidateFilter withoutGpu(nullptr);
    CHECK(withoutGpu.filter(table, exampleFleet(), 2, 8, 100) == expected);
    CHECK(withoutGpu.cpuFilters() == 1 && withoutGpu.gpuFilters() == 0);

    // Con un GpuWorker: el resultado es el mismo con o sin CUDA; sin CUDA la tarea falla y responde la CPU.
    accel::GpuWorker worker([] {});
    accel::CandidateFilter withWorker(&worker);
    CHECK(withWorker.filter(table, exampleFleet(), 2, 8, 2000) == expected);
    CHECK(withWorker.filter(table, exampleFleet(), 2, -1, 2000).empty());
    CHECK(withWorker.filter(table, FleetColumns{}, 2, 8, 2000).empty());
    CHECK(withWorker.gpuFilters() + withWorker.cpuFilters() == 3);
#ifndef USE_CUDA
    CHECK(withWorker.gpuFilters() == 0);
#endif
    worker.stop();
    // Con el worker detenido tambien responde la CPU.
    CHECK(withWorker.filter(table, exampleFleet(), 2, 8, 2000) == expected);
}

}  // namespace

int main() {
    testExample();
    testEdgeCases();
    testRandomFleets();
    testServiceFallsBackToCpu();
    std::cout << "TODO OK\n";
    return 0;
}
