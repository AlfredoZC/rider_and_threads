// Aplicacion 04: ETA en lote pedido x courier. El ejemplo de 2 pedidos x 4 candidatos del material da
// `9 9 11 9 / 6 6 9 5`; los inalcanzables se propagan; cpu1 == cpuN == referencia independiente; el gather
// compacta bien; y el servicio EtaBatch (GPU con respaldo CPU) da lo mismo. La equivalencia GPU == CPU
// (SoA y AoS) la prueba el bench (columna correct) con el build CUDA.
#include "accel/EtaBatch.h"

#include <algorithm>
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

using accel::CourierColumns;
using accel::OrderColumns;
using accel::TravelTimeTable;
using Matrix = std::vector<std::int32_t>;

// Ciudad del material: 0 -> 1 (5), 1 -> 2 (3), 2 -> 3 (1), 3 -> 0 (2), 0 -> 3 (10).
std::shared_ptr<const TravelTimeTable> exampleTable() {
    std::vector<std::int32_t> cells =
        accel::buildInitialCells(4, {{0, 1, 5}, {1, 2, 3}, {2, 3, 1}, {3, 0, 2}, {0, 3, 10}});
    accel::floydWarshallSingleThread(cells, 4);
    return std::make_shared<const TravelTimeTable>(4, std::move(cells));
}

// La flota del material (8 couriers); los candidatos del filtro son 1, 3, 4 y 7.
CourierColumns exampleFleet() {
    CourierColumns fleet;
    fleet.node = {3, 1, 0, 2, 0, 3, 1, 2};
    fleet.freeAtMs = {0, 2, 0, 5, 0, 1, 0, 4};
    return fleet;
}

OrderColumns exampleOrders() {  // A: cliente 0, lista en 6. B: cliente 3, lista en 2. Restaurante 2.
    OrderColumns orders;
    orders.restaurantNode = {2, 2};
    orders.customerNode = {0, 3};
    orders.readyAtMs = {6, 2};
    return orders;
}

// Referencia independiente: la formula escrita a mano con travelTimeMs().
Matrix referenceEta(const CourierColumns& couriers, const OrderColumns& orders, const TravelTimeTable& table) {
    Matrix eta;
    for (std::size_t order = 0; order < orders.size(); ++order) {
        for (std::size_t column = 0; column < couriers.size(); ++column) {
            const std::int64_t toRestaurant = table.travelTimeMs(static_cast<NodeIndex>(couriers.node[column]),
                                                                 static_cast<NodeIndex>(orders.restaurantNode[order]));
            const std::int64_t toCustomer = table.travelTimeMs(static_cast<NodeIndex>(orders.restaurantNode[order]),
                                                               static_cast<NodeIndex>(orders.customerNode[order]));
            if (toRestaurant == TravelTimeTable::kNoRoute || toCustomer == TravelTimeTable::kNoRoute) {
                eta.push_back(accel::kUnreachableMs);
                continue;
            }
            const std::int64_t pickup = std::max<std::int64_t>(couriers.freeAtMs[column] + toRestaurant,
                                                               orders.readyAtMs[order]);
            eta.push_back(static_cast<std::int32_t>(pickup + toCustomer));
        }
    }
    return eta;
}

void testExample() {
    const auto table = exampleTable();
    const CourierColumns columns = accel::gatherColumns(exampleFleet(), {1, 3, 4, 7});
    CHECK(columns.node == (std::vector<std::int32_t>{1, 2, 0, 2}));
    CHECK(columns.freeAtMs == (std::vector<std::int32_t>{2, 5, 0, 4}));
    const Matrix expected = {9, 9, 11, 9, 6, 6, 9, 5};  // 2 pedidos x 4 candidatos, a mano
    CHECK(accel::computeEtaSingleThread(columns, exampleOrders(), *table) == expected);
    for (const unsigned threads : {1u, 2u, 3u, 4u, 8u}) {
        accel::ThreadPool pool(threads);
        CHECK(accel::computeEtaMultiThread(columns, exampleOrders(), *table, pool) == expected);
    }
    CHECK(referenceEta(columns, exampleOrders(), *table) == expected);
}

void testUnreachableAndEdges() {
    // Dos islas: 0 <-> 1 y 2 <-> 3. Un courier de una isla no llega a un restaurante de la otra.
    std::vector<std::int32_t> cells = accel::buildInitialCells(4, {{0, 1, 4}, {1, 0, 4}, {2, 3, 6}, {3, 2, 6}});
    accel::floydWarshallSingleThread(cells, 4);
    const TravelTimeTable islands(4, cells);
    CourierColumns couriers;
    couriers.node = {0, 2};
    couriers.freeAtMs = {0, 10};
    OrderColumns orders;
    orders.restaurantNode = {1, 3, 1};
    orders.customerNode = {0, 2, 2};  // el tercero: el cliente esta en la otra isla del restaurante
    orders.readyAtMs = {0, 100, 0};
    const Matrix eta = accel::computeEtaSingleThread(couriers, orders, islands);
    const Matrix expected = {4 + 4, accel::kUnreachableMs,  // pedido 0: el courier 0 llega y entrega; el 1 no llega
                             accel::kUnreachableMs, 100 + 6,  // pedido 1: el courier 0 no llega; el 1 espera a la comida
                             accel::kUnreachableMs, accel::kUnreachableMs};  // pedido 2: nadie (cliente inalcanzable)
    CHECK(eta == expected);
    accel::ThreadPool pool(3);
    CHECK(accel::computeEtaMultiThread(couriers, orders, islands, pool) == expected);

    // Sin couriers, sin pedidos, y un solo par.
    CHECK(accel::computeEtaSingleThread(CourierColumns{}, orders, islands).empty());
    CHECK(accel::computeEtaMultiThread(couriers, OrderColumns{}, islands, pool).empty());
    CourierColumns single;
    single.node = {0};
    single.freeAtMs = {0};
    OrderColumns oneOrder;
    oneOrder.restaurantNode = {1};
    oneOrder.customerNode = {0};
    oneOrder.readyAtMs = {0};
    CHECK(accel::computeEtaMultiThread(single, oneOrder, islands, pool) == (Matrix{8}));

    // La suma nunca desborda: courier libre muy tarde + viaje largo se satura en "inalcanzable".
    CourierColumns late;
    late.node = {0};
    late.freeAtMs = {2000000000};
    CHECK(accel::computeEtaSingleThread(late, oneOrder, islands) == (Matrix{accel::kUnreachableMs}));

    // Entradas invalidas.
    bool threw = false;
    try {
        CourierColumns broken;
        broken.node = {0, 1};
        broken.freeAtMs = {0};
        (void)accel::computeEtaSingleThread(broken, oneOrder, islands);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        OrderColumns outside = oneOrder;
        outside.customerNode = {9};
        (void)accel::computeEtaSingleThread(single, outside, islands);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    threw = false;
    try {
        (void)accel::gatherColumns(single, {0, 3});
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

void testRandomBatches() {
    const std::size_t nodeCount = 120;
    std::mt19937 generator(5);
    std::vector<accel::WeightedEdge> streets;
    for (std::size_t street = 0; street < nodeCount * 2; ++street) {  // grafo ralo: hay inalcanzables
        streets.push_back({generator() % nodeCount, generator() % nodeCount, static_cast<std::int64_t>(1 + generator() % 30)});
    }
    std::vector<std::int32_t> cells = accel::buildInitialCells(nodeCount, streets);
    accel::floydWarshallSingleThread(cells, nodeCount);
    const TravelTimeTable table(nodeCount, std::move(cells));
    for (const auto& shape : {std::pair<std::size_t, std::size_t>{1, 1}, {10, 50}, {7, 333}, {40, 5}}) {
        CourierColumns couriers;
        for (std::size_t courier = 0; courier < shape.second; ++courier) {
            couriers.node.push_back(static_cast<std::int32_t>(generator() % nodeCount));
            couriers.freeAtMs.push_back(static_cast<std::int32_t>(generator() % 30));
        }
        OrderColumns orders;
        for (std::size_t order = 0; order < shape.first; ++order) {
            orders.restaurantNode.push_back(static_cast<std::int32_t>(generator() % nodeCount));
            orders.customerNode.push_back(static_cast<std::int32_t>(generator() % nodeCount));
            orders.readyAtMs.push_back(static_cast<std::int32_t>(generator() % 40));
        }
        const Matrix reference = referenceEta(couriers, orders, table);
        CHECK(accel::computeEtaSingleThread(couriers, orders, table) == reference);
        for (const unsigned threads : {2u, 4u, 7u}) {
            accel::ThreadPool pool(threads);
            CHECK(accel::computeEtaMultiThread(couriers, orders, table, pool) == reference);
        }
        // AoS: los registros llevan los mismos campos que las columnas.
        const std::vector<accel::CourierRecord> records = accel::toRecords(couriers);
        CHECK(records.size() == couriers.size());
        for (std::size_t courier = 0; courier < records.size(); ++courier) {
            CHECK(records[courier].node == couriers.node[courier] && records[courier].freeAtMs == couriers.freeAtMs[courier]);
        }
    }
}

void testServiceFallsBackToCpu() {
    const auto table = exampleTable();
    const CourierColumns columns = accel::gatherColumns(exampleFleet(), {1, 3, 4, 7});
    const Matrix expected = {9, 9, 11, 9, 6, 6, 9, 5};
    accel::EtaBatch withoutGpu(nullptr);
    CHECK(withoutGpu.compute(table, columns, exampleOrders(), 100) == expected);
    CHECK(withoutGpu.cpuBatches() == 1 && withoutGpu.gpuBatches() == 0);

    accel::GpuWorker worker([] {});
    accel::EtaBatch withWorker(&worker);
    CHECK(withWorker.compute(table, columns, exampleOrders(), 2000) == expected);
    CHECK(withWorker.compute(table, CourierColumns{}, exampleOrders(), 2000).empty());
    CHECK(withWorker.gpuBatches() + withWorker.cpuBatches() == 2);
#ifndef USE_CUDA
    CHECK(withWorker.gpuBatches() == 0);
#endif
    worker.stop();
    CHECK(withWorker.compute(table, columns, exampleOrders(), 2000) == expected);
}

}  // namespace

int main() {
    testExample();
    testUnreachableAndEdges();
    testRandomBatches();
    testServiceFallsBackToCpu();
    std::cout << "TODO OK\n";
    return 0;
}
