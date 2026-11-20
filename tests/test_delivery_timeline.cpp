#include "accel/DeliveryTimeline.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "OrderBook.h"
#include "accel/GpuWorker.h"
#include "accel/ThreadPool.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

using accel::DeliveryRecord;
using Records = std::vector<DeliveryRecord>;
using ListStart = std::vector<int>;

std::uint64_t nextRandom(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t mixed = state;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
    return mixed ^ (mixed >> 31);
}

bool sameRecords(const Records& left, const Records& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t position = 0; position < left.size(); ++position) {
        if (left[position].deliveredAtMs != right[position].deliveredAtMs || left[position].orderId != right[position].orderId) {
            return false;
        }
    }
    return true;
}

// Referencia: todas las listas concatenadas y ordenadas de forma estable por hora de entrega.
Records stableSortedReference(Records records) {
    std::stable_sort(records.begin(), records.end(), [](const DeliveryRecord& left, const DeliveryRecord& right) {
        return left.deliveredAtMs < right.deliveredAtMs;
    });
    return records;
}

// Listas pseudoaleatorias (cada una ya ordenada por tiempo); `emptyOneIn`: 1 de cada tantos couriers sin entregas.
void makeLists(std::uint64_t seed, int listCount, int maxPerList, int timeStepMs, int emptyOneIn, Records& records,
               ListStart& listStart) {
    std::uint64_t state = seed;
    records.clear();
    listStart.assign(1, 0);
    int nextOrderId = 1;
    for (int list = 0; list < listCount; ++list) {
        const int count = (emptyOneIn > 0 && nextRandom(state) % emptyOneIn == 0) ? 0 : static_cast<int>(nextRandom(state) % (maxPerList + 1));
        int timeMs = 0;
        for (int item = 0; item < count; ++item) {
            timeMs += static_cast<int>(nextRandom(state) % 5) * timeStepMs;  // pasos de 0: empates dentro de la lista
            records.push_back({timeMs, nextOrderId++});
        }
        listStart.push_back(static_cast<int>(records.size()));
    }
}

}  // namespace

int main() {
    accel::ThreadPool pool(4);
    constexpr int kMinute = 60000;

    // Ejemplo del material: dos listas de 5 y 4 (minutos). 8' es el 8 del courier 1 (orderId 12).
    {
        const Records records{{3 * kMinute, 1}, {8 * kMinute, 2}, {12 * kMinute, 3}, {15 * kMinute, 4}, {21 * kMinute, 5},
                              {5 * kMinute, 11}, {8 * kMinute, 12}, {13 * kMinute, 13}, {30 * kMinute, 14}};
        const ListStart listStart{0, 5, 9};
        const DeliveryRecord* first = records.data();
        const DeliveryRecord* second = records.data() + 5;
        // Co-rank de cada k: cuantos de las primeras k posiciones de la fusion salen de `first`.
        const int expectedFromFirst[10] = {0, 1, 1, 2, 2, 3, 3, 4, 5, 5};
        for (int outputCount = 0; outputCount <= 9; ++outputCount) {
            CHECK(accel::coRank(outputCount, first, 5, second, 4) == expectedFromFirst[outputCount]);
        }
        CHECK(accel::coRank(3, first, 5, second, 4) == 2);
        CHECK(accel::coRank(6, first, 5, second, 4) == 3);

        const Records expected{{3 * kMinute, 1}, {5 * kMinute, 11}, {8 * kMinute, 2}, {8 * kMinute, 12}, {12 * kMinute, 3},
                               {13 * kMinute, 13}, {15 * kMinute, 4}, {21 * kMinute, 5}, {30 * kMinute, 14}};
        CHECK(sameRecords(accel::mergeCourierListsCpu1(records, listStart), expected));  // el 8 del courier 0 va antes
        CHECK(sameRecords(accel::mergeCourierListsCpuN(records, listStart, pool), expected));
        CHECK(sameRecords(stableSortedReference(records), expected));
        // Reparto de la salida entre 3 hilos de 3 posiciones: el hilo 1 toma 1 de first y 2 de second.
        Records pieces(9);
        for (int threadIndex = 0; threadIndex < 3; ++threadIndex) {
            accel::mergeRoundRange(records.data(), pieces.data(), listStart.data(), 2, 3 * threadIndex, 3 * threadIndex + 3);
        }
        CHECK(sameRecords(pieces, expected));
        // Entregas por franja de 10 minutos: 4, 3, 1, 1.
        CHECK((accel::deliveriesPerBucketCpu(expected, 10 * kMinute) == std::vector<int>{4, 3, 1, 1}));
        CHECK((accel::deliveriesPerBucketBySearch(expected, 10 * kMinute) == std::vector<int>{4, 3, 1, 1}));
    }

    // Listas vacias: ninguna entrega, couriers sin entregas, 8 listas con 7 vacias, todo en el mismo ms.
    {
        CHECK(accel::mergeCourierListsCpu1({}, ListStart{0}).empty());
        CHECK(accel::mergeCourierListsCpu1({}, ListStart{0, 0, 0, 0}).empty());
        CHECK(accel::mergeCourierListsCpuN({}, ListStart{0, 0, 0, 0}, pool).empty());
        CHECK(accel::deliveriesPerBucketCpu({}, 3600000).empty());

        const Records one{{42, 7}};
        for (const ListStart& listStart : {ListStart{0, 1}, ListStart{0, 0, 1}, ListStart{0, 1, 1}, ListStart{0, 0, 0, 1, 1}}) {
            CHECK(sameRecords(accel::mergeCourierListsCpu1(one, listStart), one));
            CHECK(sameRecords(accel::mergeCourierListsCpuN(one, listStart, pool), one));
        }

        for (int fullList = 0; fullList < 8; ++fullList) {  // 8 listas, solo una con entregas
            Records records;
            ListStart listStart{0};
            for (int list = 0; list < 8; ++list) {
                if (list == fullList) {
                    for (int item = 0; item < 25; ++item) records.push_back({1000, 100 + item});  // todo en el mismo ms
                }
                listStart.push_back(static_cast<int>(records.size()));
            }
            CHECK(sameRecords(accel::mergeCourierListsCpu1(records, listStart), records));
            CHECK(sameRecords(accel::mergeCourierListsCpuN(records, listStart, pool), records));
        }
    }

    // Empates entre couriers: gana la lista de la izquierda y el orden de cada lista se conserva.
    {
        const Records records{{10, 1}, {10, 2}, {20, 3}, {10, 4}, {20, 5}, {10, 6}, {10, 7}};
        const ListStart listStart{0, 3, 5, 7};
        const Records expected{{10, 1}, {10, 2}, {10, 4}, {10, 6}, {10, 7}, {20, 3}, {20, 5}};
        CHECK(sameRecords(accel::mergeCourierListsCpu1(records, listStart), expected));
        CHECK(sameRecords(stableSortedReference(records), expected));
        for (const unsigned threads : {1u, 2u, 3u, 8u}) {
            accel::ThreadPool threadPool(threads);
            CHECK(sameRecords(accel::mergeCourierListsCpuN(records, listStart, threadPool), expected));
        }
    }

    // Pseudoaleatorio: cpu1 == cpuN (varios pools) == stable_sort, con muchos empates y couriers sin entregas.
    for (const int listCount : {1, 2, 3, 7, 100, 1999}) {
        for (const int timeStepMs : {1000, 1}) {
            Records records;
            ListStart listStart;
            makeLists(77 + listCount, listCount, 40, timeStepMs, 4, records, listStart);
            const Records expected = stableSortedReference(records);
            CHECK(sameRecords(accel::mergeCourierListsCpu1(records, listStart), expected));
            for (const unsigned threads : {1u, 2u, 3u, 4u, 7u, 16u}) {
                accel::ThreadPool threadPool(threads);
                CHECK(sameRecords(accel::mergeCourierListsCpuN(records, listStart, threadPool), expected));
            }
            if (!expected.empty()) {
                const std::vector<int> byCounting = accel::deliveriesPerBucketCpu(expected, 7000);
                CHECK(byCounting == accel::deliveriesPerBucketBySearch(expected, 7000));
                int total = 0;
                for (const int perBucket : byCounting) total += perBucket;
                CHECK(total == static_cast<int>(expected.size()));
            }
        }
    }

    // Texto para stderr y resumen completo (el build de pruebas no tiene CUDA: siempre respaldo CPU).
    {
        CHECK(accel::formatDeliveriesPerHour({4, 0, 7}, "cpu") ==
              "[estadisticas] entregas por hora simulada: h0 4, h1 0, h2 7 (backend=cpu)");
        CHECK(accel::formatDeliveriesPerHour({}, "cpu") == "[estadisticas] entregas por hora simulada: sin entregas");

        const Records records{{3 * kMinute, 1}, {8 * kMinute, 2}, {70 * kMinute, 3}, {5 * kMinute, 11}, {130 * kMinute, 12}};
        const ListStart listStart{0, 3, 3, 5};  // el courier del medio no tiene entregas
        accel::GpuWorker brokenWorker([] { throw std::runtime_error("sin GPU"); });
        accel::GpuWorker healthyWorker([] {});
        for (accel::GpuWorker* worker : {static_cast<accel::GpuWorker*>(nullptr), &brokenWorker, &healthyWorker}) {
            const accel::DeliveryTimelineSummary summary = accel::deliveryTimelineSummary(records, listStart, worker);
            CHECK((summary.deliveriesPerHour == std::vector<int>{3, 1, 1}));
            CHECK(summary.total == 5);
            CHECK(!summary.usedGpu);
        }
        brokenWorker.stop();
        healthyWorker.stop();
        CHECK(accel::deliveryTimelineSummary({}, ListStart{0}, nullptr).total == 0);
    }

    // OrderBook::deliveries: las entregas por courier, ordenadas por courier, hora y pedido.
    {
        const SimClock clock(60.0);
        EventLog log("tests/bin/timeline_book.log", clock);
        OrderBook book(10, clock, log);
        CHECK(book.deliveries().empty());
        for (OrderId id = 1; id <= 4; ++id) {
            Order order;
            order.id = id;
            order.restaurantId = "r0";
            order.pickupNode = "n15";
            order.dropoffNode = "n420";
            CHECK(book.admit(order));
        }
        CHECK(book.assign(1, 9));
        CHECK(book.assign(2, 3));
        CHECK(book.assign(3, 9));
        for (const OrderId id : {OrderId{1}, OrderId{3}, OrderId{2}}) {
            const CourierId courier = id == 2 ? 3 : 9;
            CHECK(book.pickedUp(id, courier));
            CHECK(book.delivered(id, courier));
        }
        const std::vector<DeliveryEntry> entries = book.deliveries();
        CHECK(entries.size() == 3);  // el pedido 4 sigue abierto
        CHECK(entries[0].courier == 3 && entries[0].orderId == 2);
        CHECK(entries[1].courier == 9 && entries[1].orderId == 1);
        CHECK(entries[2].courier == 9 && entries[2].orderId == 3);
        CHECK(entries[1].deliveredMs <= entries[2].deliveredMs);
    }

    std::cout << "test_delivery_timeline: OK\n";
    return 0;
}
