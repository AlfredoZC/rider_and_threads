#include "accel/OpenOrders.h"

#include <algorithm>
#include <climits>
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

using accel::kStillOpen;
using accel::OpenOrdersResult;
using accel::OrderLifetime;

std::uint64_t nextRandom(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t mixed = state;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
    return mixed ^ (mixed >> 31);
}

bool sameResult(const OpenOrdersResult& left, const OpenOrdersResult& right) {
    return left.openOrders == right.openOrders && left.peakOpenOrders == right.peakOpenOrders &&
           left.peakEvent == right.peakEvent;
}

}  // namespace

int main() {
    accel::ThreadPool pool(4);

    // Ejemplo a mano del material: 5 pedidos (A a E), 8 eventos; a las 3:00 E se crea y C se rechaza.
    {
        const std::vector<OrderLifetime> orders{
            {0, 125000},        // A: creado 0:00, entregado 2:05
            {40000, 255000},    // B: 0:40 -> 4:15
            {70000, 180000},    // C: 1:10 -> rechazado 3:00
            {150000, kStillOpen},  // D: 2:30, sigue abierto
            {180000, kStillOpen},  // E: 3:00, sigue abierto
        };
        const accel::OrderEvents events = accel::buildOrderEvents(orders);
        CHECK((events.change == std::vector<std::int32_t>{1, 1, 1, -1, 1, 1, -1, -1}));
        CHECK((events.timeMs == std::vector<std::int32_t>{0, 40000, 70000, 125000, 150000, 180000, 180000, 255000}));
        const OpenOrdersResult single = accel::openOrdersCpu1(events.change);
        CHECK((single.openOrders == std::vector<std::int32_t>{1, 2, 3, 2, 3, 4, 3, 2}));
        CHECK(single.peakOpenOrders == 4);
        CHECK(single.peakEvent == 5);  // el +1 de E va antes del -1 de C: el pico es 4
        CHECK(single.openOrders.back() == 2);
        CHECK(sameResult(single, accel::openOrdersCpuN(events.change, pool)));
    }

    // Vacio y un evento.
    {
        const std::vector<std::int32_t> none;
        const OpenOrdersResult empty = accel::openOrdersCpu1(none);
        CHECK(empty.openOrders.empty() && empty.peakOpenOrders == 0 && empty.peakEvent == 0);
        CHECK(sameResult(empty, accel::openOrdersCpuN(none, pool)));
        CHECK(accel::buildOrderEvents({}).change.empty());

        const std::vector<std::int32_t> one{1};
        const OpenOrdersResult single = accel::openOrdersCpu1(one);
        CHECK((single.openOrders == std::vector<std::int32_t>{1}));
        CHECK(single.peakOpenOrders == 1 && single.peakEvent == 0);
        CHECK(sameResult(single, accel::openOrdersCpuN(one, pool)));
    }

    // Un pedido rechazado en el mismo milisegundo en que se creo nunca deja la cuenta en negativo,
    // aunque aparezca primero en la lista; y el pico empatado es el primero.
    {
        const std::vector<OrderLifetime> orders{{500, 500}, {100, 100}, {100, kStillOpen}};
        const accel::OrderEvents events = accel::buildOrderEvents(orders);
        CHECK((events.change == std::vector<std::int32_t>{1, 1, -1, 1, -1}));
        const OpenOrdersResult single = accel::openOrdersCpu1(events.change);
        CHECK((single.openOrders == std::vector<std::int32_t>{1, 2, 1, 2, 1}));
        CHECK(single.peakOpenOrders == 2 && single.peakEvent == 1);
    }

    // Pseudoaleatorio: nunca negativo, termina en los abiertos, pico = maximo, cpu1 == cpuN para varios pools.
    for (const std::size_t orderCount : {1u, 2u, 3u, 7u, 1000u, 20011u}) {
        std::uint64_t state = 99 + orderCount;
        std::vector<OrderLifetime> orders;
        std::size_t openAtEnd = 0;
        for (std::size_t orderIndex = 0; orderIndex < orderCount; ++orderIndex) {
            const std::int64_t createdAtMs = static_cast<std::int64_t>(nextRandom(state) % 100000);
            const std::int64_t lifeMs = static_cast<std::int64_t>(nextRandom(state) % 4000);  // incluye 0
            if (nextRandom(state) % 10 == 0) {
                orders.push_back({createdAtMs, kStillOpen});
                ++openAtEnd;
            } else {
                orders.push_back({createdAtMs, createdAtMs + lifeMs});
            }
        }
        const accel::OrderEvents events = accel::buildOrderEvents(orders);
        const OpenOrdersResult single = accel::openOrdersCpu1(events.change);
        CHECK(single.openOrders.size() == events.change.size());
        CHECK(*std::min_element(single.openOrders.begin(), single.openOrders.end()) >= 0);
        CHECK(static_cast<std::size_t>(single.openOrders.back()) == openAtEnd);
        const std::int32_t maximum = *std::max_element(single.openOrders.begin(), single.openOrders.end());
        CHECK(single.peakOpenOrders == maximum);
        CHECK(single.openOrders[single.peakEvent] == maximum);
        for (std::size_t eventIndex = 0; eventIndex < single.peakEvent; ++eventIndex) {
            CHECK(single.openOrders[eventIndex] < maximum);  // el primero que lo alcanza
        }
        for (const unsigned threads : {1u, 2u, 3u, 8u}) {
            accel::ThreadPool threadPool(threads);
            CHECK(sameResult(single, accel::openOrdersCpuN(events.change, threadPool)));
        }
    }

    // Texto para stderr.
    {
        CHECK(accel::formatOpenOrders(4, 180000, 2, "cpu") ==
              "[estadisticas] pedidos en curso: pico 4 en t=180.0s, abiertos al final 2 (backend=cpu)");
        CHECK(accel::formatOpenOrders(0, 0, 0, "cpu") == "[estadisticas] pedidos en curso: sin pedidos");
    }

    // Resumen completo: sin worker, con worker roto y con uno sano pero sin CUDA dan lo mismo.
    {
        const std::vector<OrderLifetime> orders{{0, 125000}, {40000, 255000}, {70000, 180000},
                                                {150000, kStillOpen}, {180000, kStillOpen}};
        accel::GpuWorker brokenWorker([] { throw std::runtime_error("sin GPU"); });
        accel::GpuWorker healthyWorker([] {});
        for (accel::GpuWorker* worker : {static_cast<accel::GpuWorker*>(nullptr), &brokenWorker, &healthyWorker}) {
            const accel::OpenOrdersSummary summary = accel::openOrdersSummary(orders, worker);
            CHECK(summary.peakOpenOrders == 4);
            CHECK(summary.peakTimeMs == 180000);
            CHECK(summary.openAtEnd == 2);
            CHECK(!summary.usedGpu);  // el build de pruebas no tiene CUDA
        }
        brokenWorker.stop();
        healthyWorker.stop();
        const accel::OpenOrdersSummary emptySummary = accel::openOrdersSummary({}, nullptr);
        CHECK(emptySummary.peakOpenOrders == 0 && emptySummary.openAtEnd == 0);
    }

    // OrderBook::orderSpans: creacion y cierre (entregado o rechazado) de cada pedido.
    {
        const SimClock clock(60.0);
        EventLog log("tests/bin/open_orders_book.log", clock);
        OrderBook book(10, clock, log);
        CHECK(book.orderSpans().empty());
        for (OrderId id = 1; id <= 4; ++id) {
            Order order;
            order.id = id;
            order.restaurantId = "r0";
            order.pickupNode = "n15";
            order.dropoffNode = "n420";
            CHECK(book.admit(order));
        }
        CHECK(book.assign(1, 7));
        CHECK(book.pickedUp(1, 7));
        CHECK(book.delivered(1, 7));
        CHECK(book.reject(2, RejectReason::noCourier));
        CHECK(book.assignUnlessExpired(3, 7, -1) == OrderBook::AssignResult::expired);  // vencido: tambien se cierra
        const std::vector<OrderSpan> spans = book.orderSpans();
        CHECK(spans.size() == 4);
        std::size_t closedCount = 0;
        for (const OrderSpan& span : spans) {
            CHECK(span.createdMs >= 0);
            if (span.closedMs != kNoTime) {
                CHECK(span.closedMs >= span.createdMs);
                ++closedCount;
            }
        }
        CHECK(closedCount == 3);  // entregado, rechazado y vencido; el 4 sigue abierto
    }

    std::cout << "test_open_orders: OK\n";
    return 0;
}
