// Pruebas del Dispatcher.
//
// Cuatro requisitos del §4 que solo se pueden comprobar aca:
//
//   1. Elige al courier con MEJOR tiempo estimado, no al primero ni al mas cercano.
//   2. Las estimaciones se calculan AL MISMO TIEMPO. Se mide: ocho estimaciones que
//      tardan 100 ms cada una tienen que llevar ~100 ms en total, no ~800.
//   3. Espera a lo sumo quoteTimeoutMs REALES y descarta las tardias. Y lo hace sin
//      bloquearse: el destructor de un future de std::async espera a su tarea, asi que
//      si las tardias se destruyeran en el plazo, el plazo no serviria de nada.
//   4. Una estimacion que lanza una excepcion no tumba el programa.
#include "Dispatcher.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "OrderFactory.h"

namespace {

std::int64_t realElapsedMs(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - from).count();
}

Order makeOrder(OrderId id, const std::string& pickup, const std::string& dropoff) {
    Order o;
    o.id = id;
    o.restaurantId = "r0";
    o.pickupNode = pickup;
    o.dropoffNode = dropoff;
    return o;
}

// Espera a que se cumpla una condicion, hasta un limite. Devuelve si se cumplio.
template <typename Fn>
bool waitUntil(Fn condition, std::int64_t limitMs) {
    const auto t0 = std::chrono::steady_clock::now();
    while (realElapsedMs(t0) < limitMs) {
        if (condition()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return condition();
}

}  // namespace

int main() {
    Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    const std::string restaurantNode = cfg.restaurants.front().node;
    const std::string clientNode = cfg.nodes.back().id;

    // --- elige al de mejor tiempo estimado, no a cualquiera
    {
        const SimClock clock(1.0);   // tiempo real: nadie llega a moverse
        EventLog log("tests/bin/disp_best.log", clock);
        OrderBook book(50, clock, log);
        const SimContext ctx{graph, clock, log, book};

        // tres couriers a distancias muy distintas del restaurante
        std::vector<std::unique_ptr<CourierActor>> fleet;
        fleet.push_back(std::make_unique<CourierActor>(1, cfg.nodes.back().id, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(2, restaurantNode, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(3, cfg.startNode, 3, ctx));

        // el que ya esta en el restaurante tiene que ganar
        Router router(graph);
        const Order order = makeOrder(1, restaurantNode, clientNode);
        std::int64_t bestMs = -1;
        CourierId expected = kNoCourier;
        for (const auto& actor : fleet) {
            const std::int64_t ms =
                estimateDeliveryMs(router, graph, actor->snapshot(), order);
            if (bestMs < 0 || ms < bestMs) {
                bestMs = ms;
                expected = actor->id();
            }
        }

        Dispatcher dispatcher(cfg, ctx, fleet);
        assert(book.admit(order));
        std::thread worker(&Dispatcher::run, &dispatcher);
        assert(waitUntil([&] { return book.find(1)->state == OrderState::assigned; }, 3000));
        dispatcher.requestStop();
        worker.join();

        const Order assigned = *book.find(1);
        std::cout << "  eligio al courier " << assigned.assignedCourier
                  << " (mejor estimacion: " << bestMs / 1000 << " s, courier "
                  << expected << ")\n";
        assert(assigned.assignedCourier == expected);
        assert(dispatcher.quotesRequested() == 3);
    }

    // --- LA prueba del paralelismo: ocho estimaciones lentas, una sola espera
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_parallel.log", clock);
        OrderBook book(50, clock, log);
        const SimContext ctx{graph, clock, log, book};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        for (int i = 0; i < 8; ++i) {
            fleet.push_back(std::make_unique<CourierActor>(
                static_cast<CourierId>(i) + 1, cfg.startNode, 3, ctx));
        }

        cfg.quoteTimeoutMs = 5000;   // amplio: queremos medir, no cortar
        Dispatcher dispatcher(cfg, ctx, fleet);
        std::atomic<int> concurrent{0};
        std::atomic<int> maxConcurrent{0};
        dispatcher.setQuoteFunction(
            [&](Router&, const CityGraph&, const Courier&, const Order&) {
                const int now = ++concurrent;
                int previous = maxConcurrent.load();
                while (previous < now && !maxConcurrent.compare_exchange_weak(previous, now)) {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                --concurrent;
                return std::int64_t{1000};
            });

        assert(book.admit(makeOrder(1, restaurantNode, clientNode)));
        const auto t0 = std::chrono::steady_clock::now();
        std::thread worker(&Dispatcher::run, &dispatcher);
        assert(waitUntil([&] { return book.find(1)->state == OrderState::assigned; }, 5000));
        const std::int64_t ms = realElapsedMs(t0);
        dispatcher.requestStop();
        worker.join();

        std::cout << "  8 estimaciones de 100 ms tardaron " << ms
                  << " ms en total; hubo hasta " << maxConcurrent.load()
                  << " corriendo a la vez\n";
        assert(maxConcurrent.load() >= 2);   // de verdad simultaneas
        assert(ms < 600);                    // secuencialmente habrian sido ~800
    }

    // --- el plazo se respeta, y esperar no bloquea de mas
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_timeout.log", clock);
        OrderBook book(50, clock, log);
        const SimContext ctx{graph, clock, log, book};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        for (int i = 0; i < 4; ++i) {
            fleet.push_back(std::make_unique<CourierActor>(
                static_cast<CourierId>(i) + 1, cfg.startNode, 3, ctx));
        }

        cfg.quoteTimeoutMs = 150;
        Dispatcher dispatcher(cfg, ctx, fleet);
        dispatcher.setQuoteFunction(
            [](Router&, const CityGraph&, const Courier& c, const Order&) {
                // el courier 1 contesta enseguida; los demas tardan muchisimo
                if (c.id == 1) {
                    return std::int64_t{5000};
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                return std::int64_t{10};   // habria ganado, pero llega tarde
            });

        assert(book.admit(makeOrder(1, restaurantNode, clientNode)));
        const auto t0 = std::chrono::steady_clock::now();
        std::thread worker(&Dispatcher::run, &dispatcher);
        assert(waitUntil([&] { return book.find(1)->state == OrderState::assigned; }, 3000));
        const std::int64_t ms = realElapsedMs(t0);

        const Order assigned = *book.find(1);
        std::cout << "  con plazo de 150 ms asigno en " << ms << " ms al courier "
                  << assigned.assignedCourier << ", descartando "
                  << dispatcher.quotesLate() << " estimaciones tardias\n";
        assert(assigned.assignedCourier == 1);   // gano el unico que contesto a tiempo
        assert(dispatcher.quotesLate() >= 3);
        assert(ms < 1200);   // si el future tardio bloqueara, serian 1500+

        dispatcher.requestStop();
        worker.join();   // y al cerrar espera a las tardias: nada queda abandonado
    }

    // --- una estimacion que lanza no tumba el programa (§4)
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_throw.log", clock);
        OrderBook book(50, clock, log);
        const SimContext ctx{graph, clock, log, book};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        for (int i = 0; i < 4; ++i) {
            fleet.push_back(std::make_unique<CourierActor>(
                static_cast<CourierId>(i) + 1, cfg.startNode, 3, ctx));
        }

        cfg.quoteTimeoutMs = 2000;
        Dispatcher dispatcher(cfg, ctx, fleet);
        dispatcher.setQuoteFunction(
            [](Router&, const CityGraph&, const Courier& c, const Order&) -> std::int64_t {
                if (c.id != 3) {
                    throw std::runtime_error("fallo al estimar");
                }
                return 1234;   // el unico que no falla
            });

        assert(book.admit(makeOrder(1, restaurantNode, clientNode)));
        std::thread worker(&Dispatcher::run, &dispatcher);
        assert(waitUntil([&] { return book.find(1)->state == OrderState::assigned; }, 3000));
        dispatcher.requestStop();
        worker.join();

        assert(book.find(1)->assignedCourier == 3);
        std::cout << "  3 de 4 estimaciones lanzaron excepcion y la orden igual se asigno\n";
    }

    // --- una orden que nadie acepta a tiempo se rechaza con noCourier (§4)
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_nocourier.log", clock);
        OrderBook book(50, clock, log);
        const SimContext ctx{graph, clock, log, book};

        std::vector<std::unique_ptr<CourierActor>> fleet;   // flota vacia: nadie puede
        cfg.acceptTimeoutMs = 0;                            // vence de inmediato
        Dispatcher dispatcher(cfg, ctx, fleet);

        assert(book.admit(makeOrder(1, restaurantNode, clientNode)));
        std::thread worker(&Dispatcher::run, &dispatcher);
        assert(waitUntil([&] { return book.find(1)->state == OrderState::rejected; }, 3000));
        dispatcher.requestStop();
        worker.join();

        assert(book.find(1)->rejectReason == RejectReason::noCourier);
        const OrderStats s = book.stats();
        assert(s.created == s.delivered + s.rejected + s.pending);
        std::cout << "  sin couriers disponibles rechaza con noCourier\n";
    }

    std::cout << "  dispatcher ok\n";
}
