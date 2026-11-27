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
#include "accel/TravelTimeTable.h"

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


// Corre un pedido con 3 couriers (1 lejos, 2 en el restaurante, 3 en el inicio) y devuelve a quien se lo asigno
// el dispatcher con el SimContext que lleva `table` (nullptr = sin tabla).
CourierId assignedCourierWith(const Config& cfg, const CityGraph& graph,
                              const std::shared_ptr<const accel::TravelTimeTable>& table,
                              const std::string& logPath) {
    const SimClock clock(1.0);
    EventLog log(logPath, clock);
    OrderBook book(50, clock, log);
    Restaurants restaurants;
    for (const RestaurantConfig& r : cfg.restaurants) {
        restaurants.emplace(r.id, std::make_unique<Restaurant>(r.id, r.pickupSlots));
    }
    const SimContext ctx{graph, clock, log, book, restaurants, nullptr, table};

    std::vector<std::unique_ptr<CourierActor>> fleet;
    fleet.push_back(std::make_unique<CourierActor>(1, cfg.nodes.back().id, 3, ctx));
    fleet.push_back(std::make_unique<CourierActor>(2, cfg.restaurants.front().node, 3, ctx));
    fleet.push_back(std::make_unique<CourierActor>(3, cfg.startNode, 3, ctx));

    Dispatcher dispatcher(cfg, ctx, fleet);
    assert(book.admit(makeOrder(1, cfg.restaurants.front().node, cfg.nodes.back().id)));
    std::thread worker(&Dispatcher::run, &dispatcher);
    assert(waitUntil([&] { return book.find(1)->state == OrderState::assigned; }, 3000));
    dispatcher.requestStop();
    worker.join();
    return book.find(1)->assignedCourier;
}

// Tabla "trucada" de `nodeCount` nodos: todo cuesta 1.000.000 ms salvo la celda (fromNode, toNode) que cuesta 1 ms.
// No se parece a los tiempos reales del Router a proposito: asi se sabe de donde salio una estimacion.
std::shared_ptr<const accel::TravelTimeTable> riggedTable(std::size_t nodeCount, std::size_t fromNode,
                                                          std::size_t toNode) {
    std::vector<std::int32_t> cells(nodeCount * nodeCount, 1000000);
    cells[fromNode * nodeCount + toNode] = 1;
    return std::make_shared<const accel::TravelTimeTable>(nodeCount, std::move(cells));
}
}

int main() {
    Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    const std::string restaurantNode = cfg.restaurants.front().node;
    const std::string clientNode = cfg.nodes.back().id;

    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_best.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        fleet.push_back(std::make_unique<CourierActor>(1, cfg.nodes.back().id, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(2, restaurantNode, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(3, cfg.startNode, 3, ctx));

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

    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_parallel.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        for (int i = 0; i < 8; ++i) {
            fleet.push_back(std::make_unique<CourierActor>(
                static_cast<CourierId>(i) + 1, cfg.startNode, 3, ctx));
        }

        cfg.quoteTimeoutMs = 5000;
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

        const std::size_t trabajadores = dispatcher.quoteWorkers();
        std::cout << "  8 estimaciones de 100 ms tardaron " << ms << " ms en total con "
                  << trabajadores << " trabajadores; hubo hasta " << maxConcurrent.load()
                  << " corriendo a la vez\n";

        const std::int64_t kTareas = 8;
        const std::int64_t kDuracionMs = 100;
        const std::int64_t secuencial = kTareas * kDuracionMs;

        if (trabajadores < 2) {
            std::cout << "  (una sola CPU disponible: no hay paralelismo que medir)\n";
        } else {
            const std::int64_t tandas =
                (kTareas + static_cast<std::int64_t>(trabajadores) - 1) /
                static_cast<std::int64_t>(trabajadores);
            assert(maxConcurrent.load() >= 2);
            assert(ms < secuencial);
            assert(ms < tandas * kDuracionMs + 500);
        }
    }

    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_timeout.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        for (int i = 0; i < 4; ++i) {
            fleet.push_back(std::make_unique<CourierActor>(
                static_cast<CourierId>(i) + 1, cfg.startNode, 3, ctx));
        }

        cfg.quoteTimeoutMs = 150;
        Dispatcher dispatcher(cfg, ctx, fleet);
        dispatcher.setQuoteFunction(
            [](Router&, const CityGraph&, const Courier& c, const Order&) {
                if (c.id == 1) {
                    return std::int64_t{5000};
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
                return std::int64_t{10};
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
        assert(assigned.assignedCourier == 1);
        assert(dispatcher.quotesLate() >= 3);
        assert(ms < 1200);

        dispatcher.requestStop();
        worker.join();
    }

    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_throw.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

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
                return 1234;
            });

        assert(book.admit(makeOrder(1, restaurantNode, clientNode)));
        std::thread worker(&Dispatcher::run, &dispatcher);
        assert(waitUntil([&] { return book.find(1)->state == OrderState::assigned; }, 3000));
        dispatcher.requestStop();
        worker.join();

        assert(book.find(1)->assignedCourier == 3);
        std::cout << "  3 de 4 estimaciones lanzaron excepcion y la orden igual se asigno\n";
    }

    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_nocourier.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        cfg.acceptTimeoutMs = 0;
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

    {
        Config sinSalida = loadConfig("config/tests/unreachable.json");
        const CityGraph grafoChico(sinSalida);
        const SimClock clock(1.0);
        EventLog log("tests/bin/disp_unreachable.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : sinSalida.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{grafoChico, clock, log, book, restaurants};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        fleet.push_back(std::make_unique<CourierActor>(1, "a", 3, ctx));

        sinSalida.acceptTimeoutMs = 3600000;
        Dispatcher dispatcher(sinSalida, ctx, fleet);

        assert(book.admit(makeOrder(1, "a", "x")));
        assert(book.admit(makeOrder(2, "a", "y")));
        assert(book.admit(makeOrder(3, "a", "b")));

        std::thread worker(&Dispatcher::run, &dispatcher);
        assert(waitUntil([&] {
            return book.find(1)->state == OrderState::rejected &&
                   book.find(2)->state == OrderState::rejected;
        }, 3000));
        dispatcher.requestStop();
        worker.join();

        assert(book.find(1)->rejectReason == RejectReason::unreachable);
        assert(book.find(2)->rejectReason == RejectReason::unreachable);
        assert(book.find(3)->state != OrderState::rejected);
        std::cout << "  un destino inalcanzable se rechaza con unreachable, y uno"
                     " alcanzable no\n";
    }

    {
        // El dispatcher estima con la tabla cuando el SimContext la lleva y es de este grafo; si no, con el Router.
        // Con el Router gana el courier 2 (esta en el restaurante). La tabla trucada solo hace barato ir del nodo del
        // courier 1 al restaurante: si el ganador es el 1, la estimacion salio de la tabla.
        // Config propia: los bloques anteriores dejaron cfg con plazos de prueba (acceptTimeoutMs = 0 rechaza todo).
        Config freshCfg = loadConfig("config/equipetrol.json");
        freshCfg.quoteTimeoutMs = 5000;
        const std::size_t courierOneNode = graph.indexOf(cfg.nodes.back().id);
        const std::size_t pickupNode = graph.indexOf(restaurantNode);
        assert(courierOneNode != pickupNode);

        const CourierId withRouter = assignedCourierWith(freshCfg, graph, nullptr, "tests/bin/disp_no_table.log");
        assert(withRouter == 2);

        const CourierId withTable = assignedCourierWith(
            freshCfg, graph, riggedTable(graph.nodeCount(), courierOneNode, pickupNode), "tests/bin/disp_table.log");
        assert(withTable == 1);

        // Tabla de OTRO tamano: no es de este grafo, se ignora y se estima con el Router.
        const CourierId withForeignTable = assignedCourierWith(
            freshCfg, graph, riggedTable(graph.nodeCount() - 1, 0, 1), "tests/bin/disp_foreign.log");
        assert(withForeignTable == 2);
        std::cout << "  con tabla del grafo estima con la tabla (courier " << withTable
                  << "); sin tabla o con una de otro tamano, con el Router (courier " << withRouter << ")\n";
    }

    std::cout << "  dispatcher ok\n";
}
