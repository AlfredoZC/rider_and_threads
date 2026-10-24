#include "CourierActor.h"

#include "Config.h"

#include <cassert>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {

std::int64_t realElapsedMs(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - from).count();
}

double cpuMs() {
    return 1000.0 * static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}

}

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    const SimClock clock(60.0);
    EventLog log("tests/bin/courier_actor.log", clock);
    OrderBook book(cfg.maxPending, clock, log);
    Restaurants restaurants;
    for (const RestaurantConfig& r : cfg.restaurants) {
        restaurants.emplace(r.id,
                            std::make_unique<Restaurant>(r.id, r.pickupSlots));
    }
    const SimContext ctx{graph, clock, log, book, restaurants};

    {
        CourierActor actor(1, cfg.startNode, 3, ctx);
        const Courier view = actor.snapshot();
        assert(actor.id() == 1);
        assert(view.id == 1);
        assert(view.state == CourierState::idle);
        assert(currentNode(view) == "n311");
        assert(view.bag.empty());
        assert(hasRoom(view));
        std::cout << "  snapshot() devuelve el estado inicial correcto\n";
    }

    {
        const int kCouriers = 8;
        std::vector<std::unique_ptr<CourierActor>> fleet;
        for (int i = 0; i < kCouriers; ++i) {
            fleet.push_back(std::make_unique<CourierActor>(
                static_cast<CourierId>(i) + 1, cfg.startNode, 3, ctx));
        }

        std::vector<std::thread> threads;
        for (auto& actor : fleet) {
            threads.emplace_back(&CourierActor::run, actor.get());
        }

        const double cpuBefore = cpuMs();
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        const double cpuUsed = cpuMs() - cpuBefore;

        std::cout << "  8 couriers ociosos 400 ms consumieron " << cpuUsed << " ms de CPU\n";
        assert(cpuUsed < 100.0);

        for (auto& actor : fleet) {
            const Courier view = actor->snapshot();
            assert(view.state == CourierState::idle);
        }
        std::cout << "  snapshot() funciona mientras los hilos esperan\n";

        const auto t0 = std::chrono::steady_clock::now();
        for (auto& actor : fleet) { actor->requestStop(); }
        for (std::thread& t : threads) { t.join(); }
        const std::int64_t stopMs = realElapsedMs(t0);
        std::cout << "  los 8 hilos terminaron en " << stopMs << " ms\n";
        assert(stopMs < 1000);
    }

    {
        CourierActor actor(1, cfg.startNode, 3, ctx);
        actor.requestStop();
        std::thread t(&CourierActor::run, &actor);
        t.join();
        std::cout << "  requestStop() antes de arrancar el hilo no lo deja colgado\n";
    }

    {
        const int kRounds = 300;
        for (int i = 0; i < kRounds; ++i) {
            CourierActor actor(1, cfg.startNode, 3, ctx);
            std::thread t(&CourierActor::run, &actor);
            if (i % 3 == 0) {
                std::this_thread::yield();
            }
            actor.requestStop();
            actor.requestStop();
            t.join();
        }
        std::cout << "  " << kRounds << " ciclos arrancar/parar sin un solo cuelgue\n";
    }

    {
        CourierActor actor(1, cfg.startNode, 2, ctx);

        Order a;
        a.id = 1;
        Order b;
        b.id = 2;
        Order c;
        c.id = 3;

        assert(actor.reserve(a));
        assert(actor.reserve(b));
        assert(!actor.reserve(c));
        assert(actor.snapshot().bag.empty());

        actor.commitReservation(a);
        assert(actor.snapshot().bag.size() == 1);
        assert(!actor.reserve(c));

        actor.cancelReservation();
        assert(actor.snapshot().bag.size() == 1);
        assert(actor.reserve(c));
        actor.commitReservation(c);
        assert(actor.snapshot().bag.size() == 2);
        assert(!actor.reserve(b));
        std::cout << "  reserve/commit/cancel respetan bagCapacity sin tocar la bolsa\n";
    }

    {
        CourierActor actor(1, cfg.startNode, 3, ctx);
        actor.requestStop();
        Order a;
        a.id = 1;
        assert(!actor.reserve(a));
        std::cout << "  un courier detenido rechaza la reserva\n";
    }

    std::cout << "  courier actor ok\n";
}
