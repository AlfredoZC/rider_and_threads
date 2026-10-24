#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "CourierActor.h"
#include "OrderGenerator.h"

namespace {

std::int64_t realElapsedMs(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - from).count();
}

Order makeOrder(OrderId id, const std::string& pickup, const std::string& dropoff,
                std::int64_t prepMs) {
    Order o;
    o.id = id;
    o.restaurantId = "r0";
    o.pickupNode = pickup;
    o.dropoffNode = dropoff;
    o.prepMs = prepMs;
    return o;
}

std::vector<nlohmann::json> readLog(const std::string& path) {
    std::ifstream in(path);
    assert(in);
    std::vector<nlohmann::json> events;
    std::string line;
    while (std::getline(in, line)) {
        events.push_back(nlohmann::json::parse(line));
    }
    return events;
}

bool edgeExists(const CityGraph& graph, const std::string& from, const std::string& to) {
    const NodeIndex a = graph.indexOf(from);
    const NodeIndex b = graph.indexOf(to);
    if (a == kNoNode || b == kNoNode) {
        return false;
    }
    for (const Edge& e : graph.edgesFrom(a)) {
        if (e.to == b) {
            return true;
        }
    }
    return false;
}

}

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);

    const std::string restaurantNode = cfg.restaurants.front().node;
    const std::string clientNode = cfg.nodes.back().id;

    {
        const SimClock clock(600.0);
        const std::string logPath = "tests/bin/ride_full.log";
        EventLog log(logPath, clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

        CourierActor courier(1, cfg.startNode, cfg.bagCapacity, ctx);

        const OrderId id = 1;
        assert(book.admit(makeOrder(id, restaurantNode, clientNode, 0)));
        assert(book.assign(id, 1));
        assert(courier.offer(*book.find(id)));

        std::thread worker(&CourierActor::run, &courier);

        std::set<std::pair<std::string, std::string>> segments;
        std::set<std::string> statesSeen;
        const auto t0 = std::chrono::steady_clock::now();
        bool delivered = false;
        while (realElapsedMs(t0) < 20000) {
            const Courier view = courier.snapshot();
            statesSeen.insert(toString(view.state));
            if (view.fromNode != view.toNode) {
                segments.emplace(view.fromNode, view.toNode);
            }
            const double ratio = positionRatio(view, clock.nowMs());
            assert(ratio >= 0.0 && ratio <= 1.0);

            const auto current = book.find(id);
            if (current->state == OrderState::delivered) {
                delivered = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        courier.requestStop();
        worker.join();

        assert(delivered);
        std::cout << "  entrego la orden; se observaron " << segments.size()
                  << " tramos distintos\n";

        assert(!segments.empty());
        for (const auto& s : segments) {
            assert(edgeExists(graph, s.first, s.second));
        }
        std::cout << "  los " << segments.size()
                  << " tramos son calles existentes y en el sentido correcto\n";

        assert(statesSeen.count("toPickup") == 1 || statesSeen.count("atRestaurant") == 1 ||
               statesSeen.count("toDropoff") == 1);

        const Courier done = courier.snapshot();
        assert(done.bag.empty());
        assert(done.delivered == 1);
        assert(done.state == CourierState::idle);
        assert(currentNode(done) == clientNode);
        assert(done.fromNode == done.toNode);

        const std::vector<nlohmann::json> events = readLog(logPath);
        std::vector<std::string> names;
        std::int64_t last = -1;
        for (const nlohmann::json& e : events) {
            names.push_back(e["event"].get<std::string>());
            const std::int64_t t = e["t"].get<std::int64_t>();
            assert(t >= last);
            last = t;
        }
        const std::vector<std::string> expected = {
            "orderCreated", "orderAssigned", "pickupStarted", "orderPickedUp",
            "orderDelivered"};
        assert(names == expected);
        std::cout << "  el log tiene los 5 eventos del §6 en orden\n";

        const OrderStats stats = book.stats();
        assert(stats.delivered == 1);
        assert(stats.pending == 0);
        assert(stats.created == stats.delivered + stats.rejected + stats.pending);
        assert(stats.maxDeliveryMs > 0);
        std::cout << "  tiempo de entrega: " << stats.maxDeliveryMs / 1000 << " s simulados\n";
    }

    {
        const SimClock clock(2000.0);
        EventLog log("tests/bin/ride_prep.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};
        CourierActor courier(1, restaurantNode, cfg.bagCapacity, ctx);

        const OrderId id = 1;
        assert(book.admit(makeOrder(id, restaurantNode, clientNode, 400000)));
        assert(book.assign(id, 1));
        assert(courier.offer(*book.find(id)));

        std::thread worker(&CourierActor::run, &courier);
        const auto t0 = std::chrono::steady_clock::now();
        while (realElapsedMs(t0) < 2000 &&
               book.find(id)->state != OrderState::pickedUp) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const Order afterPickup = *book.find(id);
        courier.requestStop();
        worker.join();

        assert(afterPickup.pickedUpMs >= afterPickup.createdMs + afterPickup.prepMs);
        std::cout << "  espero la preparacion: retiro a los "
                  << afterPickup.pickedUpMs / 1000 << " s, comida lista a los "
                  << (afterPickup.createdMs + afterPickup.prepMs) / 1000 << " s\n";
    }

    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/ride_stop.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};
        CourierActor courier(1, cfg.startNode, cfg.bagCapacity, ctx);

        const OrderId id = 1;
        assert(book.admit(makeOrder(id, restaurantNode, clientNode, 0)));
        assert(book.assign(id, 1));
        assert(courier.offer(*book.find(id)));

        std::thread worker(&CourierActor::run, &courier);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        const Courier moving = courier.snapshot();
        assert(moving.state == CourierState::toPickup);
        assert(moving.fromNode != moving.toNode);

        const auto t0 = std::chrono::steady_clock::now();
        courier.requestStop();
        worker.join();
        const std::int64_t stopMs = realElapsedMs(t0);
        std::cout << "  parada a mitad de viaje en " << stopMs << " ms\n";
        assert(stopMs < 500);

        const Order left = *book.find(id);
        assert(!isFinalState(left.state));
        assert(book.stats().delivered == 0);
    }

    std::cout << "  courier ride ok\n";
}
