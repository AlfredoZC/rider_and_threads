// Pruebas de las averias y el traspaso todo-o-nada (§4, DP.6).
//
// Las ordenes de un courier averiado se dividen en dos grupos y cada uno necesita algo
// distinto:
//
//   - Las que todavia no retiro vuelven a la cola: nadie tiene su comida en la mano, asi
//     que no hace falta un traspaso entre couriers. El dispatcher las reparte de nuevo.
//   - Las que ya lleva encima SI necesitan un traspaso de verdad, y ahi se aplica el
//     todo-o-nada: o un companero se hace cargo de todas, o la averia no ocurre.
//
// La prueba central es la de averias simultaneas: es el interbloqueo de manual, porque
// el traspaso toma dos candados de courier a la vez. Un interbloqueo no falla, CUELGA,
// asi que la unica forma de buscarlo es provocar la carrera muchas veces.
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <thread>
#include <vector>

#include "CourierActor.h"

namespace {

Order makeOrder(OrderId id, const std::string& pickup, const std::string& dropoff) {
    Order o;
    o.id = id;
    o.restaurantId = "r0";
    o.pickupNode = pickup;
    o.dropoffNode = dropoff;
    return o;
}

std::int64_t realElapsedMs(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - from).count();
}

// Pone una orden en la bolsa del courier, en el estado indicado.
void darOrden(OrderBook& book, CourierActor& courier, const Order& order, bool retirada) {
    assert(book.admit(order));
    assert(book.assign(order.id, courier.id()));
    assert(courier.offer(*book.find(order.id)));
    if (retirada) {
        assert(book.pickedUp(order.id, courier.id()));
    }
}

}  // namespace

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    const std::string restaurantNode = cfg.restaurants.front().node;
    const std::string clientNode = cfg.nodes.back().id;

    // --- las que no retiro vuelven a la cola, no se traspasan
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/brk_pool.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        fleet.push_back(std::make_unique<CourierActor>(1, cfg.startNode, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(2, cfg.startNode, 3, ctx));
        for (auto& actor : fleet) {
            actor->setFleet(&fleet);
        }

        for (OrderId id = 1; id <= 3; ++id) {
            darOrden(book, *fleet[0], makeOrder(id, restaurantNode, clientNode), false);
        }
        assert(fleet[0]->forceBreakdown());

        assert(fleet[0]->snapshot().state == CourierState::brokenDown);
        assert(fleet[0]->snapshot().bag.empty());
        assert(fleet[1]->snapshot().bag.size() == 3);  // TODAS al receptor
        assert(book.unassignedIds().empty());          // ninguna vuelve a la cola

        const OrderStats s = book.stats();
        assert(s.breakdowns == 1);
        assert(s.reassigned == 3);                     // una linea orderReassigned por orden
        assert(s.created == s.delivered + s.rejected + s.pending);
        std::cout << "  el traspaso mueve las 3 ordenes a un solo companero\n";
    }

    // --- las que lleva con comida encima SI se traspasan
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/brk_food.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        fleet.push_back(std::make_unique<CourierActor>(1, cfg.startNode, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(2, cfg.startNode, 3, ctx));
        for (auto& actor : fleet) {
            actor->setFleet(&fleet);
        }

        darOrden(book, *fleet[0], makeOrder(1, restaurantNode, clientNode), true);
        darOrden(book, *fleet[0], makeOrder(2, restaurantNode, clientNode), true);
        darOrden(book, *fleet[0], makeOrder(3, restaurantNode, clientNode), false);

        assert(fleet[0]->forceBreakdown());

        const Courier receptor = fleet[1]->snapshot();
        assert(receptor.bag.size() == 3);              // todas, con comida o sin ella
        assert(fleet[0]->snapshot().bag.empty());
        assert(book.unassignedIds().empty());

        for (OrderId id = 1; id <= 2; ++id) {
            const Order o = *book.find(id);
            assert(o.assignedCourier == 2);            // ahora la lleva el receptor
            assert(o.state == OrderState::pickedUp);   // y sigue retirada: no vuelve atras
            assert(o.reassigned);
        }
        // el receptor puede terminar la entrega que empezo otro
        assert(book.delivered(1, 2));
        std::cout << "  una orden con comida encima pasa al companero, que puede entregarla\n";
    }

    // --- todo-o-nada: si las que llevan comida no entran todas, la averia no ocurre
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/brk_allornothing.log", clock);
        OrderBook book(50, clock, log);
        Restaurants restaurants;
        for (const RestaurantConfig& r : cfg.restaurants) {
            restaurants.emplace(r.id,
                                std::make_unique<Restaurant>(r.id, r.pickupSlots));
        }
        const SimContext ctx{graph, clock, log, book, restaurants};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        fleet.push_back(std::make_unique<CourierActor>(1, cfg.startNode, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(2, cfg.startNode, 3, ctx));
        for (auto& actor : fleet) {
            actor->setFleet(&fleet);
        }

        // el 1 lleva 3; al receptor solo le entra 1
        for (OrderId id = 1; id <= 3; ++id) {
            darOrden(book, *fleet[0], makeOrder(id, restaurantNode, clientNode), true);
        }
        for (OrderId id = 10; id <= 11; ++id) {
            darOrden(book, *fleet[1], makeOrder(id, restaurantNode, clientNode), false);
        }

        assert(!fleet[0]->forceBreakdown());           // la averia no ocurre

        assert(fleet[0]->snapshot().state != CourierState::brokenDown);
        assert(fleet[0]->snapshot().bag.size() == 3);  // sigue con las suyas
        assert(fleet[1]->snapshot().bag.size() == 2);  // el receptor, intacto
        assert(book.stats().breakdowns == 0);
        assert(book.stats().reassigned == 0);
        std::cout << "  si no entran todas, no se mueve ninguna y la averia no ocurre\n";
    }

    // --- LA prueba: averias simultaneas cruzadas, muchas veces
    {
        const int kRounds = 300;
        const auto t0 = std::chrono::steady_clock::now();

        for (int round = 0; round < kRounds; ++round) {
            const SimClock clock(1.0);
            EventLog log("tests/bin/brk_cross.log", clock);
            OrderBook book(50, clock, log);
            Restaurants restaurants;
            for (const RestaurantConfig& r : cfg.restaurants) {
                restaurants.emplace(r.id,
                                    std::make_unique<Restaurant>(r.id, r.pickupSlots));
            }
            const SimContext ctx{graph, clock, log, book, restaurants};

            std::vector<std::unique_ptr<CourierActor>> fleet;
            fleet.push_back(std::make_unique<CourierActor>(1, cfg.startNode, 4, ctx));
            fleet.push_back(std::make_unique<CourierActor>(2, cfg.startNode, 4, ctx));
            for (auto& actor : fleet) {
                actor->setFleet(&fleet);
            }

            // los dos con comida encima: asi el traspaso es real y toma los dos candados
            OrderId id = 1;
            for (int k = 0; k < 2; ++k) {
                darOrden(book, *fleet[0], makeOrder(id, restaurantNode, clientNode), true);
                ++id;
            }
            for (int k = 0; k < 2; ++k) {
                darOrden(book, *fleet[1], makeOrder(id, restaurantNode, clientNode), true);
                ++id;
            }

            // Los dos se averian A LA VEZ, cada uno eligiendo al otro. Tomando los
            // candados uno por uno, cada hilo tendria el suyo esperando el del otro.
            std::thread a([&fleet]() { fleet[0]->forceBreakdown(); });
            std::thread b([&fleet]() { fleet[1]->forceBreakdown(); });
            a.join();
            b.join();

            // Pase lo que pase, ninguna orden se perdio ni quedo en dos bolsas.
            std::set<OrderId> vistas;
            for (const auto& actor : fleet) {
                for (const OrderId held : actor->snapshot().bag) {
                    assert(vistas.insert(held).second);
                }
            }
            const OrderStats s = book.stats();
            assert(s.created == 4);
            assert(s.created == s.delivered + s.rejected + s.pending);
        }

        std::cout << "  " << kRounds << " averias simultaneas cruzadas en "
                  << realElapsedMs(t0) << " ms, sin un solo bloqueo\n";
    }

    std::cout << "  breakdown ok\n";
}
