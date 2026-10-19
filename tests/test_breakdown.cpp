// Pruebas de las averias y el traspaso todo-o-nada (§4, DP.6).
//
// Tres cosas que el enunciado exige y que solo se pueden comprobar aca:
//
//   1. El traspaso es TODO-O-NADA: o se mueven todas las ordenes del courier averiado,
//      o no se mueve ninguna. Nunca la mitad.
//   2. Las averias SIMULTANEAS no bloquean el sistema. Es el interbloqueo de manual: si
//      A se averia y elige a B mientras B se averia y elige a A, tomar los candados uno
//      por uno dejaria a cada uno esperando el del otro para siempre. La prueba lo
//      provoca a proposito, muchas veces: un interbloqueo no falla, CUELGA.
//   3. Las ordenes no se pierden ni se duplican: la bolsa del averiado queda vacia, la
//      del receptor tiene exactamente las que habia, y el libro sabe quien las lleva.
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

}  // namespace

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    const std::string restaurantNode = cfg.restaurants.front().node;
    const std::string clientNode = cfg.nodes.back().id;

    // --- el traspaso mueve TODAS las ordenes y deja el libro coherente
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/brk_basic.log", clock);
        OrderBook book(50, clock, log);
        const SimContext ctx{graph, clock, log, book};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        fleet.push_back(std::make_unique<CourierActor>(1, cfg.startNode, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(2, cfg.startNode, 3, ctx));
        for (auto& actor : fleet) {
            actor->setFleet(&fleet);
        }

        // tres ordenes en la bolsa del courier 1
        for (OrderId id = 1; id <= 3; ++id) {
            assert(book.admit(makeOrder(id, restaurantNode, clientNode)));
            assert(book.assign(id, 1));
            assert(fleet[0]->offer(*book.find(id)));
        }
        assert(fleet[0]->snapshot().bag.size() == 3);

        assert(fleet[0]->forceBreakdown());

        const Courier roto = fleet[0]->snapshot();
        const Courier receptor = fleet[1]->snapshot();
        assert(roto.state == CourierState::brokenDown);
        assert(roto.bag.empty());                    // no quedo ninguna a medias
        assert(receptor.bag.size() == 3);            // y estan TODAS en el receptor

        // el libro sabe que ahora las lleva el 2
        for (OrderId id = 1; id <= 3; ++id) {
            const Order o = *book.find(id);
            assert(o.assignedCourier == 2);
            assert(o.reassigned);
        }
        const OrderStats s = book.stats();
        assert(s.breakdowns == 1);
        assert(s.reassigned == 3);
        assert(s.created == s.delivered + s.rejected + s.pending);
        std::cout << "  traspaso completo: 3 ordenes pasaron del courier 1 al 2\n";
    }

    // --- todo-o-nada: si no entran todas, no se mueve ninguna
    {
        const SimClock clock(1.0);
        EventLog log("tests/bin/brk_allornothing.log", clock);
        OrderBook book(50, clock, log);
        const SimContext ctx{graph, clock, log, book};

        std::vector<std::unique_ptr<CourierActor>> fleet;
        fleet.push_back(std::make_unique<CourierActor>(1, cfg.startNode, 3, ctx));
        fleet.push_back(std::make_unique<CourierActor>(2, cfg.startNode, 3, ctx));
        for (auto& actor : fleet) {
            actor->setFleet(&fleet);
        }

        // el 1 lleva 3 y el receptor ya tiene 2: solo le entra 1, asi que no entra nada
        for (OrderId id = 1; id <= 3; ++id) {
            assert(book.admit(makeOrder(id, restaurantNode, clientNode)));
            assert(book.assign(id, 1));
            assert(fleet[0]->offer(*book.find(id)));
        }
        for (OrderId id = 10; id <= 11; ++id) {
            assert(book.admit(makeOrder(id, restaurantNode, clientNode)));
            assert(book.assign(id, 2));
            assert(fleet[1]->offer(*book.find(id)));
        }

        assert(!fleet[0]->forceBreakdown());         // la averia no ocurre

        const Courier intacto = fleet[0]->snapshot();
        assert(intacto.state != CourierState::brokenDown);
        assert(intacto.bag.size() == 3);             // sigue con las suyas
        assert(fleet[1]->snapshot().bag.size() == 2);
        assert(book.stats().breakdowns == 0);
        assert(book.stats().reassigned == 0);
        std::cout << "  si no entran todas no se mueve ninguna, y la averia no ocurre\n";
    }

    // --- LA prueba: averias simultaneas cruzadas, muchas veces
    {
        const int kRounds = 300;
        const auto t0 = std::chrono::steady_clock::now();

        for (int round = 0; round < kRounds; ++round) {
            const SimClock clock(1.0);
            EventLog log("tests/bin/brk_cross.log", clock);
            OrderBook book(50, clock, log);
            const SimContext ctx{graph, clock, log, book};

            std::vector<std::unique_ptr<CourierActor>> fleet;
            fleet.push_back(std::make_unique<CourierActor>(1, cfg.startNode, 4, ctx));
            fleet.push_back(std::make_unique<CourierActor>(2, cfg.startNode, 4, ctx));
            for (auto& actor : fleet) {
                actor->setFleet(&fleet);
            }

            OrderId id = 1;
            for (int k = 0; k < 2; ++k) {
                assert(book.admit(makeOrder(id, restaurantNode, clientNode)));
                assert(book.assign(id, 1));
                assert(fleet[0]->offer(*book.find(id)));
                ++id;
            }
            for (int k = 0; k < 2; ++k) {
                assert(book.admit(makeOrder(id, restaurantNode, clientNode)));
                assert(book.assign(id, 2));
                assert(fleet[1]->offer(*book.find(id)));
                ++id;
            }

            // Los dos se averian A LA VEZ, cada uno eligiendo al otro. Tomando los
            // candados uno por uno, cada hilo tendria el suyo esperando el del otro.
            std::thread a([&fleet]() { fleet[0]->forceBreakdown(); });
            std::thread b([&fleet]() { fleet[1]->forceBreakdown(); });
            a.join();
            b.join();

            // Pase lo que pase, ninguna orden se perdio ni se duplico.
            std::set<OrderId> vistas;
            for (const auto& actor : fleet) {
                for (const OrderId held : actor->snapshot().bag) {
                    assert(vistas.insert(held).second);   // no esta en dos bolsas
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
