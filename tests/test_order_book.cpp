// Pruebas de OrderBook, el monitor central.
//
// La prueba que importa es la del limite bajo rafaga: 16 hilos intentando admitir a la
// vez contra un maxPending de 50. Si admit() consultara el contador y agregara en dos
// pasos, entre uno y otro se soltaria el candado y entrarian de mas. El numero exacto
// es la unica evidencia valida de que la verificacion y la accion son atomicas.
#include "OrderBook.h"

#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

#include "OrderFactory.h"

namespace {

Order makeOrder(OrderId id) {
    Order o;
    o.id = id;
    o.restaurantId = "r0";
    o.pickupNode = "n15";
    o.dropoffNode = "n420";
    return o;
}

}  // namespace

int main() {
    const SimClock clock(60.0);

    // --- admision normal y limite exacto
    {
        EventLog log("tests/bin/book_basic.log", clock);
        OrderBook book(3, clock, log);

        assert(book.admit(makeOrder(1)));
        assert(book.admit(makeOrder(2)));
        assert(book.admit(makeOrder(3)));
        assert(!book.admit(makeOrder(4)));      // la cuarta no entra

        const OrderStats s = book.stats();
        assert(s.created == 4);                 // la rechazada tambien se creo
        assert(s.rejected == 1);
        assert(s.pending == 3);
        assert(s.created == s.delivered + s.rejected + s.pending);
        std::cout << "  limite respetado: 3 admitidas, la 4a rechazada\n";

        // la rechazada quedo registrada con su motivo, no desaparecio
        const auto fourth = book.find(4);
        assert(fourth.has_value());
        assert(fourth->state == OrderState::rejected);
        assert(fourth->rejectReason == RejectReason::queueFull);
        assert(isFinalState(fourth->state));

        // createdMs viene de la t que devolvio el log: no son dos relojes distintos
        const auto first = book.find(1);
        assert(first.has_value());
        assert(first->createdMs >= 0);
        assert(first->state == OrderState::created);

        assert(!book.find(99).has_value());
        assert(book.unassignedIds().size() == 3);
        std::cout << "  find() devuelve copias y unassignedIds() solo las que esperan\n";
    }

    // --- una orden en estado final no se rechaza dos veces
    {
        EventLog log("tests/bin/book_final.log", clock);
        OrderBook book(10, clock, log);
        assert(book.admit(makeOrder(1)));
        assert(book.reject(1, RejectReason::unreachable));
        assert(!book.reject(1, RejectReason::noCourier));   // ya es final
        assert(!book.reject(777, RejectReason::noCourier)); // no existe

        const OrderStats s = book.stats();
        assert(s.rejected == 1);                            // no se conto dos veces
        assert(s.pending == 0);
        assert(s.created == s.delivered + s.rejected + s.pending);
        std::cout << "  una orden final no se puede rechazar de nuevo\n";
    }

    // --- LA prueba: 16 hilos admitiendo a la vez contra un limite de 50
    {
        EventLog log("tests/bin/book_burst.log", clock);
        const int kLimit = 50;
        const int kThreads = 16;
        const int kPerThread = 100;
        OrderBook book(kLimit, clock, log);

        std::vector<std::thread> threads;
        std::vector<int> admitted(kThreads, 0);
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&book, &admitted, t]() {
                for (int i = 0; i < kPerThread; ++i) {
                    const OrderId id = static_cast<OrderId>(t * kPerThread + i) + 1;
                    if (book.admit(makeOrder(id))) {
                        admitted[static_cast<std::size_t>(t)]++;
                    }
                }
            });
        }
        for (std::thread& th : threads) { th.join(); }

        int totalAdmitted = 0;
        for (const int n : admitted) { totalAdmitted += n; }

        const OrderStats s = book.stats();
        std::cout << "  " << kThreads * kPerThread << " intentos desde " << kThreads
                  << " hilos contra maxPending=" << kLimit << ": admitidas="
                  << totalAdmitted << " rechazadas=" << s.rejected << "\n";

        assert(totalAdmitted == kLimit);        // ni una de mas
        assert(s.pending == kLimit);
        assert(s.created == kThreads * kPerThread);
        assert(s.rejected == kThreads * kPerThread - kLimit);
        assert(s.created == s.delivered + s.rejected + s.pending);
        std::cout << "  ninguna orden perdida ni duplicada, limite exacto\n";
    }

    std::cout << "  order book ok\n";
}
