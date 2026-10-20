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

    // --- un id repetido se rechaza sin ensuciar los contadores
    //
    // emplace() no inserta si la clave ya existe, pero no falla ni avisa: sin la
    // guarda de admit(), la orden quedaba contada como creada sin estar en el libro y
    // el invariante dejaba de cerrar.
    {
        EventLog log("tests/bin/book_dup.log", clock);
        OrderBook book(10, clock, log);
        assert(book.admit(makeOrder(7)));
        assert(!book.admit(makeOrder(7)));          // mismo id: no entra

        const OrderStats s = book.stats();
        assert(s.created == 1);                     // el duplicado NO se conto
        assert(s.rejected == 0);                    // tampoco es un rechazo del §4
        assert(s.pending == 1);
        assert(s.created == s.delivered + s.rejected + s.pending);

        // y la orden original quedo intacta
        const auto original = book.find(7);
        assert(original.has_value());
        assert(original->state == OrderState::created);
        assert(book.unassignedIds().size() == 1);
        std::cout << "  un id repetido se rechaza sin romper los contadores\n";
    }

    // --- el ciclo completo de una orden y los numeros del resumen
    {
        EventLog log("tests/bin/book_cycle.log", clock);
        OrderBook book(10, clock, log);
        assert(book.admit(makeOrder(1)));

        assert(book.assign(1, 7));
        assert(!book.assign(1, 9));          // ya asignada: no se asigna de nuevo
        assert(book.find(1)->assignedCourier == 7);

        assert(!book.pickedUp(1, 9));        // otro courier no puede retirarla
        assert(book.pickedUp(1, 7));
        assert(!book.pickedUp(1, 7));        // ni dos veces

        assert(!book.delivered(1, 9));       // tampoco entregarla
        assert(book.delivered(1, 7));
        assert(!book.delivered(1, 7));       // ni entregarla dos veces

        const Order done = *book.find(1);
        assert(done.state == OrderState::delivered);
        assert(done.assignedMs >= done.createdMs);
        assert(done.pickedUpMs >= done.assignedMs);
        assert(done.deliveredMs >= done.pickedUpMs);

        const OrderStats s = book.stats();
        assert(s.delivered == 1);
        assert(s.pending == 0);
        assert(s.created == s.delivered + s.rejected + s.pending);
        assert(s.maxDeliveryMs == deliveryMs(done));
        assert(s.avgDeliveryMs == deliveryMs(done));
        std::cout << "  ciclo completo: asignada, retirada y entregada una sola vez\n";
    }

    // --- unassign: una orden asignada vuelve a la cola en vez de quedar colgada
    //
    // Sin esto, si la oferta al courier falla la orden se queda en assigned con un
    // courier que no la tiene: nadie la entrega y nadie la rechaza.
    {
        EventLog log("tests/bin/book_unassign.log", clock);
        OrderBook book(10, clock, log);
        assert(book.admit(makeOrder(1)));
        assert(book.assign(1, 5));

        // oferta rechazada: vuelve a la cola pero NO cuenta como reasignacion
        assert(book.unassign(1, false));
        const Order back = *book.find(1);
        assert(back.state == OrderState::created);
        assert(back.assignedCourier == kNoCourier);
        assert(!back.reassigned);
        assert(book.stats().reassigned == 0);
        assert(book.unassignedIds().size() == 1);   // vuelve a estar disponible

        // averia: tambien vuelve, pero esta si cuenta (§6)
        assert(book.assign(1, 6));
        assert(book.unassign(1, true));
        assert(book.find(1)->reassigned);
        assert(book.stats().reassigned == 1);

        // una orden que no esta asignada no se puede desasignar
        assert(book.assign(1, 7));
        assert(book.pickedUp(1, 7));
        assert(!book.unassign(1, true));            // ya la retiro: no vuelve atras
        assert(!book.unassign(999, true));          // no existe
        std::cout << "  unassign devuelve la orden a la cola sin perderla" << "\n";
    }

    // --- waitForUnassigned: espera dormido y admit() la despierta
    {
        EventLog log("tests/bin/book_wait.log", clock);
        OrderBook book(10, clock, log);

        // con una orden ya esperando, vuelve enseguida
        assert(book.admit(makeOrder(1)));
        auto t0 = std::chrono::steady_clock::now();
        assert(book.waitForUnassigned(std::chrono::seconds(5)));
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        assert(ms < 50);

        // sin ordenes pendientes, espera el plazo SIN consumir CPU
        assert(book.assign(1, 1));                  // ya no hay ninguna en created
        const double cpuBefore = 1000.0 * static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
        t0 = std::chrono::steady_clock::now();
        assert(book.waitForUnassigned(std::chrono::milliseconds(300)));
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - t0).count();
        const double cpuUsed =
            1000.0 * static_cast<double>(std::clock()) / CLOCKS_PER_SEC - cpuBefore;
        assert(ms >= 290);                          // espero de verdad
        std::cout << "  esperar 300 ms sin trabajo consumio " << cpuUsed
                  << " ms de CPU" << "\n";
        assert(cpuUsed < 50.0);                     // no gira

        // admit() desde otro hilo la despierta al instante
        std::thread producer([&book]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            book.admit(makeOrder(2));
        });
        t0 = std::chrono::steady_clock::now();
        assert(book.waitForUnassigned(std::chrono::seconds(10)));
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - t0).count();
        producer.join();
        std::cout << "  admit() la desperto en " << ms << " ms (el plazo era 10000)" << "\n";
        assert(ms < 1000);

        // stopWaiting() la despierta y devuelve false
        std::thread stopper([&book]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            book.stopWaiting();
        });
        assert(book.assign(2, 1));                  // que no haya trabajo pendiente
        assert(!book.waitForUnassigned(std::chrono::seconds(10)));
        stopper.join();
        std::cout << "  stopWaiting() la despierta y devuelve false" << "\n";
    }

    // --- al cerrar, las ordenes vivas pasan a pending y el invariante cierra (§7)
    {
        EventLog log("tests/bin/book_pending.log", clock);
        OrderBook book(50, clock, log);
        for (OrderId id = 1; id <= 5; ++id) {
            assert(book.admit(makeOrder(id)));
        }
        // una entregada, una rechazada, y tres vivas en distintas etapas
        assert(book.assign(1, 1) && book.pickedUp(1, 1) && book.delivered(1, 1));
        assert(book.reject(2, RejectReason::unreachable));
        assert(book.assign(3, 1));
        assert(book.assign(4, 1) && book.pickedUp(4, 1));
        // la 5 queda en created

        const std::int64_t marked = book.markRemainingAsPending();
        assert(marked == 3);                 // solo las vivas: la entregada y la
                                             // rechazada son finales y no se tocan
        const OrderStats s = book.stats();
        assert(s.created == 5);
        assert(s.delivered == 1);
        assert(s.rejected == 1);
        // pending cuenta las que TERMINARON en ese estado, no las vivas: al marcarlas,
        // pending es un estado final y el contador de vivas baja a cero. Si stats()
        // informara solo las vivas, el resumen diria siempre cero pendientes.
        assert(s.pending == 3);
        assert(s.created == s.delivered + s.rejected + s.pending);
        assert(book.find(1)->state == OrderState::delivered);
        assert(book.find(2)->state == OrderState::rejected);
        assert(book.find(3)->state == OrderState::pending);
        assert(book.find(5)->state == OrderState::pending);

        // llamarla dos veces no cambia nada: de pending no se sale
        assert(book.markRemainingAsPending() == 0);
        std::cout << "  al cerrar, 3 ordenes vivas pasaron a pending y 2 finales quedaron igual\n";
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
