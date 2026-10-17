// Pruebas de CourierActor: el ciclo de vida del hilo.
//
// Tres cosas que el brief exige y que aca se comprueban con numeros:
//   1. un courier sin trabajo NO consume CPU (§4),
//   2. todos los hilos terminan, ninguno queda abandonado (§7),
//   3. terminan rapido: el §7 da 5 segundos para cerrar todo.
//
// La prueba del aviso perdido es la repeticion: un lost wakeup no falla, CUELGA. Si
// esta prueba no termina, ese es el sintoma.
#include "CourierActor.h"

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

// Tiempo de CPU consumido por el proceso entero, en milisegundos.
double cpuMs() {
    return 1000.0 * static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}

}  // namespace

int main() {
    // --- estado inicial coherente, visto desde afuera
    {
        CourierActor actor(1, "n311", 3);
        const Courier view = actor.snapshot();
        assert(actor.id() == 1);
        assert(view.id == 1);
        assert(view.state == CourierState::idle);
        assert(currentNode(view) == "n311");
        assert(view.bag.empty());
        assert(hasRoom(view));
        std::cout << "  snapshot() devuelve el estado inicial correcto\n";
    }

    // --- 8 hilos esperando: no pueden consumir CPU (§4)
    {
        const int kCouriers = 8;
        std::vector<std::unique_ptr<CourierActor>> fleet;
        for (int i = 0; i < kCouriers; ++i) {
            fleet.push_back(std::make_unique<CourierActor>(
                static_cast<CourierId>(i) + 1, "n311", 3));
        }

        std::vector<std::thread> threads;
        for (auto& actor : fleet) {
            threads.emplace_back(&CourierActor::run, actor.get());
        }

        const double cpuBefore = cpuMs();
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        const double cpuUsed = cpuMs() - cpuBefore;

        // 8 hilos girando 400 ms gastarian unos 3200 ms de CPU. Esperando, casi cero.
        std::cout << "  8 couriers ociosos 400 ms consumieron " << cpuUsed << " ms de CPU\n";
        assert(cpuUsed < 100.0);

        // mientras esperan, el candado esta libre: snapshot() no se bloquea
        for (auto& actor : fleet) {
            const Courier view = actor->snapshot();
            assert(view.state == CourierState::idle);
        }
        std::cout << "  snapshot() funciona mientras los hilos esperan\n";

        // --- terminan todos, y rapido (§7: 5 segundos para cerrar)
        const auto t0 = std::chrono::steady_clock::now();
        for (auto& actor : fleet) { actor->requestStop(); }
        for (std::thread& t : threads) { t.join(); }
        const std::int64_t stopMs = realElapsedMs(t0);
        std::cout << "  los 8 hilos terminaron en " << stopMs << " ms\n";
        assert(stopMs < 1000);
    }

    // --- requestStop() antes de que el hilo arranque: el hilo no debe quedar colgado.
    // Es el caso del aviso perdido visto al reves, y lo cubre el predicado de wait().
    {
        CourierActor actor(1, "n311", 3);
        actor.requestStop();                       // la bandera ya esta en true
        std::thread t(&CourierActor::run, &actor);
        t.join();                                  // si colgara, la prueba no termina
        std::cout << "  requestStop() antes de arrancar el hilo no lo deja colgado\n";
    }

    // --- repeticion: el aviso perdido es una carrera, asi que hay que provocarla
    // muchas veces con temporizaciones distintas
    {
        const int kRounds = 300;
        for (int i = 0; i < kRounds; ++i) {
            CourierActor actor(1, "n311", 3);
            std::thread t(&CourierActor::run, &actor);
            if (i % 3 == 0) {
                std::this_thread::yield();         // deja que el hilo llegue al wait
            }
            actor.requestStop();
            actor.requestStop();                   // llamarla dos veces es inofensivo
            t.join();
        }
        std::cout << "  " << kRounds << " ciclos arrancar/parar sin un solo cuelgue\n";
    }

    std::cout << "  courier actor ok\n";
}
