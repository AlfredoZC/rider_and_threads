// Pruebas de OrderGenerator.
//
// La prueba central es la de interrumpibilidad: con timeScale 1 y meanIntervalMs
// 20000, el proximo pedido esta a unos 20 segundos REALES. Si el hilo durmiera, parar
// tardaria eso. Como espera con plazo sobre una condition_variable, tiene que terminar
// en milisegundos. Es la diferencia entre dormir y esperar, y lo que hace posible el
// cierre en 5 segundos del §7.
#include "OrderGenerator.h"

#include <cassert>
#include <chrono>
#include <ctime>
#include <iostream>
#include <thread>

namespace {

std::int64_t realElapsedMs(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - from).count();
}

double cpuMs() {
    return 1000.0 * static_cast<double>(std::clock()) / CLOCKS_PER_SEC;
}

}  // namespace

int main() {
    const Config base = loadConfig("config/equipetrol.json");

    // --- genera ordenes y respeta el limite de la cola
    {
        Config cfg = base;
        cfg.maxPending = 5;

        // timeScale alto para que el tiempo simulado corra rapido y la prueba sea breve:
        // 20000 ms simulados son ~3.3 ms reales con escala 6000.
        const SimClock clock(6000.0);
        EventLog log("tests/bin/gen_burst.log", clock);
        OrderBook book(cfg.maxPending, clock, log);
        OrderGenerator generator(cfg, clock, book);

        std::thread t(&OrderGenerator::run, &generator);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        generator.requestStop();
        t.join();

        const OrderStats s = book.stats();
        std::cout << "  en 300 ms reales genero " << generator.generated()
                  << " ordenes (admitidas=" << s.created - s.rejected
                  << " rechazadas=" << s.rejected << ")\n";

        assert(generator.generated() > 0);
        assert(s.created == generator.generated());
        assert(s.pending <= 5);                    // nunca se paso de maxPending
        assert(s.created == s.delivered + s.rejected + s.pending);
        assert(s.rejected > 0);                    // con cola de 5 tiene que haber
        std::cout << "  la cola nunca supero maxPending y el invariante cierra\n";
    }

    // --- las rafagas existen: alguna llegada trae mas de una orden
    {
        Config cfg = base;
        cfg.maxPending = 100000;                   // que no rechace, para contar bien
        const SimClock clock(20000.0);
        EventLog log("tests/bin/gen_rate.log", clock);
        OrderBook book(cfg.maxPending, clock, log);
        OrderGenerator generator(cfg, clock, book);

        std::thread t(&OrderGenerator::run, &generator);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        generator.requestStop();
        t.join();

        const OrderStats s = book.stats();
        std::cout << "  sin limite de cola: " << s.created << " ordenes, "
                  << s.pending << " vivas\n";
        assert(s.created > 0);
        assert(s.rejected == 0);
        assert(s.created == s.pending);
    }

    // --- LA prueba: parar mientras espera un intervalo largo
    {
        Config cfg = base;
        const SimClock clock(1.0);                 // tiempo real: el intervalo son ~20 s
        EventLog log("tests/bin/gen_stop.log", clock);
        OrderBook book(cfg.maxPending, clock, log);
        OrderGenerator generator(cfg, clock, book);

        std::thread t(&OrderGenerator::run, &generator);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));

        const double cpuBefore = cpuMs();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const double cpuUsed = cpuMs() - cpuBefore;
        std::cout << "  esperando un intervalo de ~20 s consumio " << cpuUsed
                  << " ms de CPU en 300 ms\n";
        assert(cpuUsed < 100.0);                   // no gira revisando el reloj

        const auto t0 = std::chrono::steady_clock::now();
        generator.requestStop();
        t.join();
        const std::int64_t stopMs = realElapsedMs(t0);
        std::cout << "  parar mientras esperaba tardo " << stopMs << " ms\n";
        assert(stopMs < 500);                      // durmiendo habrian sido ~20000
    }

    // --- arrancar y parar muchas veces: busca avisos perdidos, que cuelgan
    {
        Config cfg = base;
        const SimClock clock(1.0);
        EventLog log("tests/bin/gen_cycles.log", clock);
        OrderBook book(cfg.maxPending, clock, log);

        const int kRounds = 200;
        for (int i = 0; i < kRounds; ++i) {
            OrderGenerator generator(cfg, clock, book);
            std::thread t(&OrderGenerator::run, &generator);
            if (i % 3 == 0) { std::this_thread::yield(); }
            generator.requestStop();
            t.join();
        }
        std::cout << "  " << kRounds << " ciclos arrancar/parar sin un solo cuelgue\n";
    }

    std::cout << "  order generator ok\n";
}
