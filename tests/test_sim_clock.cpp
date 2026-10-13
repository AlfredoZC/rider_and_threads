// Pruebas de SimClock.
//
// Las pruebas de tiempo son delicadas: el sistema operativo puede demorar mas de lo
// pedido en despertar un hilo, nunca menos. Por eso los margenes son generosos hacia
// arriba y estrictos hacia abajo: si durmio MENOS de lo pedido, es un bug de verdad.
#include "SimClock.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

namespace {

std::int64_t realElapsedMs(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - from).count();
}

}  // namespace

int main() {
    // --- conversiones: no dependen de ninguna medicion, son exactas
    const SimClock fast(60.0);
    assert(fast.realDuration(60000).count() == 1000);   // 60 s simulados = 1 s real
    assert(fast.realDuration(600).count() == 10);
    assert(fast.realDuration(0).count() == 0);
    assert(fast.timeScale() == 60.0);

    const SimClock same(1.0);
    assert(same.realDuration(1500).count() == 1500);    // escala 1: son lo mismo

    const SimClock slow(0.5);                           // mas lento que el mundo real
    assert(slow.realDuration(1000).count() == 2000);

    // redondeo, no truncamiento: 599/60 = 9.98 -> 10, no 9
    assert(fast.realDuration(599).count() == 10);

    // --- el reloj arranca cerca de cero
    const SimClock clock(60.0);
    assert(clock.nowMs() >= 0);
    assert(clock.nowMs() < 1000);          // menos de 1 s simulado = 16 ms reales

    // --- monotonia: la t del log nunca puede decrecer (§6)
    std::int64_t last = clock.nowMs();
    for (int i = 0; i < 200000; ++i) {
        const std::int64_t now = clock.nowMs();
        assert(now >= last);
        last = now;
    }

    // --- el tiempo simulado avanza timeScale veces mas rapido que el real
    {
        const SimClock c(60.0);
        const auto t0 = std::chrono::steady_clock::now();
        const std::int64_t before = c.nowMs();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const std::int64_t after = c.nowMs();
        const std::int64_t realMs = realElapsedMs(t0);

        const std::int64_t simulated = after - before;
        std::cout << "  " << realMs << " ms reales -> " << simulated << " ms simulados\n";
        assert(simulated >= realMs * 55);        // no puede ir mas lento que la escala
        assert(simulated <= realMs * 65);        // ni mas rapido
    }

    // --- sleepSim duerme el tiempo real equivalente
    {
        const SimClock c(60.0);
        const auto t0 = std::chrono::steady_clock::now();
        c.sleepSim(6000);                        // 6 s simulados = 100 ms reales
        const std::int64_t realMs = realElapsedMs(t0);
        std::cout << "  sleepSim(6000) con escala 60 durmio " << realMs << " ms reales\n";
        assert(realMs >= 95);                    // nunca menos de lo pedido
        assert(realMs < 600);                    // margen amplio: el SO puede demorarse
    }

    // --- varios hilos leyendo a la vez: el reloj es compartible sin candado.
    // Cada hilo verifica su propia secuencia; si hubiera estado mutable compartido,
    // ThreadSanitizer lo reportaria al correr este mismo test con -fsanitize=thread.
    {
        const SimClock c(60.0);
        std::vector<std::thread> threads;
        std::vector<int> ok(4, 0);
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&c, &ok, t]() {
                std::int64_t prev = c.nowMs();
                for (int i = 0; i < 50000; ++i) {
                    const std::int64_t now = c.nowMs();
                    if (now < prev) { return; }   // no monotono: deja ok[t] en 0
                    prev = now;
                }
                ok[static_cast<std::size_t>(t)] = 1;
            });
        }
        for (std::thread& th : threads) { th.join(); }
        for (int v : ok) { assert(v == 1); }
        std::cout << "  4 hilos leyendo el reloj a la vez: monotono en todos\n";
    }

    std::cout << "  sim clock ok\n";
}
