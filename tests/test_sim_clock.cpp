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

}

int main() {
    const SimClock fast(60.0);
    assert(fast.realDuration(60000).count() == 1000000);
    assert(fast.realDuration(600).count() == 10000);
    assert(fast.realDuration(0).count() == 0);
    assert(fast.timeScale() == 60.0);

    const SimClock same(1.0);
    assert(same.realDuration(1500).count() == 1500000);

    const SimClock slow(0.5);
    assert(slow.realDuration(1000).count() == 2000000);

    assert(fast.realDuration(599).count() == 9983);

    for (std::int64_t simMs = 1; simMs <= 100; ++simMs) {
        assert(fast.realDuration(simMs).count() > 0);
    }
    const SimClock veryFast(1000.0);
    assert(veryFast.realDuration(1).count() > 0);

    const SimClock clock(60.0);
    assert(clock.nowMs() >= 0);
    assert(clock.nowMs() < 1000);

    std::int64_t last = clock.nowMs();
    for (int i = 0; i < 200000; ++i) {
        const std::int64_t now = clock.nowMs();
        assert(now >= last);
        last = now;
    }

    {
        const SimClock c(60.0);
        const auto t0 = std::chrono::steady_clock::now();
        const std::int64_t before = c.nowMs();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const std::int64_t after = c.nowMs();
        const std::int64_t realMs = realElapsedMs(t0);

        const std::int64_t simulated = after - before;
        std::cout << "  " << realMs << " ms reales -> " << simulated << " ms simulados\n";
        assert(simulated >= realMs * 55);
        assert(simulated <= realMs * 65);
    }

    {
        const SimClock c(60.0);
        const auto t0 = std::chrono::steady_clock::now();
        c.sleepSim(6000);
        const std::int64_t realMs = realElapsedMs(t0);
        std::cout << "  sleepSim(6000) con escala 60 durmio " << realMs << " ms reales\n";
        assert(realMs >= 95);
        assert(realMs < 600);
    }

    {
        const SimClock c(60.0);
        std::vector<std::thread> threads;
        std::vector<int> ok(4, 0);
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back([&c, &ok, t]() {
                std::int64_t prev = c.nowMs();
                for (int i = 0; i < 50000; ++i) {
                    const std::int64_t now = c.nowMs();
                    if (now < prev) { return; }
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
