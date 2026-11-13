#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>

// Suma de milisegundos simulados saturada a int64 (evita UB por desbordamiento).
inline std::int64_t saturatingAddMs(std::int64_t a, std::int64_t b) {
    if (b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) {
        return std::numeric_limits<std::int64_t>::max();
    }
    if (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return a + b;
}

class SimClock {
public:
    // Tope de una duracion real, en microsegundos (~31 anios): con steady_clock (ns en int64)
    // now() + duracion nunca desborda, ni hacia el futuro ni hacia el pasado.
    static constexpr double kMaxRealUs = 1e15;

    explicit SimClock(double timeScale)
        : start_(std::chrono::steady_clock::now()), timeScale_(timeScale) {
        assert(timeScale > 0.0);
    }

    std::int64_t nowMs() const {
        const std::chrono::duration<double, std::milli> real =
            std::chrono::steady_clock::now() - start_;
        const double sim = real.count() * timeScale_;
        if (sim >= 9.2e18) {
            return INT64_MAX;
        }
        return static_cast<std::int64_t>(sim);
    }

    // Conversion sim -> real. Se recorta en double ANTES de convertir a entero
    // (llround fuera de rango es indefinido) y por ambos lados; 1e15 us ~ 31 anios.
    std::chrono::microseconds realDuration(std::int64_t simMs) const {
        const double us = std::clamp(static_cast<double>(simMs) * 1000.0 / timeScale_,
                                     -kMaxRealUs, kMaxRealUs);
        return std::chrono::microseconds(std::llround(us));
    }

    void sleepSim(std::int64_t simMs) const {
        std::this_thread::sleep_for(realDuration(simMs));
    }

    // Instante real en que el reloj simulado llega a simMs. Un plazo ya vencido
    // devuelve "ahora"; uno muy lejano se recorta a 24 h DESDE AHORA (el llamador
    // despierta, re-evalua y vuelve a dormir; nunca queda en el pasado).
    std::chrono::steady_clock::time_point timePointAt(std::int64_t simMs) const {
        constexpr std::chrono::microseconds kMaxAhead = std::chrono::hours(24);
        const auto now = std::chrono::steady_clock::now();
        const std::chrono::microseconds elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(now - start_);
        const std::chrono::microseconds remaining = realDuration(simMs) - elapsed;
        return now + std::clamp(remaining, std::chrono::microseconds(0), kMaxAhead);
    }

    double timeScale() const { return timeScale_; }

private:
    const std::chrono::steady_clock::time_point start_;
    const double timeScale_;
};
