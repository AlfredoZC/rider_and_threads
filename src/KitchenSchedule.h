#pragma once

#include <cstdint>
#include <limits>

// Calendario de pausas de un restaurante: funcion pura del tiempo simulado.
// El restaurante `index` de `count` esta en pausa en
// [k*P + offset, k*P + offset + D), k = 0, 1, 2..., con offset = (index*P)/count (division entera).
// Toda la aritmetica es sin desborde: offset se calcula como index*(P/count) + (index*(P%count))/count
// (identico a (index*P)/count) y las sumas saturan en kNever. Un limite mayor que int64 es
// "nunca": simplemente no ocurre durante la corrida.
struct KitchenSchedule {
    static constexpr std::int64_t kNever = std::numeric_limits<std::int64_t>::max();

    std::int64_t pauseEveryMs = 0;
    std::int64_t pauseMs = 0;
    std::int64_t index = 0;
    std::int64_t count = 1;

    bool enabled() const { return pauseEveryMs > 0 && pauseMs > 0; }

    // 0 <= index < count: index*q < P y index*r < count^2, nada desborda.
    std::int64_t offset() const {
        const std::int64_t q = pauseEveryMs / count;
        const std::int64_t r = pauseEveryMs % count;
        return index * q + (index * r) / count;
    }

    // Inicio de la k-esima pausa (k >= 0), o kNever si no es representable.
    std::int64_t startOf(std::int64_t k) const {
        const std::int64_t off = offset();
        if (k < 0 || (k > 0 && pauseEveryMs > (kNever - off) / k)) {
            return kNever;
        }
        return k * pauseEveryMs + off;
    }

    // Fin de la k-esima pausa, o kNever si no es representable.
    std::int64_t endOf(std::int64_t k) const {
        const std::int64_t start = startOf(k);
        if (start == kNever || start > kNever - pauseMs) {
            return kNever;
        }
        return start + pauseMs;
    }

    bool pausedAt(std::int64_t t) const {
        if (!enabled() || t < 0 || t < offset()) {
            return false;
        }
        return (t - offset()) % pauseEveryMs < pauseMs;
    }
};
