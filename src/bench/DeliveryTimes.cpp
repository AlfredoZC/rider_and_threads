#include "bench/DeliveryTimes.h"

#include <cmath>

#include "accel/DeliveryHistogram.h"

namespace bench {

namespace {

// splitmix64: uniforme en (0, 1), determinista segun el estado.
double nextUniform(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t mixed = state;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
    mixed ^= mixed >> 31;
    return (static_cast<double>(mixed >> 11) + 0.5) / 9007199254740992.0;
}
}  // namespace

void fillDeliveryTimes(std::vector<std::int32_t>& deliveryTimesMs, std::uint32_t seed) {
    constexpr double kMedianMinutes = 28.0;
    constexpr double kSigma = 0.45;  // ~95% entre 11 y 68 min; cola larga
    constexpr double kPi = 3.14159265358979323846;
    std::uint64_t state = seed;
    for (std::size_t deliveryIndex = 0; deliveryIndex < deliveryTimesMs.size(); ++deliveryIndex) {
        if (deliveryIndex % 1024 == 1023) {
            // Borde exacto de minuto: 0, 1, 2, ... 70 minutos (los >= 60 caen en el desborde).
            deliveryTimesMs[deliveryIndex] = static_cast<std::int32_t>((deliveryIndex / 1024) % 71) * accel::kMillisecondsPerMinute;
            continue;
        }
        const double firstUniform = nextUniform(state);
        const double secondUniform = nextUniform(state);
        const double normal = std::sqrt(-2.0 * std::log(firstUniform)) * std::cos(2.0 * kPi * secondUniform);
        const double minutes = kMedianMinutes * std::exp(kSigma * normal);
        deliveryTimesMs[deliveryIndex] = static_cast<std::int32_t>(minutes * accel::kMillisecondsPerMinute);
    }
}

}  // namespace bench
