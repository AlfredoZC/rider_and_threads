#include "accel/DeliveryPercentiles.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "accel/GpuWorker.h"
#include "accel/ThreadPool.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

using Keys = std::vector<std::uint32_t>;

std::uint64_t nextRandom(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t mixed = state;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
    return mixed ^ (mixed >> 31);
}

Keys sortedCopy(Keys keys) {
    std::sort(keys.begin(), keys.end());
    return keys;
}

}  // namespace

int main() {
    accel::ThreadPool pool(4);

    // Bits significativos: los del numero mas grande.
    CHECK(accel::significantBits(0) == 0);
    CHECK(accel::significantBits(1) == 1);
    CHECK(accel::significantBits(7) == 3);
    CHECK(accel::significantBits(8) == 4);
    CHECK(accel::significantBits(7200000) == 23);
    CHECK(accel::significantBits(0xFFFFFFFFu) == 32);

    // Rango mas cercano: posicion ceil(p/100 * N) - 1, en enteros y sin desbordar con N grande.
    CHECK(accel::nearestRankIndex(50, 20) == 9);
    CHECK(accel::nearestRankIndex(90, 20) == 17);
    CHECK(accel::nearestRankIndex(99, 20) == 19);
    CHECK(accel::nearestRankIndex(50, 1) == 0 && accel::nearestRankIndex(99, 1) == 0);
    CHECK(accel::nearestRankIndex(99, 100) == 98);
    CHECK(accel::nearestRankIndex(99, 100000000) == 98999999);
    CHECK(accel::nearestRankIndex(99, 3000000000ull) == 2969999999ull);

    // Ejemplo de 20 tiempos del material: p50 = 17, p90 = 31, p99 = 47.
    {
        const Keys sorted{9, 11, 12, 13, 14, 14, 15, 16, 16, 17, 18, 19, 20, 22, 23, 25, 28, 31, 35, 47};
        const accel::DeliveryPercentiles percentiles = accel::percentilesFromSorted(sorted);
        CHECK(percentiles.p50Ms == 17 && percentiles.p90Ms == 31 && percentiles.p99Ms == 47);
    }

    // Ejemplo a mano de 8 numeros de 3 bits (6 3 5 2 7 3 0 4), pasada por pasada.
    {
        const Keys input{6, 3, 5, 2, 7, 3, 0, 4};
        const Keys afterBit0 = accel::radixSortPassCpu(input, 0);
        CHECK((afterBit0 == Keys{6, 2, 0, 4, 3, 5, 7, 3}));
        const Keys afterBit1 = accel::radixSortPassCpu(afterBit0, 1);
        CHECK((afterBit1 == Keys{0, 4, 5, 6, 2, 3, 7, 3}));
        const Keys afterBit2 = accel::radixSortPassCpu(afterBit1, 2);
        CHECK((afterBit2 == Keys{0, 2, 3, 3, 4, 5, 6, 7}));
        CHECK(accel::radixSortCpu1(input) == afterBit2);
        CHECK(accel::radixSortCpuN(input, pool) == afterBit2);
    }

    // Estabilidad: tras la pasada del bit b, las claves con los mismos b+1 bits bajos conservan el orden
    // relativo que traian (si no, cada pasada desharia a las anteriores).
    {
        std::uint64_t state = 5;
        Keys keys(500);
        for (auto& key : keys) key = static_cast<std::uint32_t>(nextRandom(state) % 4096);
        for (int bit = 0; bit < 12; ++bit) {
            const Keys after = accel::radixSortPassCpu(keys, bit);
            const std::uint32_t mask = (2u << bit) - 1;  // los bit + 1 bits bajos
            for (std::size_t position = 1; position < after.size(); ++position) {
                CHECK((after[position - 1] & mask) <= (after[position] & mask));
            }
            for (std::uint32_t lowBits = 0; lowBits <= mask && lowBits < 4096; ++lowBits) {
                Keys before;
                Keys later;
                for (const std::uint32_t key : keys) if ((key & mask) == lowBits) before.push_back(key);
                for (const std::uint32_t key : after) if ((key & mask) == lowBits) later.push_back(key);
                CHECK(before == later);  // mismo orden relativo dentro de cada grupo
            }
            keys = after;
        }
    }

    // Vacio, un elemento, todos iguales, todos cero, extremos.
    {
        const Keys none;
        CHECK(accel::radixSortCpu1(none).empty());
        CHECK(accel::radixSortCpuN(none, pool).empty());

        const Keys one{123456};
        CHECK(accel::radixSortCpu1(one) == one);
        CHECK(accel::radixSortCpuN(one, pool) == one);
        const accel::DeliveryPercentiles single = accel::percentilesFromSorted(one);
        CHECK(single.p50Ms == 123456 && single.p90Ms == 123456 && single.p99Ms == 123456);

        const Keys same(1001, 900000);
        CHECK(accel::radixSortCpu1(same) == same);
        CHECK(accel::radixSortCpuN(same, pool) == same);

        const Keys zeros(77, 0);  // ningun bit que mirar: cero pasadas
        CHECK(accel::radixSortCpu1(zeros) == zeros);
        CHECK(accel::radixSortCpuN(zeros, pool) == zeros);

        const Keys extremes{0xFFFFFFFFu, 0, 0x80000000u, 1, 0xFFFFFFFEu, 0x7FFFFFFFu, 0xFFFFFFFFu};
        const Keys expected = sortedCopy(extremes);
        CHECK(accel::radixSortCpu1(extremes) == expected);
        CHECK(accel::radixSortCpuN(extremes, pool) == expected);
        CHECK(expected.back() == 0xFFFFFFFFu);
    }

    // Pseudoaleatorio: radix 1 hilo == radix N hilos == std::sort, y percentiles iguales.
    for (const std::size_t count : {2u, 3u, 5u, 1000u, 4099u, 100003u}) {
        std::uint64_t state = 31 + count;
        Keys keys(count);
        for (auto& key : keys) key = static_cast<std::uint32_t>(nextRandom(state) % 7200000);
        const Keys expected = sortedCopy(keys);
        CHECK(accel::radixSortCpu1(keys) == expected);
        for (const unsigned threads : {1u, 2u, 3u, 8u}) {
            accel::ThreadPool threadPool(threads);
            CHECK(accel::radixSortCpuN(keys, threadPool) == expected);
        }
        const accel::DeliveryPercentiles fromRadix = accel::percentilesFromSorted(accel::radixSortCpu1(keys));
        const accel::DeliveryPercentiles fromStdSort = accel::percentilesFromSorted(expected);
        CHECK(fromRadix.p50Ms == fromStdSort.p50Ms && fromRadix.p90Ms == fromStdSort.p90Ms &&
              fromRadix.p99Ms == fromStdSort.p99Ms);
    }

    // Texto para stderr, en minutos.
    {
        CHECK(accel::formatDeliveryPercentiles({17 * 60000u, 31 * 60000u, 47 * 60000u}, 20, "cpu") ==
              "[estadisticas] percentiles de entrega: p50 17.0 min, p90 31.0 min, p99 47.0 min (backend=cpu)");
        CHECK(accel::formatDeliveryPercentiles({0, 0, 0}, 0, "cpu") ==
              "[estadisticas] percentiles de entrega: sin entregas");
    }

    // Resumen completo con y sin worker (el build de pruebas no tiene CUDA: siempre respaldo CPU), y con
    // tiempos negativos o enormes recortados a los bordes.
    {
        const std::vector<std::int64_t> minutes{47, 9, 28, 17, 11, 12, 35, 13, 14, 14, 15, 16, 16, 18, 19, 20, 22, 23, 25, 31};
        std::vector<std::int64_t> deliveryTimesMs;
        for (const std::int64_t minute : minutes) deliveryTimesMs.push_back(minute * 60000);
        accel::GpuWorker brokenWorker([] { throw std::runtime_error("sin GPU"); });
        accel::GpuWorker healthyWorker([] {});
        for (accel::GpuWorker* worker : {static_cast<accel::GpuWorker*>(nullptr), &brokenWorker, &healthyWorker}) {
            const accel::DeliveryPercentilesSummary summary = accel::deliveryPercentiles(deliveryTimesMs, worker);
            CHECK(summary.count == 20);
            CHECK(summary.percentiles.p50Ms == 17 * 60000u);
            CHECK(summary.percentiles.p90Ms == 31 * 60000u);
            CHECK(summary.percentiles.p99Ms == 47 * 60000u);
            CHECK(!summary.usedGpu);
        }
        brokenWorker.stop();
        healthyWorker.stop();

        const accel::DeliveryPercentilesSummary empty = accel::deliveryPercentiles({}, nullptr);
        CHECK(empty.count == 0);
        const accel::DeliveryPercentilesSummary clamped =
            accel::deliveryPercentiles({-5, 100, 9000000000000LL}, nullptr);
        CHECK(clamped.percentiles.p50Ms == 100);
        CHECK(clamped.percentiles.p99Ms == 0xFFFFFFFFu);
    }

    std::cout << "test_delivery_percentiles: OK\n";
    return 0;
}
