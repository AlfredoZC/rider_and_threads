#include "accel/DeliveryPercentiles.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "accel/OpenOrders.h"  // stripeOfChunk

namespace accel {

namespace {

bool bitIsZero(std::uint32_t key, int bit) { return ((key >> bit) & 1u) == 0; }

std::uint32_t maxKeyOf(const std::vector<std::uint32_t>& keys) {
    return keys.empty() ? 0u : *std::max_element(keys.begin(), keys.end());
}

std::uint32_t clampToUint32(std::int64_t deliveryTimeMs) {
    return static_cast<std::uint32_t>(std::clamp<std::int64_t>(deliveryTimeMs, 0, std::numeric_limits<std::uint32_t>::max()));
}

}  // namespace

int significantBits(std::uint32_t maxKey) {
    int bits = 0;
    while (maxKey > 0) {
        ++bits;
        maxKey >>= 1;
    }
    return bits;
}

std::size_t nearestRankIndex(unsigned percent, std::size_t count) {
    return (static_cast<std::size_t>(percent) * count + 99) / 100 - 1;
}

DeliveryPercentiles percentilesFromSorted(const std::vector<std::uint32_t>& sortedMs) {
    DeliveryPercentiles percentiles;
    percentiles.p50Ms = sortedMs[nearestRankIndex(50, sortedMs.size())];
    percentiles.p90Ms = sortedMs[nearestRankIndex(90, sortedMs.size())];
    percentiles.p99Ms = sortedMs[nearestRankIndex(99, sortedMs.size())];
    return percentiles;
}

std::vector<std::uint32_t> radixSortPassCpu(const std::vector<std::uint32_t>& keys, int bit) {
    std::size_t totalZeros = 0;  // ahi empiezan los unos
    for (const std::uint32_t key : keys) {
        totalZeros += bitIsZero(key, bit) ? 1 : 0;
    }
    std::vector<std::uint32_t> rearranged(keys.size());
    std::size_t nextZero = 0;
    std::size_t nextOne = totalZeros;
    for (const std::uint32_t key : keys) {
        if (bitIsZero(key, bit)) {
            rearranged[nextZero++] = key;
        } else {
            rearranged[nextOne++] = key;
        }
    }
    return rearranged;
}

std::vector<std::uint32_t> radixSortCpu1(std::vector<std::uint32_t> keys) {
    const int bitCount = significantBits(maxKeyOf(keys));
    for (int bit = 0; bit < bitCount; ++bit) {
        keys = radixSortPassCpu(keys, bit);  // la salida de esta pasada es la entrada de la siguiente
    }
    return keys;
}

std::vector<std::uint32_t> radixSortCpuN(std::vector<std::uint32_t> keys, ThreadPool& pool) {
    const std::size_t keyCount = keys.size();
    const std::size_t stripeCount = pool.threads();
    std::vector<std::uint32_t> rearranged(keyCount);
    std::vector<std::size_t> stripeZeros(stripeCount, 0);
    std::vector<std::size_t> zerosBeforeStripe(stripeCount, 0);
    const int bitCount = significantBits(maxKeyOf(keys));
    for (int bit = 0; bit < bitCount; ++bit) {
        const std::uint32_t* source = keys.data();
        std::uint32_t* destination = rearranged.data();
        std::size_t* stripeZerosData = stripeZeros.data();

        // Fase 1 (paralela): los ceros de cada franja; cada hilo escribe SOLO su casilla.
        pool.parallelFor(keyCount, [=](std::size_t chunkBegin, std::size_t chunkEnd) {
            std::size_t zeros = 0;
            for (std::size_t position = chunkBegin; position < chunkEnd; ++position) {
                zeros += bitIsZero(source[position], bit) ? 1 : 0;
            }
            stripeZerosData[stripeOfChunk(keyCount, stripeCount, chunkBegin)] = zeros;
        });

        // Fase 2 (1 hilo, stripeCount numeros): scan exclusivo de los ceros por franja.
        std::size_t totalZeros = 0;
        for (std::size_t stripe = 0; stripe < stripeCount; ++stripe) {
            zerosBeforeStripe[stripe] = totalZeros;
            totalZeros += stripeZeros[stripe];
        }

        // Fase 3 (paralela): las mismas franjas que en la fase 1; cada una escribe solo en posiciones propias.
        const std::size_t* zerosBeforeData = zerosBeforeStripe.data();
        pool.parallelFor(keyCount, [=](std::size_t chunkBegin, std::size_t chunkEnd) {
            const std::size_t stripe = stripeOfChunk(keyCount, stripeCount, chunkBegin);
            std::size_t nextZero = zerosBeforeData[stripe];
            std::size_t nextOne = totalZeros + (chunkBegin - zerosBeforeData[stripe]);  // unos antes de la franja
            for (std::size_t position = chunkBegin; position < chunkEnd; ++position) {
                if (bitIsZero(source[position], bit)) {
                    destination[nextZero++] = source[position];
                } else {
                    destination[nextOne++] = source[position];
                }
            }
        });
        keys.swap(rearranged);
    }
    return keys;
}

std::string radixSortGpu(const std::vector<std::uint32_t>& keys, std::vector<std::uint32_t>& sortedKeys) {
    DeviceRadixSort device;
    std::string errorMessage = device.allocate(keys.size());
    if (errorMessage.empty()) errorMessage = device.upload(keys.data(), keys.size(), nullptr);
    if (errorMessage.empty()) errorMessage = device.run(nullptr);
    std::vector<std::uint32_t> result(keys.size());
    if (errorMessage.empty()) errorMessage = device.download(result.data(), nullptr);
    if (errorMessage.empty()) {
        sortedKeys = std::move(result);
    }
    return errorMessage;
}

DeliveryPercentilesSummary deliveryPercentiles(const std::vector<std::int64_t>& deliveryTimesMs, GpuWorker* worker,
                                               std::chrono::milliseconds timeout) {
    DeliveryPercentilesSummary summary;
    summary.count = deliveryTimesMs.size();
    if (deliveryTimesMs.empty()) {
        return summary;
    }
    // Compartido con la tarea: puede seguir viva despues de que esta funcion vuelva (plazo vencido).
    auto keys = std::make_shared<std::vector<std::uint32_t>>(deliveryTimesMs.size());
    std::transform(deliveryTimesMs.begin(), deliveryTimesMs.end(), keys->begin(), clampToUint32);

    if (worker != nullptr && worker->healthy()) {
        try {
            const auto deadline = deadlineAfterMs(timeout.count());
            std::future<std::vector<std::uint32_t>> future = worker->submit(
                [keys] {
                    std::vector<std::uint32_t> sortedKeys;
                    const std::string errorMessage = radixSortGpu(*keys, sortedKeys);
                    if (!errorMessage.empty()) {
                        throw std::runtime_error(errorMessage);
                    }
                    return sortedKeys;
                },
                deadline);
            if (waitUntilDeadline(future, deadline)) {
                summary.percentiles = percentilesFromSorted(future.get());
                summary.usedGpu = true;
                return summary;
            }
        } catch (const std::exception&) {
            // GPU no disponible, tarea vencida o fallida: respaldo CPU.
        }
    }
    summary.percentiles = percentilesFromSorted(radixSortCpu1(*keys));
    return summary;
}

std::string formatDeliveryPercentiles(const DeliveryPercentiles& percentiles, std::size_t count,
                                      const std::string& backendName) {
    const std::string prefix = "[estadisticas] percentiles de entrega: ";
    if (count == 0) {
        return prefix + "sin entregas";
    }
    char text[160];
    std::snprintf(text, sizeof text, "p50 %.1f min, p90 %.1f min, p99 %.1f min", percentiles.p50Ms / 60000.0,
                  percentiles.p90Ms / 60000.0, percentiles.p99Ms / 60000.0);
    return prefix + text + " (backend=" + backendName + ")";
}

}  // namespace accel
