#include "accel/DeliveryHistogram.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <utility>

namespace accel {

std::vector<unsigned> deliveryTimeHistogramCpu1(const std::vector<std::int32_t>& deliveryTimesMs) {
    std::vector<unsigned> bins(kDeliveryHistogramBins, 0);
    for (const std::int32_t deliveryTimeMs : deliveryTimesMs) {
        ++bins[static_cast<std::size_t>(deliveryMinuteBin(deliveryTimeMs))];
    }
    return bins;
}

std::vector<unsigned> deliveryTimeHistogramCpuN(const std::vector<std::int32_t>& deliveryTimesMs, ThreadPool& pool) {
    std::vector<unsigned> totalBins(kDeliveryHistogramBins, 0);
    std::mutex totalBinsMutex;
    const std::int32_t* times = deliveryTimesMs.data();
    pool.parallelFor(deliveryTimesMs.size(), [&totalBins, &totalBinsMutex, times](std::size_t chunkBegin, std::size_t chunkEnd) {
        std::vector<unsigned> chunkBins(kDeliveryHistogramBins, 0);  // copia privada del trozo
        for (std::size_t deliveryIndex = chunkBegin; deliveryIndex < chunkEnd; ++deliveryIndex) {
            ++chunkBins[static_cast<std::size_t>(deliveryMinuteBin(times[deliveryIndex]))];
        }
        const std::lock_guard<std::mutex> guard(totalBinsMutex);  // fusion: una sola vez por trozo
        for (std::size_t binIndex = 0; binIndex < chunkBins.size(); ++binIndex) {
            totalBins[binIndex] += chunkBins[binIndex];
        }
    });
    return totalBins;  // parallelFor ya espero a todos los trozos
}

std::string deliveryTimeHistogramGpu(const std::vector<std::int32_t>& deliveryTimesMs,
                                     std::vector<unsigned>& bins, HistogramKernel kernel) {
    DeviceHistogram device;
    std::string errorMessage = device.allocate(deliveryTimesMs.size());
    if (errorMessage.empty()) errorMessage = device.upload(deliveryTimesMs.data(), deliveryTimesMs.size(), nullptr);
    if (errorMessage.empty()) errorMessage = device.run(kernel, nullptr);
    std::vector<unsigned> resultBins(kDeliveryHistogramBins, 0);
    if (errorMessage.empty()) errorMessage = device.download(resultBins.data(), nullptr);
    if (errorMessage.empty()) {
        bins = std::move(resultBins);
    }
    return errorMessage;
}

namespace {

std::int32_t clampToInt32(std::int64_t deliveryTimeMs) {
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(
        deliveryTimeMs, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()));
}

}  // namespace

DeliveryHistogramResult deliveryTimeHistogram(const std::vector<std::int64_t>& deliveryTimesMs,
                                              GpuWorker* worker, std::chrono::milliseconds timeout) {
    // Compartido con la tarea: puede seguir viva despues de que esta funcion vuelva (plazo vencido).
    auto times = std::make_shared<std::vector<std::int32_t>>(deliveryTimesMs.size());
    std::transform(deliveryTimesMs.begin(), deliveryTimesMs.end(), times->begin(), clampToInt32);

    DeliveryHistogramResult result;
    if (worker != nullptr && worker->healthy()) {
        try {
            const auto deadline = deadlineAfterMs(timeout.count());
            std::future<std::vector<unsigned>> future = worker->submit(
                [times] {
                    std::vector<unsigned> bins;
                    const std::string errorMessage = deliveryTimeHistogramGpu(*times, bins);
                    if (!errorMessage.empty()) {
                        throw std::runtime_error(errorMessage);
                    }
                    return bins;
                },
                deadline);
            if (waitUntilDeadline(future, deadline)) {
                result.bins = future.get();
                result.usedGpu = true;
                return result;
            }
        } catch (const std::exception&) {
            // GPU no disponible, tarea vencida o fallida: respaldo CPU.
        }
    }
    result.bins = deliveryTimeHistogramCpu1(*times);
    result.usedGpu = false;
    return result;
}

std::string formatDeliveryHistogram(const std::vector<unsigned>& bins, const std::string& backendName) {
    const std::string prefix = "[estadisticas] histograma de entrega (min): ";
    std::uint64_t deliveryCount = 0;
    std::string nonEmptyBins;
    for (std::size_t binIndex = 0; binIndex < bins.size(); ++binIndex) {
        if (bins[binIndex] == 0) {
            continue;
        }
        deliveryCount += bins[binIndex];
        const bool isOverflowBin = binIndex + 1 == bins.size();
        nonEmptyBins += std::to_string(binIndex) + (isOverflowBin ? "+:" : ":") + std::to_string(bins[binIndex]) + " ";
    }
    if (deliveryCount == 0) {
        return prefix + "sin entregas";
    }
    return prefix + nonEmptyBins + "(n=" + std::to_string(deliveryCount) + ", backend=" + backendName + ")";
}

}  // namespace accel
