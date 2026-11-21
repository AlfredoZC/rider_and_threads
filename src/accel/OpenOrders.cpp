#include "accel/OpenOrders.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace accel {

namespace {

std::int32_t clampToInt32(std::int64_t timeMs) {
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(
        timeMs, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()));
}

}  // namespace

std::size_t stripeOfChunk(std::size_t itemCount, std::size_t stripeCount, std::size_t chunkBegin) {
    for (std::size_t stripe = 0; stripe < stripeCount; ++stripe) {
        const Range range = chunkRange(itemCount, stripeCount, stripe);
        if (range.end > range.begin && range.begin == chunkBegin) {
            return stripe;
        }
    }
    throw std::logic_error("stripeOfChunk: trozo desconocido");
}

OrderEvents buildOrderEvents(const std::vector<OrderLifetime>& orders) {
    struct Event {
        std::int32_t timeMs;
        std::int32_t change;
    };
    std::vector<Event> events;
    events.reserve(2 * orders.size());
    for (const OrderLifetime& order : orders) {
        events.push_back({clampToInt32(order.createdAtMs), +1});
        if (order.closedAtMs != kStillOpen) {
            events.push_back({clampToInt32(order.closedAtMs), -1});
        }
    }
    std::sort(events.begin(), events.end(), [](const Event& left, const Event& right) {
        if (left.timeMs != right.timeMs) return left.timeMs < right.timeMs;
        return left.change > right.change;  // empate: +1 antes que -1
    });
    OrderEvents result;
    result.timeMs.reserve(events.size());
    result.change.reserve(events.size());
    for (const Event& event : events) {
        result.timeMs.push_back(event.timeMs);
        result.change.push_back(event.change);
    }
    return result;
}

OpenOrdersResult openOrdersCpu1(const std::vector<std::int32_t>& change) {
    OpenOrdersResult result;
    result.openOrders.resize(change.size());
    std::int32_t runningTotal = 0;
    for (std::size_t eventIndex = 0; eventIndex < change.size(); ++eventIndex) {
        runningTotal += change[eventIndex];
        result.openOrders[eventIndex] = runningTotal;
        if (runningTotal > result.peakOpenOrders) {  // ">" estricto: se queda con el PRIMER pico
            result.peakOpenOrders = runningTotal;
            result.peakEvent = eventIndex;
        }
    }
    return result;
}

OpenOrdersResult openOrdersCpuN(const std::vector<std::int32_t>& change, ThreadPool& pool) {
    const std::size_t eventCount = change.size();
    const std::size_t stripeCount = pool.threads();
    const std::int32_t* changeData = change.data();

    // Pasada 1 (paralela): cada franja suma sus cambios; cada hilo escribe SOLO su casilla.
    std::vector<std::int32_t> stripeTotal(stripeCount, 0);
    std::int32_t* stripeTotalData = stripeTotal.data();
    pool.parallelFor(eventCount, [=](std::size_t chunkBegin, std::size_t chunkEnd) {
        std::int32_t total = 0;
        for (std::size_t eventIndex = chunkBegin; eventIndex < chunkEnd; ++eventIndex) {
            total += changeData[eventIndex];
        }
        stripeTotalData[stripeOfChunk(eventCount, stripeCount, chunkBegin)] = total;
    });

    // En el medio (1 hilo, stripeCount numeros): cuanto suman todas las franjas anteriores.
    std::vector<std::int32_t> totalBeforeStripe(stripeCount, 0);
    for (std::size_t stripe = 1; stripe < stripeCount; ++stripe) {
        totalBeforeStripe[stripe] = totalBeforeStripe[stripe - 1] + stripeTotal[stripe - 1];
    }

    // Pasada 2 (paralela): scan de la franja desde su offset y pico de la franja.
    OpenOrdersResult result;
    result.openOrders.resize(eventCount);
    std::int32_t* openOrdersData = result.openOrders.data();
    std::vector<std::int32_t> stripePeak(stripeCount, 0);
    std::vector<std::size_t> stripePeakEvent(stripeCount, 0);
    const std::int32_t* totalBeforeData = totalBeforeStripe.data();
    std::int32_t* stripePeakData = stripePeak.data();
    std::size_t* stripePeakEventData = stripePeakEvent.data();
    pool.parallelFor(eventCount, [=](std::size_t chunkBegin, std::size_t chunkEnd) {
        const std::size_t stripe = stripeOfChunk(eventCount, stripeCount, chunkBegin);
        std::int32_t runningTotal = totalBeforeData[stripe];
        std::int32_t peak = 0;
        std::size_t peakEvent = 0;
        for (std::size_t eventIndex = chunkBegin; eventIndex < chunkEnd; ++eventIndex) {
            runningTotal += changeData[eventIndex];
            openOrdersData[eventIndex] = runningTotal;  // franjas disjuntas: nadie mas escribe aqui
            if (runningTotal > peak) {
                peak = runningTotal;
                peakEvent = eventIndex;
            }
        }
        stripePeakData[stripe] = peak;
        stripePeakEventData[stripe] = peakEvent;
    });

    // Pico global: el mayor de las franjas; con ">" gana la franja mas temprana.
    for (std::size_t stripe = 0; stripe < stripeCount; ++stripe) {
        if (stripePeak[stripe] > result.peakOpenOrders) {
            result.peakOpenOrders = stripePeak[stripe];
            result.peakEvent = stripePeakEvent[stripe];
        }
    }
    return result;
}

void applyPeakKey(unsigned long long key, OpenOrdersResult& result) {
    result.peakOpenOrders = static_cast<std::int32_t>(key >> 32);
    result.peakEvent = static_cast<std::size_t>(0xFFFFFFFFu - static_cast<std::uint32_t>(key & 0xFFFFFFFFu));
}

std::string openOrdersGpu(const std::vector<std::int32_t>& change, OpenOrdersResult& result) {
    DeviceOpenOrders device;
    std::string errorMessage = device.allocate(change.size());
    if (errorMessage.empty()) errorMessage = device.upload(change.data(), change.size(), nullptr);
    if (errorMessage.empty()) errorMessage = device.run(nullptr);
    OpenOrdersResult gpuResult;
    gpuResult.openOrders.assign(change.size(), 0);
    unsigned long long peakKeyOnHost = 0;
    if (errorMessage.empty()) errorMessage = device.download(gpuResult.openOrders.data(), &peakKeyOnHost, nullptr);
    if (errorMessage.empty()) {
        if (!change.empty()) {
            applyPeakKey(peakKeyOnHost, gpuResult);
        }
        result = std::move(gpuResult);
    }
    return errorMessage;
}

OpenOrdersSummary openOrdersSummary(const std::vector<OrderLifetime>& orders, GpuWorker* worker,
                                    std::chrono::milliseconds timeout) {
    // Compartido con la tarea: puede seguir viva despues de que esta funcion vuelva (plazo vencido).
    const auto events = std::make_shared<const OrderEvents>(buildOrderEvents(orders));

    OpenOrdersResult result;
    bool usedGpu = false;
    if (worker != nullptr && worker->healthy() && !events->change.empty()) {
        try {
            const auto deadline = deadlineAfterMs(timeout.count());
            std::future<OpenOrdersResult> future = worker->submit(
                [events] {
                    OpenOrdersResult gpuResult;
                    const std::string errorMessage = openOrdersGpu(events->change, gpuResult);
                    if (!errorMessage.empty()) {
                        throw std::runtime_error(errorMessage);
                    }
                    return gpuResult;
                },
                deadline);
            if (waitUntilDeadline(future, deadline)) {
                result = future.get();
                usedGpu = true;
            }
        } catch (const std::exception&) {
            // GPU no disponible, tarea vencida o fallida: respaldo CPU.
        }
    }
    if (!usedGpu) {
        result = openOrdersCpu1(events->change);
    }

    OpenOrdersSummary summary;
    summary.usedGpu = usedGpu;
    summary.peakOpenOrders = result.peakOpenOrders;
    if (!events->change.empty()) {
        summary.peakTimeMs = events->timeMs[result.peakEvent];
        summary.openAtEnd = result.openOrders.back();
    }
    return summary;
}

std::string formatOpenOrders(std::int32_t peakOpenOrders, std::int32_t peakTimeMs, std::int32_t openAtEnd,
                             const std::string& backendName) {
    const std::string prefix = "[estadisticas] pedidos en curso: ";
    if (peakOpenOrders == 0 && openAtEnd == 0) {
        return prefix + "sin pedidos";
    }
    char seconds[32];
    std::snprintf(seconds, sizeof seconds, "%.1f", static_cast<double>(peakTimeMs) / 1000.0);
    return prefix + "pico " + std::to_string(peakOpenOrders) + " en t=" + seconds + "s, abiertos al final " +
           std::to_string(openAtEnd) + " (backend=" + backendName + ")";
}

}  // namespace accel
