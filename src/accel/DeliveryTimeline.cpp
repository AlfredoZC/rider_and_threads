#include "accel/DeliveryTimeline.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>

#include "accel/OpenOrders.h"  // stripeOfChunk

namespace accel {

namespace {

// Rango [begin, end) de cada trozo de salida que parallelFor reparte: se usa tal cual (int).
void mergeOneRound(const DeliveryRecord* roundInput, DeliveryRecord* roundOutput, const std::vector<int>& listStart,
                   std::size_t totalCount, ThreadPool* pool) {
    const int listCount = static_cast<int>(listStart.size()) - 1;
    const int* listStartData = listStart.data();
    if (pool == nullptr) {
        mergeRoundRange(roundInput, roundOutput, listStartData, listCount, 0, static_cast<int>(totalCount));
        return;
    }
    pool->parallelFor(totalCount, [=](std::size_t chunkBegin, std::size_t chunkEnd) {
        mergeRoundRange(roundInput, roundOutput, listStartData, listCount, static_cast<int>(chunkBegin),
                        static_cast<int>(chunkEnd));
    });
}

std::vector<DeliveryRecord> mergeTree(std::vector<DeliveryRecord> records, std::vector<int> listStart, ThreadPool* pool) {
    std::vector<DeliveryRecord> roundOutput(records.size());
    while (listStart.size() > 2) {  // mientras haya mas de una lista
        mergeOneRound(records.data(), roundOutput.data(), listStart, records.size(), pool);
        records.swap(roundOutput);  // la salida de esta ronda es la entrada de la siguiente
        listStart = listStartAfterRound(listStart);
    }
    return records;
}

}  // namespace

std::vector<int> listStartAfterRound(const std::vector<int>& listStart) {
    const int listCount = static_cast<int>(listStart.size()) - 1;
    std::vector<int> nextListStart;
    for (int list = 0; list < listCount; list += 2) {
        nextListStart.push_back(listStart[static_cast<std::size_t>(list)]);
    }
    nextListStart.push_back(listStart[static_cast<std::size_t>(listCount)]);
    return nextListStart;
}

std::vector<DeliveryRecord> mergeCourierListsCpu1(std::vector<DeliveryRecord> records, std::vector<int> listStart) {
    return mergeTree(std::move(records), std::move(listStart), nullptr);
}

std::vector<DeliveryRecord> mergeCourierListsCpuN(std::vector<DeliveryRecord> records, std::vector<int> listStart,
                                                  ThreadPool& pool) {
    return mergeTree(std::move(records), std::move(listStart), &pool);
}

std::vector<int> deliveriesPerBucketCpu(const std::vector<DeliveryRecord>& timeline, int bucketMs) {
    if (timeline.empty()) {
        return {};
    }
    std::vector<int> perBucket(static_cast<std::size_t>(timeline.back().deliveredAtMs / bucketMs) + 1, 0);
    for (const DeliveryRecord& record : timeline) {
        ++perBucket[static_cast<std::size_t>(record.deliveredAtMs / bucketMs)];
    }
    return perBucket;
}

std::vector<int> deliveriesPerBucketBySearch(const std::vector<DeliveryRecord>& timeline, int bucketMs) {
    if (timeline.empty()) {
        return {};
    }
    const int count = static_cast<int>(timeline.size());
    std::vector<int> perBucket(static_cast<std::size_t>(timeline.back().deliveredAtMs / bucketMs) + 1, 0);
    for (std::size_t bucket = 0; bucket < perBucket.size(); ++bucket) {
        perBucket[bucket] = firstAtOrAfter(timeline.data(), count, static_cast<long long>(bucket + 1) * bucketMs) -
                            firstAtOrAfter(timeline.data(), count, static_cast<long long>(bucket) * bucketMs);
    }
    return perBucket;
}

TimelinePlan makeTimelinePlan(const std::vector<DeliveryRecord>& records, const std::vector<int>& listStart, int bucketMs) {
    TimelinePlan plan;
    plan.bucketMs = bucketMs;
    for (std::vector<int> roundStart = listStart; roundStart.size() > 2; roundStart = listStartAfterRound(roundStart)) {
        plan.allListStarts.insert(plan.allListStarts.end(), roundStart.begin(), roundStart.end());
        plan.listCountPerRound.push_back(static_cast<int>(roundStart.size()) - 1);
    }
    if (!records.empty()) {
        int latestMs = 0;
        for (const DeliveryRecord& record : records) {
            latestMs = std::max(latestMs, record.deliveredAtMs);
        }
        plan.bucketCount = latestMs / bucketMs + 1;
    }
    return plan;
}

std::string deliveryTimelineGpu(const std::vector<DeliveryRecord>& records, const std::vector<int>& listStart,
                                int bucketMs, DeliveryTimelineResult& result) {
    const TimelinePlan plan = makeTimelinePlan(records, listStart, bucketMs);
    DeviceTimeline device;
    std::string errorMessage = device.allocate(records.size(), plan);
    if (errorMessage.empty()) errorMessage = device.upload(records.data(), records.size(), plan, nullptr);
    if (errorMessage.empty()) errorMessage = device.run(nullptr);
    DeliveryTimelineResult gpuResult;
    gpuResult.timeline.assign(records.size(), DeliveryRecord());
    gpuResult.deliveriesPerBucket.assign(static_cast<std::size_t>(plan.bucketCount), 0);
    if (errorMessage.empty()) {
        errorMessage = device.download(gpuResult.timeline.data(), gpuResult.deliveriesPerBucket.data(), nullptr);
    }
    if (errorMessage.empty()) {
        result = std::move(gpuResult);
    }
    return errorMessage;
}

DeliveryTimelineSummary deliveryTimelineSummary(const std::vector<DeliveryRecord>& records,
                                                const std::vector<int>& listStart, GpuWorker* worker,
                                                std::chrono::milliseconds timeout) {
    DeliveryTimelineSummary summary;
    if (records.empty()) {
        return summary;
    }
    // Compartido con la tarea: puede seguir viva despues de que esta funcion vuelva (plazo vencido).
    const auto sharedRecords = std::make_shared<const std::vector<DeliveryRecord>>(records);
    const auto sharedListStart = std::make_shared<const std::vector<int>>(listStart);

    if (worker != nullptr && worker->healthy()) {
        try {
            const auto deadline = deadlineAfterMs(timeout.count());
            std::future<DeliveryTimelineResult> future = worker->submit(
                [sharedRecords, sharedListStart] {
                    DeliveryTimelineResult gpuResult;
                    const std::string errorMessage =
                        deliveryTimelineGpu(*sharedRecords, *sharedListStart, kMillisecondsPerHour, gpuResult);
                    if (!errorMessage.empty()) {
                        throw std::runtime_error(errorMessage);
                    }
                    return gpuResult;
                },
                deadline);
            if (waitUntilDeadline(future, deadline)) {
                summary.deliveriesPerHour = future.get().deliveriesPerBucket;
                summary.usedGpu = true;
            }
        } catch (const std::exception&) {
            // GPU no disponible, tarea vencida o fallida: respaldo CPU.
        }
    }
    if (!summary.usedGpu) {
        summary.deliveriesPerHour =
            deliveriesPerBucketCpu(mergeCourierListsCpu1(*sharedRecords, *sharedListStart), kMillisecondsPerHour);
    }
    for (const int perHour : summary.deliveriesPerHour) {
        summary.total += static_cast<std::size_t>(perHour);
    }
    return summary;
}

std::string formatDeliveriesPerHour(const std::vector<int>& deliveriesPerHour, const std::string& backendName) {
    const std::string prefix = "[estadisticas] entregas por hora simulada: ";
    if (deliveriesPerHour.empty()) {
        return prefix + "sin entregas";
    }
    std::string hours;
    for (std::size_t hour = 0; hour < deliveriesPerHour.size(); ++hour) {
        hours += (hour == 0 ? "h" : ", h") + std::to_string(hour) + " " + std::to_string(deliveriesPerHour[hour]);
    }
    return prefix + hours + " (backend=" + backendName + ")";
}

}  // namespace accel
