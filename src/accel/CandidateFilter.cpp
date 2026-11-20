#include "accel/CandidateFilter.h"

#include <algorithm>
#include <exception>
#include <future>
#include <limits>
#include <stdexcept>
#include <utility>

#include "accel/GpuContext.h"
#include "accel/GpuError.h"

namespace accel {

namespace {

// Valida las entradas del filtro (las mismas reglas para 1 y N hilos y para la GPU).
void validateFilterInput(const FleetColumns& fleet, const TravelTimeTable& table, std::int32_t restaurantNode) {
    if (fleet.node.size() != fleet.available.size()) {
        throw std::invalid_argument("filtro de candidatos: las columnas de la flota tienen tamanos distintos");
    }
    if (fleet.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("filtro de candidatos: flota demasiado grande");
    }
    if (restaurantNode < 0 || static_cast<std::size_t>(restaurantNode) >= table.nodeCount()) {
        throw std::invalid_argument("filtro de candidatos: restaurante fuera de la tabla");
    }
    for (std::size_t courier = 0; courier < fleet.size(); ++courier) {
        if (fleet.available[courier] != 0 &&
            (fleet.node[courier] < 0 || static_cast<std::size_t>(fleet.node[courier]) >= table.nodeCount())) {
            throw std::invalid_argument("filtro de candidatos: courier fuera de la tabla");
        }
    }
}

void filterRange(const FleetColumns& fleet, const TravelTimeTable& table, std::int32_t restaurantNode,
                 std::int32_t maxTravelMs, std::size_t firstCourier, std::size_t endCourier,
                 std::vector<std::int32_t>& candidates) {
    for (std::size_t courier = firstCourier; courier < endCourier; ++courier) {
        if (isCandidate(courier, fleet.node.data(), fleet.available.data(), table.cells().data(), table.nodeCount(),
                        restaurantNode, maxTravelMs)) {
            candidates.push_back(static_cast<std::int32_t>(courier));
        }
    }
}

}  // namespace

std::vector<std::int32_t> filterCandidatesSingleThread(const FleetColumns& fleet, const TravelTimeTable& table,
                                                       std::int32_t restaurantNode, std::int32_t maxTravelMs) {
    validateFilterInput(fleet, table, restaurantNode);
    std::vector<std::int32_t> candidates;
    filterRange(fleet, table, restaurantNode, maxTravelMs, 0, fleet.size(), candidates);
    return candidates;
}

std::vector<std::int32_t> filterCandidatesMultiThread(const FleetColumns& fleet, const TravelTimeTable& table,
                                                      std::int32_t restaurantNode, std::int32_t maxTravelMs,
                                                      ThreadPool& pool) {
    validateFilterInput(fleet, table, restaurantNode);
    const std::size_t stripeCount = pool.threads();
    std::vector<std::vector<std::int32_t>> stripeCandidates(stripeCount);  // una lista por franja
    pool.parallelFor(stripeCount, [&](std::size_t firstStripe, std::size_t endStripe) {
        for (std::size_t stripe = firstStripe; stripe < endStripe; ++stripe) {
            const Range couriers = chunkRange(fleet.size(), stripeCount, stripe);
            filterRange(fleet, table, restaurantNode, maxTravelMs, couriers.begin, couriers.end,
                        stripeCandidates[stripe]);
        }
    });
    std::vector<std::int32_t> candidates;
    for (const std::vector<std::int32_t>& stripeList : stripeCandidates) {
        candidates.insert(candidates.end(), stripeList.begin(), stripeList.end());
    }
    return candidates;
}

// Estado de la GPU del servicio: lo toca SOLO el hilo del GpuWorker (dentro de las tareas).
struct CandidateFilter::DeviceState {
    CandidateFilterDevice device;
    std::shared_ptr<const TravelTimeTable> uploadedTable;  // la tabla que esta en el device
};

CandidateFilter::CandidateFilter(GpuWorker* gpu) : gpu_(gpu), device_(std::make_shared<DeviceState>()) {}

CandidateFilter::~CandidateFilter() = default;

std::vector<std::int32_t> CandidateFilter::filter(const std::shared_ptr<const TravelTimeTable>& table,
                                                  FleetColumns fleet, std::int32_t restaurantNode,
                                                  std::int32_t maxTravelMs, std::int64_t timeoutMs) {
    validateFilterInput(fleet, *table, restaurantNode);
    // Todo lo que usa la tarea se captura por valor / shared_ptr: puede vivir mas que esta llamada.
    const auto columns = std::make_shared<const FleetColumns>(std::move(fleet));
    if (gpu_ != nullptr && gpu_->healthy() && columns->size() > 0) {
        try {
            const auto deadline = deadlineAfterMs(timeoutMs);
            const std::shared_ptr<DeviceState> device = device_;
            const std::shared_ptr<const TravelTimeTable> tableCopy = table;
            std::future<std::vector<std::int32_t>> future = gpu_->submit(
                [device, tableCopy, columns, restaurantNode, maxTravelMs] {
                    std::string errorMessage;
                    if (device->uploadedTable != tableCopy) {
                        device->uploadedTable = nullptr;  // si la subida falla no queda un puntero viejo
                        errorMessage = device->device.uploadTable(tableCopy->cells().data(), tableCopy->nodeCount(),
                                                                  nullptr);
                        if (errorMessage.empty()) device->uploadedTable = tableCopy;
                    }
                    if (errorMessage.empty()) errorMessage = device->device.uploadFleet(*columns, nullptr);
                    if (errorMessage.empty()) {
                        errorMessage = device->device.runFilter(false, restaurantNode, maxTravelMs, nullptr);
                    }
                    std::vector<std::int32_t> candidates;
                    if (errorMessage.empty()) errorMessage = device->device.download(candidates, nullptr);
                    if (!errorMessage.empty()) {
                        throwIfGpuError("filtro de candidatos");  // pegajoso -> GpuFatalError
                        throw std::runtime_error(errorMessage);
                    }
                    std::sort(candidates.begin(), candidates.end());  // mismo orden que la CPU
                    return candidates;
                },
                deadline);
            if (waitUntilDeadline(future, deadline)) {
                std::vector<std::int32_t> candidates = future.get();
                ++gpuFilters_;
                return candidates;
            }
            // Vencio: el future se descarta (la tarea sigue con sus copias) y se usa la CPU.
        } catch (const std::exception&) {
            // GPU no disponible, tarea vencida o fallida: respaldo CPU.
        }
    }
    ++cpuFilters_;
    return filterCandidatesSingleThread(*columns, *table, restaurantNode, maxTravelMs);
}

}  // namespace accel
