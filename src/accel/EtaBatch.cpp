#include "accel/EtaBatch.h"

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

// Las mismas reglas para 1 y N hilos y para la GPU.
void validateEtaInput(const CourierColumns& couriers, const OrderColumns& orders, const TravelTimeTable& table) {
    if (couriers.node.size() != couriers.freeAtMs.size() || orders.restaurantNode.size() != orders.customerNode.size() ||
        orders.restaurantNode.size() != orders.readyAtMs.size()) {
        throw std::invalid_argument("ETA en lote: las columnas tienen tamanos distintos");
    }
    const std::size_t nodeCount = table.nodeCount();
    auto outside = [nodeCount](std::int32_t node) { return node < 0 || static_cast<std::size_t>(node) >= nodeCount; };
    if (couriers.size() != 0 && orders.size() != 0 &&
        couriers.size() > std::numeric_limits<std::size_t>::max() / orders.size()) {
        throw std::invalid_argument("ETA en lote: la matriz no cabe en memoria");
    }
    if (std::any_of(couriers.node.begin(), couriers.node.end(), outside) ||
        std::any_of(orders.restaurantNode.begin(), orders.restaurantNode.end(), outside) ||
        std::any_of(orders.customerNode.begin(), orders.customerNode.end(), outside)) {
        throw std::invalid_argument("ETA en lote: un nodo esta fuera de la tabla");
    }
}

void fillColumns(const CourierColumns& couriers, const OrderColumns& orders, const TravelTimeTable& table,
                 std::size_t firstColumn, std::size_t endColumn, std::vector<std::int32_t>& eta) {
    const std::size_t columnCount = couriers.size();
    for (std::size_t order = 0; order < orders.size(); ++order) {
        for (std::size_t column = firstColumn; column < endColumn; ++column) {
            eta[order * columnCount + column] =
                etaForPair(couriers.node[column], couriers.freeAtMs[column], orders.restaurantNode[order],
                           orders.customerNode[order], orders.readyAtMs[order], table.cells().data(),
                           table.nodeCount());
        }
    }
}

}  // namespace

CourierColumns gatherColumns(const CourierColumns& fleet, const std::vector<std::int32_t>& courierIds) {
    CourierColumns columns;
    columns.node.reserve(courierIds.size());
    columns.freeAtMs.reserve(courierIds.size());
    for (const std::int32_t courier : courierIds) {
        if (courier < 0 || static_cast<std::size_t>(courier) >= fleet.size()) {
            throw std::out_of_range("gatherColumns: courier inexistente");
        }
        columns.node.push_back(fleet.node[static_cast<std::size_t>(courier)]);
        columns.freeAtMs.push_back(fleet.freeAtMs[static_cast<std::size_t>(courier)]);
    }
    return columns;
}

std::vector<CourierRecord> toRecords(const CourierColumns& columns) {
    std::vector<CourierRecord> records(columns.size());
    for (std::size_t courier = 0; courier < columns.size(); ++courier) {
        const std::int32_t marker = static_cast<std::int32_t>(courier % 1000);
        records[courier] = CourierRecord{columns.node[courier], columns.freeAtMs[courier], marker % 3, 3, marker % 6,
                                         marker, marker / 7, marker * 13};
    }
    return records;
}

std::vector<std::int32_t> computeEtaSingleThread(const CourierColumns& couriers, const OrderColumns& orders,
                                                 const TravelTimeTable& table) {
    validateEtaInput(couriers, orders, table);
    std::vector<std::int32_t> eta(orders.size() * couriers.size());
    fillColumns(couriers, orders, table, 0, couriers.size(), eta);
    return eta;
}

std::vector<std::int32_t> computeEtaMultiThread(const CourierColumns& couriers, const OrderColumns& orders,
                                                const TravelTimeTable& table, ThreadPool& pool) {
    validateEtaInput(couriers, orders, table);
    std::vector<std::int32_t> eta(orders.size() * couriers.size());  // tamano fijo ANTES de repartir
    pool.parallelFor(couriers.size(), [&](std::size_t firstColumn, std::size_t endColumn) {
        fillColumns(couriers, orders, table, firstColumn, endColumn, eta);  // columnas distintas: sin choques
    });
    return eta;
}

// Estado de la GPU del servicio: lo toca SOLO el hilo del GpuWorker (dentro de las tareas).
struct EtaBatch::DeviceState {
    EtaDevice device;
    std::shared_ptr<const TravelTimeTable> uploadedTable;
};

EtaBatch::EtaBatch(GpuWorker* gpu) : gpu_(gpu), device_(std::make_shared<DeviceState>()) {}

EtaBatch::~EtaBatch() = default;

std::vector<std::int32_t> EtaBatch::compute(const std::shared_ptr<const TravelTimeTable>& table,
                                            CourierColumns couriers, OrderColumns orders, std::int64_t timeoutMs) {
    validateEtaInput(couriers, orders, *table);
    // Todo lo que usa la tarea se captura por valor / shared_ptr: puede vivir mas que esta llamada.
    const auto courierColumns = std::make_shared<const CourierColumns>(std::move(couriers));
    const auto orderColumns = std::make_shared<const OrderColumns>(std::move(orders));
    if (gpu_ != nullptr && gpu_->healthy() && courierColumns->size() > 0 && orderColumns->size() > 0) {
        try {
            const auto deadline = deadlineAfterMs(timeoutMs);
            const std::shared_ptr<DeviceState> device = device_;
            const std::shared_ptr<const TravelTimeTable> tableCopy = table;
            std::future<std::vector<std::int32_t>> future = gpu_->submit(
                [device, tableCopy, courierColumns, orderColumns] {
                    std::string errorMessage;
                    if (device->uploadedTable != tableCopy) {
                        errorMessage = device->device.uploadTable(tableCopy->cells().data(), tableCopy->nodeCount(),
                                                                  nullptr);
                        if (errorMessage.empty()) device->uploadedTable = tableCopy;
                    }
                    if (errorMessage.empty()) errorMessage = device->device.uploadCouriersSoA(*courierColumns, nullptr);
                    if (errorMessage.empty()) errorMessage = device->device.uploadOrders(*orderColumns, nullptr);
                    if (errorMessage.empty()) errorMessage = device->device.runEta(false, nullptr);
                    std::vector<std::int32_t> eta;
                    if (errorMessage.empty()) errorMessage = device->device.download(eta, nullptr);
                    if (!errorMessage.empty()) {
                        throwIfGpuError("ETA en lote");  // pegajoso -> GpuFatalError
                        throw std::runtime_error(errorMessage);
                    }
                    return eta;
                },
                deadline);
            if (waitUntilDeadline(future, deadline)) {
                std::vector<std::int32_t> eta = future.get();
                ++gpuBatches_;
                return eta;
            }
            // Vencio: el future se descarta (la tarea sigue con sus copias) y se usa la CPU.
        } catch (const std::exception&) {
            // GPU no disponible, tarea vencida o fallida: respaldo CPU con la misma formula.
        }
    }
    ++cpuBatches_;
    return computeEtaSingleThread(*courierColumns, *orderColumns, *table);
}

}  // namespace accel
