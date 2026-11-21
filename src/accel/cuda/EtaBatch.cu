// ETA en lote (aplicacion 04): una matriz pedidos x couriers, un hilo por par, el courier en el eje x.
// Solo kernels y envoltorios; la formula vive en accel/EtaFormula.h y la comparten la CPU y la GPU.
//
//   SoA  un arreglo por campo: los 32 hilos de un warp leen 32 enteros seguidos (lectura coalescida)
//   AoS  un registro de 32 bytes por courier: cada hilo lee 8 bytes de un registro de 32, asi que el warp
//        recorre 1024 bytes para usar 256
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "accel/EtaBatch.h"

namespace accel {

namespace {

constexpr int kBlockSize = 256;
constexpr std::size_t kMaxOrdersPerLaunch = 65535;  // gridDim.y

__global__ void computeEtaSoA(const std::int32_t* courierNode, const std::int32_t* courierFreeAtMs, int courierCount,
                              const std::int32_t* orderRestaurantNode, const std::int32_t* orderCustomerNode,
                              const std::int32_t* orderReadyAtMs, int orderCount, const std::int32_t* travelTimeMs,
                              std::size_t nodeCount, std::int32_t* etaMs) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;  // x: couriers -> lecturas coalescidas
    const int order = blockIdx.y;                              // y: un pedido por fila de bloques
    if (column >= courierCount || order >= orderCount) {
        return;  // sin barreras en el kernel: return seguro
    }
    etaMs[static_cast<std::size_t>(order) * courierCount + column] =  // escritura coalescida
        etaForPair(courierNode[column], courierFreeAtMs[column], orderRestaurantNode[order],
                   orderCustomerNode[order], orderReadyAtMs[order], travelTimeMs, nodeCount);
}

__global__ void computeEtaAoS(const CourierRecord* couriers, int courierCount, const std::int32_t* orderRestaurantNode,
                              const std::int32_t* orderCustomerNode, const std::int32_t* orderReadyAtMs,
                              int orderCount, const std::int32_t* travelTimeMs, std::size_t nodeCount,
                              std::int32_t* etaMs) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int order = blockIdx.y;
    if (column >= courierCount || order >= orderCount) {
        return;
    }
    const CourierRecord& courier = couriers[column];  // hilos vecinos: registros a 32 bytes
    etaMs[static_cast<std::size_t>(order) * courierCount + column] =
        etaForPair(courier.node, courier.freeAtMs, orderRestaurantNode[order], orderCustomerNode[order],
                   orderReadyAtMs[order], travelTimeMs, nodeCount);
}

// Un error de ejecucion se devuelve SIN limpiar el error pegajoso (lo ve throwIfGpuError).
std::string describeError(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return "";
    }
    return std::string(operation) + ": " + cudaGetErrorString(status);
}

void freePointer(void** pointer) {
    if (*pointer != nullptr) {
        cudaFree(*pointer);
        *pointer = nullptr;
    }
}

// Corre `work` entre dos cudaEvent (si hay eventos y se pide el tiempo).
template <typename Work>
std::string timedRegion(void* startEvent, void* stopEvent, double* elapsedMs, Work work) {
    if (elapsedMs == nullptr || startEvent == nullptr || stopEvent == nullptr) {
        return work();
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent);
    std::string errorMessage = describeError(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = work();
    if (errorMessage.empty()) errorMessage = describeError(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = describeError(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) {
        errorMessage = describeError(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    }
    if (errorMessage.empty()) *elapsedMs = elapsed;
    return errorMessage;
}

}  // namespace

EtaDevice::~EtaDevice() { release(); }

void EtaDevice::release() {
    freePointer(&tablePointer_);
    freePointer(&soaNodePointer_);
    freePointer(&soaFreePointer_);
    freePointer(&aosPointer_);
    freePointer(&restaurantPointer_);
    freePointer(&customerPointer_);
    freePointer(&readyPointer_);
    freePointer(&etaPointer_);
    tableNodeCount_ = soaCapacity_ = aosCapacity_ = orderCapacity_ = etaCapacity_ = 0;
    courierCount_ = orderCount_ = 0;
    if (startEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
        startEvent_ = nullptr;
    }
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
        stopEvent_ = nullptr;
    }
}

// Garantiza lugar para `count` elementos de `elementBytes`; un fallo de cudaMalloc es recuperable y se limpia.
std::string EtaDevice::ensure(void** pointer, std::size_t& capacity, std::size_t count, std::size_t elementBytes,
                              const char* what) {
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() / elementBytes) {
        return std::string(what) + ": tamano fuera de rango";
    }
    if (count <= capacity && *pointer != nullptr) {
        return "";
    }
    freePointer(pointer);
    capacity = 0;
    const cudaError_t status = cudaMalloc(pointer, count * elementBytes);
    if (status != cudaSuccess) {
        *pointer = nullptr;
        const std::string message = std::string("cudaMalloc de ") + what + ": " + cudaGetErrorString(status);
        cudaGetLastError();
        return message;
    }
    capacity = count;
    return "";
}

std::string EtaDevice::uploadTable(const std::int32_t* cells, std::size_t nodeCount, double* elapsedMs) {
    if (nodeCount == 0 || nodeCount > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) / nodeCount) {
        return "EtaDevice::uploadTable: tamano fuera de rango";
    }
    if (startEvent_ == nullptr) {
        cudaEvent_t startCreated = nullptr;
        cudaEvent_t stopCreated = nullptr;
        std::string errorMessage = describeError(cudaEventCreate(&startCreated), "cudaEventCreate");
        if (errorMessage.empty()) {
            startEvent_ = startCreated;
            errorMessage = describeError(cudaEventCreate(&stopCreated), "cudaEventCreate");
        }
        if (!errorMessage.empty()) return errorMessage;
        stopEvent_ = stopCreated;
    }
    std::string errorMessage;
    if (tablePointer_ == nullptr || tableNodeCount_ != nodeCount) {  // mismo tamano: se reutiliza la reserva
        freePointer(&tablePointer_);
        tableNodeCount_ = 0;
        std::size_t capacity = 0;
        errorMessage = ensure(&tablePointer_, capacity, nodeCount * nodeCount, sizeof(std::int32_t), "la tabla");
        if (!errorMessage.empty()) return errorMessage;
    }
    const std::size_t bytes = nodeCount * nodeCount * sizeof(std::int32_t);
    errorMessage = timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(tablePointer_, cells, bytes, cudaMemcpyHostToDevice), "cudaMemcpy tabla H2D");
    });
    if (errorMessage.empty()) tableNodeCount_ = nodeCount;
    return errorMessage;
}

std::string EtaDevice::uploadCouriersSoA(const CourierColumns& couriers, double* elapsedMs) {
    if (couriers.node.size() != couriers.freeAtMs.size()) return "EtaDevice: columnas de tamanos distintos";
    if (couriers.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return "EtaDevice: demasiados couriers";
    std::size_t capacity = soaCapacity_;
    std::string errorMessage = ensure(&soaNodePointer_, soaCapacity_, couriers.size(), sizeof(std::int32_t), "los nodos");
    if (errorMessage.empty()) {
        std::size_t freeCapacity = capacity;  // la segunda columna crece junto con la primera
        errorMessage = ensure(&soaFreePointer_, freeCapacity, couriers.size(), sizeof(std::int32_t), "los libre-desde");
    }
    if (!errorMessage.empty()) return errorMessage;
    courierCount_ = 0;
    errorMessage = timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        std::string copyError = describeError(cudaMemcpy(soaNodePointer_, couriers.node.data(),
                                                         couriers.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                                              "cudaMemcpy nodos H2D");
        if (copyError.empty()) {
            copyError = describeError(cudaMemcpy(soaFreePointer_, couriers.freeAtMs.data(),
                                                 couriers.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                                      "cudaMemcpy libre-desde H2D");
        }
        return copyError;
    });
    if (errorMessage.empty()) courierCount_ = couriers.size();
    return errorMessage;
}

std::string EtaDevice::uploadCouriersAoS(const std::vector<CourierRecord>& couriers, double* elapsedMs) {
    if (couriers.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return "EtaDevice: demasiados couriers";
    std::string errorMessage = ensure(&aosPointer_, aosCapacity_, couriers.size(), sizeof(CourierRecord), "los registros");
    if (!errorMessage.empty()) return errorMessage;
    courierCount_ = 0;
    errorMessage = timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(aosPointer_, couriers.data(), couriers.size() * sizeof(CourierRecord),
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy registros H2D");
    });
    if (errorMessage.empty()) courierCount_ = couriers.size();
    return errorMessage;
}

std::string EtaDevice::uploadOrders(const OrderColumns& orders, double* elapsedMs) {
    if (orders.restaurantNode.size() != orders.customerNode.size() || orders.restaurantNode.size() != orders.readyAtMs.size()) {
        return "EtaDevice: columnas de pedidos de tamanos distintos";
    }
    if (orders.size() > kMaxOrdersPerLaunch) return "EtaDevice: gridDim.y admite hasta 65535 pedidos por lanzamiento";
    std::size_t restaurantCapacity = orderCapacity_;
    std::size_t customerCapacity = orderCapacity_;
    std::size_t readyCapacity = orderCapacity_;
    std::string errorMessage =
        ensure(&restaurantPointer_, restaurantCapacity, orders.size(), sizeof(std::int32_t), "los restaurantes");
    if (errorMessage.empty()) {
        errorMessage = ensure(&customerPointer_, customerCapacity, orders.size(), sizeof(std::int32_t), "los clientes");
    }
    if (errorMessage.empty()) {
        errorMessage = ensure(&readyPointer_, readyCapacity, orders.size(), sizeof(std::int32_t), "los listos");
    }
    if (!errorMessage.empty()) return errorMessage;
    orderCapacity_ = std::max(orders.size(), orderCapacity_);
    orderCount_ = 0;
    const std::size_t bytes = orders.size() * sizeof(std::int32_t);
    errorMessage = timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        std::string copyError = describeError(
            cudaMemcpy(restaurantPointer_, orders.restaurantNode.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy restaurantes H2D");
        if (copyError.empty()) {
            copyError = describeError(cudaMemcpy(customerPointer_, orders.customerNode.data(), bytes, cudaMemcpyHostToDevice),
                                      "cudaMemcpy clientes H2D");
        }
        if (copyError.empty()) {
            copyError = describeError(cudaMemcpy(readyPointer_, orders.readyAtMs.data(), bytes, cudaMemcpyHostToDevice),
                                      "cudaMemcpy listos H2D");
        }
        return copyError;
    });
    if (errorMessage.empty()) orderCount_ = orders.size();
    return errorMessage;
}

std::string EtaDevice::runEta(bool aos, double* elapsedMs) {
    if (tablePointer_ == nullptr || courierCount_ == 0 || orderCount_ == 0) {
        return "EtaDevice::runEta: falta la tabla, los couriers o los pedidos";
    }
    if (orderCount_ > kMaxOrdersPerLaunch) return "EtaDevice::runEta: demasiados pedidos";
    if (courierCount_ > std::numeric_limits<std::size_t>::max() / orderCount_) return "EtaDevice::runEta: matriz demasiado grande";
    std::size_t capacity = etaCapacity_;
    std::string errorMessage = ensure(&etaPointer_, capacity, orderCount_ * courierCount_, sizeof(std::int32_t), "la matriz");
    if (!errorMessage.empty()) return errorMessage;
    etaCapacity_ = std::max(etaCapacity_, orderCount_ * courierCount_);
    const int courierCount = static_cast<int>(courierCount_);
    const int orderCount = static_cast<int>(orderCount_);
    const dim3 blocks(static_cast<unsigned>((courierCount + kBlockSize - 1) / kBlockSize), static_cast<unsigned>(orderCount));
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        const std::int32_t* table = static_cast<const std::int32_t*>(tablePointer_);
        const std::int32_t* restaurants = static_cast<const std::int32_t*>(restaurantPointer_);
        const std::int32_t* customers = static_cast<const std::int32_t*>(customerPointer_);
        const std::int32_t* ready = static_cast<const std::int32_t*>(readyPointer_);
        std::int32_t* eta = static_cast<std::int32_t*>(etaPointer_);
        if (aos) {
            computeEtaAoS<<<blocks, kBlockSize>>>(static_cast<const CourierRecord*>(aosPointer_), courierCount,
                                                  restaurants, customers, ready, orderCount, table, tableNodeCount_, eta);
        } else {
            computeEtaSoA<<<blocks, kBlockSize>>>(static_cast<const std::int32_t*>(soaNodePointer_),
                                                  static_cast<const std::int32_t*>(soaFreePointer_), courierCount,
                                                  restaurants, customers, ready, orderCount, table, tableNodeCount_, eta);
        }
        return describeError(cudaGetLastError(), "lanzamiento del kernel de ETA");
    });
}

std::string EtaDevice::download(std::vector<std::int32_t>& eta, double* elapsedMs) {
    if (etaPointer_ == nullptr || courierCount_ == 0 || orderCount_ == 0) return "EtaDevice::download: sin matriz";
    const std::size_t count = orderCount_ * courierCount_;
    eta.resize(count);
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(eta.data(), etaPointer_, count * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                             "cudaMemcpy matriz D2H");
    });
}

}  // namespace accel
