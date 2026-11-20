// Filtro de candidatos (aplicacion 10): compactar los couriers que pasan un predicado. Solo kernels y
// envoltorios; la logica del filtro (el predicado) vive en accel/CandidatePredicate.h y la comparten la
// CPU y la GPU.
//
//   ingenuo        un atomicAdd GLOBAL por cada candidato: todos los hilos chocan en el mismo contador
//   por bloque     cada bloque junta sus candidatos en una lista privada en memoria compartida con un
//                  atomico en compartida y despues reserva su tramo de la salida con UN atomico global
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "accel/CandidateFilter.h"

namespace accel {

namespace {

constexpr int kBlockSize = 256;

__global__ void filterCandidatesNaive(const std::int32_t* courierNode, const std::uint8_t* courierAvailable,
                                      int courierCount, const std::int32_t* travelTimeMs, std::size_t nodeCount,
                                      std::int32_t restaurantNode, std::int32_t maxTravelMs,
                                      std::int32_t* candidateCouriers, int* candidateCount) {
    const int courier = blockIdx.x * blockDim.x + threadIdx.x;  // un hilo por courier
    if (courier < courierCount && isCandidate(static_cast<std::size_t>(courier), courierNode, courierAvailable,
                                              travelTimeMs, nodeCount, restaurantNode, maxTravelMs)) {
        const int slot = atomicAdd(candidateCount, 1);  // todos chocan en el MISMO contador
        candidateCouriers[slot] = courier;
    }
}

template <int BLOCK_SIZE>
__global__ void filterCandidatesBlockPrivate(const std::int32_t* courierNode, const std::uint8_t* courierAvailable,
                                             int courierCount, const std::int32_t* travelTimeMs,
                                             std::size_t nodeCount, std::int32_t restaurantNode,
                                             std::int32_t maxTravelMs, std::int32_t* candidateCouriers,
                                             int* candidateCount) {
    __shared__ std::int32_t blockCandidates[BLOCK_SIZE];  // lista privada del bloque
    __shared__ int blockCandidateCount;                   // contador privado del bloque
    __shared__ int blockStartInOutput;                    // donde empieza el tramo del bloque en la salida
    if (threadIdx.x == 0) {
        blockCandidateCount = 0;
    }
    __syncthreads();  // el contador esta en 0 antes de que alguien sume

    // Sin return temprano: los hilos sobrantes tambien llegan a las barreras.
    const int courier = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (courier < courierCount && isCandidate(static_cast<std::size_t>(courier), courierNode, courierAvailable,
                                              travelTimeMs, nodeCount, restaurantNode, maxTravelMs)) {
        const int slot = atomicAdd(&blockCandidateCount, 1);  // atomico en compartida: barato
        blockCandidates[slot] = courier;
    }
    __syncthreads();  // el bloque termino de contar

    if (threadIdx.x == 0) {
        blockStartInOutput = atomicAdd(candidateCount, blockCandidateCount);  // UN atomico global por bloque
    }
    __syncthreads();  // todos ven blockStartInOutput

    if (static_cast<int>(threadIdx.x) < blockCandidateCount) {  // hilos consecutivos, posiciones consecutivas
        candidateCouriers[blockStartInOutput + threadIdx.x] = blockCandidates[threadIdx.x];
    }
}

// Un error de ejecucion se devuelve SIN limpiar el error pegajoso (lo ve throwIfGpuError).
std::string describeError(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return "";
    }
    return std::string(operation) + ": " + cudaGetErrorString(status);
}

// Reserva `count` elementos; un fallo de cudaMalloc es recuperable y se limpia.
template <typename Element>
std::string allocateArray(void** pointer, std::size_t count, const char* what) {
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
        return std::string(what) + ": tamano fuera de rango";
    }
    const cudaError_t status = cudaMalloc(pointer, count * sizeof(Element));
    if (status != cudaSuccess) {
        *pointer = nullptr;
        const std::string message = std::string("cudaMalloc de ") + what + ": " + cudaGetErrorString(status);
        cudaGetLastError();
        return message;
    }
    return "";
}

void freePointer(void** pointer) {
    if (*pointer != nullptr) {
        cudaFree(*pointer);
        *pointer = nullptr;
    }
}

// Corre `work` entre dos cudaEvent (si hay eventos) y devuelve el tiempo en ms.
template <typename Work>
std::string timedRegion(void* startEvent, void* stopEvent, double* elapsedMs, Work work) {
    if (elapsedMs == nullptr || startEvent == nullptr || stopEvent == nullptr) {
        std::string errorMessage = work();
        return errorMessage;
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

CandidateFilterDevice::~CandidateFilterDevice() { release(); }

void CandidateFilterDevice::releaseFleet() {
    freePointer(&nodePointer_);
    freePointer(&availablePointer_);
    freePointer(&candidatesPointer_);
    freePointer(&countPointer_);
    fleetCapacity_ = 0;
    fleetSize_ = 0;
}

void CandidateFilterDevice::release() {
    releaseFleet();
    freePointer(&tablePointer_);
    tableNodeCount_ = 0;
    if (startEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
        startEvent_ = nullptr;
    }
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
        stopEvent_ = nullptr;
    }
}

std::string CandidateFilterDevice::uploadTable(const std::int32_t* cells, std::size_t nodeCount, double* elapsedMs) {
    if (nodeCount == 0 || nodeCount > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) / nodeCount) {
        return "CandidateFilterDevice::uploadTable: tamano fuera de rango";
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
        errorMessage = allocateArray<std::int32_t>(&tablePointer_, nodeCount * nodeCount, "la tabla");
        if (!errorMessage.empty()) return errorMessage;
    }
    const std::size_t bytes = nodeCount * nodeCount * sizeof(std::int32_t);
    errorMessage = timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(tablePointer_, cells, bytes, cudaMemcpyHostToDevice), "cudaMemcpy tabla H2D");
    });
    if (errorMessage.empty()) tableNodeCount_ = nodeCount;
    return errorMessage;
}

std::string CandidateFilterDevice::reserveFleet(std::size_t courierCount) {
    if (courierCount == 0 || courierCount > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return "CandidateFilterDevice::reserveFleet: tamano fuera de rango";
    }
    if (courierCount <= fleetCapacity_) {
        return "";
    }
    releaseFleet();
    std::string errorMessage = allocateArray<std::int32_t>(&nodePointer_, courierCount, "los nodos de la flota");
    if (errorMessage.empty()) {
        errorMessage = allocateArray<std::uint8_t>(&availablePointer_, courierCount, "la disponibilidad de la flota");
    }
    if (errorMessage.empty()) {
        errorMessage = allocateArray<std::int32_t>(&candidatesPointer_, courierCount, "la lista de candidatos");
    }
    if (errorMessage.empty()) errorMessage = allocateArray<int>(&countPointer_, 1, "el contador");
    if (!errorMessage.empty()) {
        releaseFleet();
        return errorMessage;
    }
    fleetCapacity_ = courierCount;
    return "";
}

std::string CandidateFilterDevice::uploadFleet(const FleetColumns& fleet, double* elapsedMs) {
    if (fleet.node.size() != fleet.available.size()) {
        return "CandidateFilterDevice::uploadFleet: columnas de tamanos distintos";
    }
    std::string errorMessage = reserveFleet(fleet.size());
    if (!errorMessage.empty()) return errorMessage;
    fleetSize_ = 0;
    errorMessage = timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        std::string copyError = describeError(
            cudaMemcpy(nodePointer_, fleet.node.data(), fleet.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice),
            "cudaMemcpy nodos H2D");
        if (copyError.empty()) {
            copyError = describeError(cudaMemcpy(availablePointer_, fleet.available.data(),
                                                 fleet.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice),
                                      "cudaMemcpy disponibilidad H2D");
        }
        return copyError;
    });
    if (errorMessage.empty()) fleetSize_ = fleet.size();
    return errorMessage;
}

std::string CandidateFilterDevice::runFilter(bool naive, std::int32_t restaurantNode, std::int32_t maxTravelMs,
                                             double* elapsedMs) {
    if (tablePointer_ == nullptr || fleetSize_ == 0) return "CandidateFilterDevice::runFilter: falta la tabla o la flota";
    if (restaurantNode < 0 || static_cast<std::size_t>(restaurantNode) >= tableNodeCount_) {
        return "CandidateFilterDevice::runFilter: restaurante fuera de la tabla";
    }
    const int courierCount = static_cast<int>(fleetSize_);
    const int blockCount = (courierCount + kBlockSize - 1) / kBlockSize;
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        std::string errorMessage = describeError(cudaMemset(countPointer_, 0, sizeof(int)), "cudaMemset contador");
        if (!errorMessage.empty()) return errorMessage;
        const std::int32_t* nodes = static_cast<const std::int32_t*>(nodePointer_);
        const std::uint8_t* available = static_cast<const std::uint8_t*>(availablePointer_);
        const std::int32_t* table = static_cast<const std::int32_t*>(tablePointer_);
        std::int32_t* candidates = static_cast<std::int32_t*>(candidatesPointer_);
        int* count = static_cast<int*>(countPointer_);
        if (naive) {
            filterCandidatesNaive<<<blockCount, kBlockSize>>>(nodes, available, courierCount, table, tableNodeCount_,
                                                              restaurantNode, maxTravelMs, candidates, count);
        } else {
            filterCandidatesBlockPrivate<kBlockSize><<<blockCount, kBlockSize>>>(
                nodes, available, courierCount, table, tableNodeCount_, restaurantNode, maxTravelMs, candidates,
                count);
        }
        return describeError(cudaGetLastError(), "lanzamiento del filtro de candidatos");
    });
}

std::string CandidateFilterDevice::download(std::vector<std::int32_t>& candidates, double* elapsedMs) {
    if (countPointer_ == nullptr) return "CandidateFilterDevice::download: sin reservar";
    int candidateCount = 0;
    std::string errorMessage = timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        // cudaMemcpy espera al kernel: primero cuantos hay, despues solo esos.
        std::string copyError = describeError(
            cudaMemcpy(&candidateCount, countPointer_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy contador D2H");
        if (!copyError.empty()) return copyError;
        if (candidateCount < 0 || static_cast<std::size_t>(candidateCount) > fleetSize_) {
            return std::string("el contador de candidatos es invalido");
        }
        candidates.resize(static_cast<std::size_t>(candidateCount));
        if (candidateCount > 0) {
            copyError = describeError(cudaMemcpy(candidates.data(), candidatesPointer_,
                                                 static_cast<std::size_t>(candidateCount) * sizeof(std::int32_t),
                                                 cudaMemcpyDeviceToHost),
                                      "cudaMemcpy candidatos D2H");
        }
        return copyError;
    });
    return errorMessage;
}

}  // namespace accel
