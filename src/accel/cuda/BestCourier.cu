// Mejor courier (aplicacion 08): una reduccion de minimo por fila de la matriz de ETA, un bloque por pedido.
// Solo kernels y envoltorios; la clave (ETA << 32 | id real) y el minimo viven en accel/BestCourier.h.
//
// Fase 1: cada hilo recorre su parte de la fila con paso BLOCK_SIZE (lecturas coalescidas) y se queda con su minimo.
// Fase 2: arbol en memoria compartida.
//   divergente  empareja vecinos (stride 1, 2, 4...): los hilos activos quedan desparramados en cada warp
//   convergente stride BLOCK_SIZE/2, /4...: los hilos activos son siempre los primeros, los warps de arriba
//               salen enteros y no hay divergencia dentro de un warp hasta el final
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "accel/BestCourier.h"

namespace accel {

namespace {

constexpr int kBlockSize = 256;

template <int BLOCK_SIZE>
__device__ unsigned long long threadBestKeyOfRow(const std::int32_t* etaMs, const std::int32_t* courierIds,
                                                 int courierCount, int order) {
    const std::size_t rowStart = static_cast<std::size_t>(order) * courierCount;
    unsigned long long threadBestKey = kNoCourierKey;
    for (int column = threadIdx.x; column < courierCount; column += BLOCK_SIZE) {
        threadBestKey = minKey(threadBestKey, packCourierKey(etaMs[rowStart + column], courierIds[column]));
    }
    return threadBestKey;
}

template <int BLOCK_SIZE>
__global__ void bestCourierDivergent(const std::int32_t* etaMs, const std::int32_t* courierIds, int courierCount,
                                     unsigned long long* bestKeyPerOrder) {
    __shared__ unsigned long long partialBestKey[BLOCK_SIZE];
    const int order = blockIdx.x;
    partialBestKey[threadIdx.x] = threadBestKeyOfRow<BLOCK_SIZE>(etaMs, courierIds, courierCount, order);
    __syncthreads();
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        if (threadIdx.x % (2 * stride) == 0) {  // activos: 0, 2, 4...; despues 0, 4, 8...: desparramados
            partialBestKey[threadIdx.x] = minKey(partialBestKey[threadIdx.x], partialBestKey[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        bestKeyPerOrder[order] = partialBestKey[0];
    }
}

template <int BLOCK_SIZE>
__global__ void bestCourierConvergent(const std::int32_t* etaMs, const std::int32_t* courierIds, int courierCount,
                                      unsigned long long* bestKeyPerOrder) {
    static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE tiene que ser potencia de 2");
    __shared__ unsigned long long partialBestKey[BLOCK_SIZE];  // un minimo parcial por hilo
    const int order = blockIdx.x;                              // un bloque por pedido
    partialBestKey[threadIdx.x] = threadBestKeyOfRow<BLOCK_SIZE>(etaMs, courierIds, courierCount, order);
    __syncthreads();  // todos los parciales escritos
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (static_cast<int>(threadIdx.x) < stride) {  // activos: los primeros "stride" hilos
            partialBestKey[threadIdx.x] = minKey(partialBestKey[threadIdx.x], partialBestKey[threadIdx.x + stride]);
        }
        __syncthreads();  // el paso siguiente lee lo que escribio este
    }
    if (threadIdx.x == 0) {
        bestKeyPerOrder[order] = partialBestKey[0];
    }
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

// Garantiza lugar para `count` elementos; un fallo de cudaMalloc es recuperable y se limpia.
std::string ensure(void** pointer, std::size_t& capacity, std::size_t count, std::size_t elementBytes, const char* what) {
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

// Los cudaEvent se crean la primera vez que hacen falta.
std::string createEvents(void** startEvent, void** stopEvent) {
    if (*startEvent != nullptr) {
        return "";
    }
    cudaEvent_t startCreated = nullptr;
    cudaEvent_t stopCreated = nullptr;
    std::string errorMessage = describeError(cudaEventCreate(&startCreated), "cudaEventCreate");
    if (errorMessage.empty()) {
        *startEvent = startCreated;
        errorMessage = describeError(cudaEventCreate(&stopCreated), "cudaEventCreate");
    }
    if (errorMessage.empty()) *stopEvent = stopCreated;
    return errorMessage;
}

}  // namespace

BestCourierDevice::~BestCourierDevice() { release(); }

void BestCourierDevice::release() {
    freePointer(&etaPointer_);
    freePointer(&idsPointer_);
    freePointer(&keysPointer_);
    etaCapacity_ = idsCapacity_ = keysCapacity_ = 0;
    orderCount_ = 0;
    if (startEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
        startEvent_ = nullptr;
    }
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
        stopEvent_ = nullptr;
    }
}

std::string BestCourierDevice::uploadEta(const std::int32_t* etaMs, std::size_t orderCount, std::size_t courierCount,
                                         double* elapsedMs) {
    if (orderCount == 0 || courierCount == 0 || orderCount > std::numeric_limits<std::size_t>::max() / courierCount) {
        return "BestCourierDevice::uploadEta: tamano fuera de rango";
    }
    std::string errorMessage = createEvents(&startEvent_, &stopEvent_);
    if (errorMessage.empty()) {
        errorMessage = ensure(&etaPointer_, etaCapacity_, orderCount * courierCount, sizeof(std::int32_t), "la matriz de ETA");
    }
    if (!errorMessage.empty()) return errorMessage;
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(etaPointer_, etaMs, orderCount * courierCount * sizeof(std::int32_t),
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy matriz H2D");
    });
}

std::string BestCourierDevice::uploadCourierIds(const std::int32_t* courierIds, std::size_t courierCount,
                                                double* elapsedMs) {
    if (courierCount == 0 || courierCount > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return "BestCourierDevice::uploadCourierIds: tamano fuera de rango";
    }
    std::string errorMessage = createEvents(&startEvent_, &stopEvent_);
    if (errorMessage.empty()) errorMessage = ensure(&idsPointer_, idsCapacity_, courierCount, sizeof(std::int32_t), "los ids");
    if (!errorMessage.empty()) return errorMessage;
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(idsPointer_, courierIds, courierCount * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                             "cudaMemcpy ids H2D");
    });
}

std::string BestCourierDevice::run(const void* externalEta, std::size_t orderCount, std::size_t courierCount,
                                   bool divergent, double* elapsedMs) {
    const void* eta = externalEta != nullptr ? externalEta : etaPointer_;
    if (eta == nullptr || idsPointer_ == nullptr || orderCount == 0 || courierCount == 0) {
        return "BestCourierDevice::run: faltan la matriz o los ids";
    }
    if (courierCount > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        orderCount > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return "BestCourierDevice::run: tamano fuera de rango";
    }
    std::string errorMessage = createEvents(&startEvent_, &stopEvent_);
    if (errorMessage.empty()) {
        errorMessage = ensure(&keysPointer_, keysCapacity_, orderCount, sizeof(unsigned long long), "las claves");
    }
    if (!errorMessage.empty()) return errorMessage;
    orderCount_ = orderCount;
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        const std::int32_t* etaMs = static_cast<const std::int32_t*>(eta);
        const std::int32_t* ids = static_cast<const std::int32_t*>(idsPointer_);
        unsigned long long* keys = static_cast<unsigned long long*>(keysPointer_);
        if (divergent) {
            bestCourierDivergent<kBlockSize><<<static_cast<unsigned>(orderCount), kBlockSize>>>(
                etaMs, ids, static_cast<int>(courierCount), keys);
        } else {
            bestCourierConvergent<kBlockSize><<<static_cast<unsigned>(orderCount), kBlockSize>>>(
                etaMs, ids, static_cast<int>(courierCount), keys);
        }
        return describeError(cudaGetLastError(), "lanzamiento del kernel de argmin");
    });
}

std::string BestCourierDevice::download(std::vector<unsigned long long>& keys, double* elapsedMs) {
    if (keysPointer_ == nullptr || orderCount_ == 0) return "BestCourierDevice::download: sin resultado";
    keys.resize(orderCount_);
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(keys.data(), keysPointer_, orderCount_ * sizeof(unsigned long long),
                                        cudaMemcpyDeviceToHost),
                             "cudaMemcpy claves D2H");
    });
}

}  // namespace accel
