// Floyd-Warshall por tiles (aplicacion 03). Solo kernels y envoltorios: el relleno, la
// concurrencia y la tabla viven en accel/TravelTimeTable.cpp.
//
// Por cada tile pivote hay tres lanzamientos en el mismo stream (el fin de uno es la barrera
// global del siguiente):
//   fase 1  el tile pivote, solo (un bloque)
//   fase 2  los tiles de la fila y de la columna del pivote
//   fase 3  todos los demas tiles (el producto de matrices con (min, +))
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "accel/TravelTimeTable.h"

namespace accel {

namespace {

__host__ __device__ inline std::size_t cellOf(int fromNode, int toNode, int nodeCount) {
    return static_cast<std::size_t>(fromNode) * static_cast<std::size_t>(nodeCount) + static_cast<std::size_t>(toNode);
}

template <int TILE_SIZE>
__global__ void floydWarshallPhase1(std::int32_t* travelTimeMs, int nodeCount, int pivotTile) {
    __shared__ std::int32_t pivotBlock[TILE_SIZE][TILE_SIZE];
    const int row = threadIdx.y;
    const int column = threadIdx.x;
    const int pivotBase = pivotTile * TILE_SIZE;
    const std::size_t cell = cellOf(pivotBase + row, pivotBase + column, nodeCount);

    pivotBlock[row][column] = travelTimeMs[cell];
    __syncthreads();
    for (int viaNode = 0; viaNode < TILE_SIZE; ++viaNode) {
        const std::int32_t timeThroughVia = pivotBlock[row][viaNode] + pivotBlock[viaNode][column];
        if (timeThroughVia < pivotBlock[row][column]) {
            pivotBlock[row][column] = timeThroughVia;  // nunca escribe la fila ni la columna viaNode
        }
        __syncthreads();
    }
    travelTimeMs[cell] = pivotBlock[row][column];
}

template <int TILE_SIZE>
__global__ void floydWarshallPhase2(std::int32_t* travelTimeMs, int nodeCount, int pivotTile) {
    const int otherTile = blockIdx.x;
    if (otherTile == pivotTile) {
        return;
    }
    const bool isPivotRow = (blockIdx.y == 0);
    __shared__ std::int32_t pivotBlock[TILE_SIZE][TILE_SIZE];
    __shared__ std::int32_t ownBlock[TILE_SIZE][TILE_SIZE];
    const int row = threadIdx.y;
    const int column = threadIdx.x;
    const int pivotBase = pivotTile * TILE_SIZE;
    const int otherBase = otherTile * TILE_SIZE;
    const int fromNode = isPivotRow ? pivotBase + row : otherBase + row;
    const int toNode = isPivotRow ? otherBase + column : pivotBase + column;
    const std::size_t cell = cellOf(fromNode, toNode, nodeCount);

    pivotBlock[row][column] = travelTimeMs[cellOf(pivotBase + row, pivotBase + column, nodeCount)];
    ownBlock[row][column] = travelTimeMs[cell];
    __syncthreads();
    for (int viaNode = 0; viaNode < TILE_SIZE; ++viaNode) {
        const std::int32_t timeThroughVia = isPivotRow ? pivotBlock[row][viaNode] + ownBlock[viaNode][column]
                                                       : ownBlock[row][viaNode] + pivotBlock[viaNode][column];
        if (timeThroughVia < ownBlock[row][column]) {
            ownBlock[row][column] = timeThroughVia;
        }
        __syncthreads();
    }
    travelTimeMs[cell] = ownBlock[row][column];
}

template <int TILE_SIZE>
__global__ void floydWarshallPhase3(std::int32_t* travelTimeMs, int nodeCount, int pivotTile) {
    const int tileRow = blockIdx.y;
    const int tileColumn = blockIdx.x;
    if (tileRow == pivotTile || tileColumn == pivotTile) {
        return;
    }
    __shared__ std::int32_t pivotColumnTile[TILE_SIZE][TILE_SIZE];
    __shared__ std::int32_t pivotRowTile[TILE_SIZE][TILE_SIZE];
    const int row = threadIdx.y;
    const int column = threadIdx.x;
    const int fromNode = tileRow * TILE_SIZE + row;
    const int toNode = tileColumn * TILE_SIZE + column;
    const int pivotBase = pivotTile * TILE_SIZE;

    pivotColumnTile[row][column] = travelTimeMs[cellOf(fromNode, pivotBase + column, nodeCount)];
    pivotRowTile[row][column] = travelTimeMs[cellOf(pivotBase + row, toNode, nodeCount)];
    __syncthreads();

    const std::size_t cell = cellOf(fromNode, toNode, nodeCount);
    std::int32_t bestTime = travelTimeMs[cell];
    for (int viaNode = 0; viaNode < TILE_SIZE; ++viaNode) {
        bestTime = min(bestTime, pivotColumnTile[row][viaNode] + pivotRowTile[viaNode][column]);
    }
    travelTimeMs[cell] = bestTime;
}

// Un error de ejecucion se devuelve SIN limpiar el error pegajoso (lo ve throwIfGpuError).
std::string describeError(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return "";
    }
    return std::string(operation) + ": " + cudaGetErrorString(status);
}

template <int TILE_SIZE>
std::string launchAllPivots(std::int32_t* devicePointer, int paddedCount) {
    const int tileCount = paddedCount / TILE_SIZE;
    const dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    for (int pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        floydWarshallPhase1<TILE_SIZE><<<1, threadsPerBlock>>>(devicePointer, paddedCount, pivotTile);
        std::string errorMessage = describeError(cudaGetLastError(), "lanzamiento de floydWarshallPhase1");
        if (!errorMessage.empty()) return errorMessage;
        floydWarshallPhase2<TILE_SIZE><<<dim3(tileCount, 2), threadsPerBlock>>>(devicePointer, paddedCount, pivotTile);
        errorMessage = describeError(cudaGetLastError(), "lanzamiento de floydWarshallPhase2");
        if (!errorMessage.empty()) return errorMessage;
        floydWarshallPhase3<TILE_SIZE><<<dim3(tileCount, tileCount), threadsPerBlock>>>(devicePointer, paddedCount,
                                                                                         pivotTile);
        errorMessage = describeError(cudaGetLastError(), "lanzamiento de floydWarshallPhase3");
        if (!errorMessage.empty()) return errorMessage;
    }
    return "";
}

}  // namespace

ApspDevice::~ApspDevice() { release(); }

void ApspDevice::release() {
    if (startEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
        startEvent_ = nullptr;
    }
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
        stopEvent_ = nullptr;
    }
    if (devicePointer_ != nullptr) {
        cudaFree(devicePointer_);
        devicePointer_ = nullptr;
    }
    paddedCount_ = 0;
}

std::string ApspDevice::allocate(std::size_t paddedCount) {
    release();
    if (paddedCount == 0 || paddedCount > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        paddedCount > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) / paddedCount) {
        return "ApspDevice: tamano fuera de rango";
    }
    const std::size_t bytes = paddedCount * paddedCount * sizeof(std::int32_t);
    cudaError_t status = cudaMalloc(&devicePointer_, bytes);
    if (status != cudaSuccess) {  // sin memoria: recuperable, se limpia
        const std::string message = std::string("cudaMalloc de ") + std::to_string(bytes) + " bytes: " + cudaGetErrorString(status);
        devicePointer_ = nullptr;
        cudaGetLastError();
        return message;
    }
    paddedCount_ = paddedCount;
    cudaEvent_t startCreated = nullptr;
    cudaEvent_t stopCreated = nullptr;
    status = cudaEventCreate(&startCreated);
    if (status == cudaSuccess) {
        startEvent_ = startCreated;
        status = cudaEventCreate(&stopCreated);
    }
    if (status == cudaSuccess) {
        stopEvent_ = stopCreated;
        return "";
    }
    const std::string message = describeError(status, "cudaEventCreate");
    release();
    return message;
}

namespace {
// Corre `work` entre dos cudaEvent y devuelve el tiempo en ms (si `elapsedMs` no es nulo).
template <typename Work>
std::string timedRegion(void* startEvent, void* stopEvent, double* elapsedMs, Work work) {
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent);
    std::string errorMessage = describeError(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = work();
    if (errorMessage.empty()) errorMessage = describeError(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = describeError(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = describeError(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}
}  // namespace

std::string ApspDevice::upload(const std::int32_t* host, double* elapsedMs) {
    if (devicePointer_ == nullptr) return "ApspDevice::upload: sin reservar";
    const std::size_t bytes = paddedCount_ * paddedCount_ * sizeof(std::int32_t);
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(devicePointer_, host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    });
}

std::string ApspDevice::runTiled(int tileSize, double* elapsedMs) {
    if (devicePointer_ == nullptr) return "ApspDevice::runTiled: sin reservar";
    if ((tileSize != 16 && tileSize != 32) || paddedCount_ % static_cast<std::size_t>(tileSize) != 0) {
        return "ApspDevice::runTiled: tile invalido (16 o 32) o tabla no multiplo del tile";
    }
    std::int32_t* devicePointer = static_cast<std::int32_t*>(devicePointer_);
    const int paddedCount = static_cast<int>(paddedCount_);
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return tileSize == 32 ? launchAllPivots<32>(devicePointer, paddedCount)
                              : launchAllPivots<16>(devicePointer, paddedCount);
    });
}

std::string ApspDevice::download(std::int32_t* host, double* elapsedMs) {
    if (devicePointer_ == nullptr) return "ApspDevice::download: sin reservar";
    const std::size_t bytes = paddedCount_ * paddedCount_ * sizeof(std::int32_t);
    return timedRegion(startEvent_, stopEvent_, elapsedMs, [&] {
        return describeError(cudaMemcpy(host, devicePointer_, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    });
}

}  // namespace accel
