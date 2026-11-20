#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "accel/cuda/Scan.cuh"

namespace accel {

namespace {

// Convierte un cudaError_t en mensaje ("" si no hubo error) y limpia el error pegajoso.
std::string check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return "";
    }
    const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
    cudaGetLastError();
    return message;
}

std::size_t ceilDivide(std::size_t numerator, std::size_t denominator) {
    return (numerator + denominator - 1) / denominator;
}

// Paso 1: scan inclusivo de cada tramo de blockDim.x elementos (Kogge-Stone con doble buffer).
__global__ void scanBlockKoggeStone(const std::int32_t* input, std::int32_t* output, std::int32_t* blockTotal,
                                    std::size_t count) {
    extern __shared__ std::int32_t scanBuffers[];  // 2 x blockDim.x enteros
    const unsigned blockSize = blockDim.x;
    const unsigned positionInBlock = threadIdx.x;
    std::int32_t* readBuffer = scanBuffers;
    std::int32_t* writeBuffer = scanBuffers + blockSize;
    const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockSize + positionInBlock;
    readBuffer[positionInBlock] = (element < count) ? input[element] : 0;  // relleno: 0 no cambia la suma
    __syncthreads();

    for (unsigned distance = 1; distance < blockSize; distance *= 2) {
        std::int32_t value = readBuffer[positionInBlock];
        if (positionInBlock >= distance) {
            value += readBuffer[positionInBlock - distance];  // el que esta `distance` lugares atras
        }
        writeBuffer[positionInBlock] = value;  // se escribe en el OTRO buffer
        __syncthreads();                       // todos escribieron antes del paso siguiente
        std::int32_t* swapBuffer = readBuffer;
        readBuffer = writeBuffer;
        writeBuffer = swapBuffer;
    }

    if (element < count) {
        output[element] = readBuffer[positionInBlock];
    }
    if (positionInBlock == blockSize - 1) {
        blockTotal[blockIdx.x] = readBuffer[positionInBlock];  // total del tramo (el relleno suma 0)
    }
}

// Paso 3: a cada tramo se le suma el total acumulado de todos los tramos anteriores.
__global__ void addPreviousBlocksTotal(std::int32_t* output, const std::int32_t* accumulatedBlockTotal,
                                       std::size_t count) {
    const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && element < count) {
        output[element] += accumulatedBlockTotal[blockIdx.x - 1];
    }
}
}  // namespace

// Enteros auxiliares para el scan de `count` elementos: por nivel, los totales de tramo y su scan.
std::size_t scanScratchCount(std::size_t count) {
    const std::size_t blockCount = ceilDivide(count, kScanBlockSize);
    return blockCount <= 1 ? 1 : 2 * blockCount + scanScratchCount(blockCount);
}

// Scan inclusivo jerarquico de `count` enteros que ya estan en la GPU (cualquier largo).
std::string scanInclusiveOnDevice(const std::int32_t* input, std::int32_t* output, std::size_t count,
                                  std::int32_t* scratch) {
    if (count == 0) {
        return "";
    }
    const std::size_t blockCount = ceilDivide(count, kScanBlockSize);
    std::int32_t* blockTotal = scratch;
    const std::size_t sharedBytes = 2 * kScanBlockSize * sizeof(std::int32_t);
    scanBlockKoggeStone<<<static_cast<unsigned>(blockCount), kScanBlockSize, sharedBytes>>>(input, output, blockTotal, count);  // paso 1
    std::string errorMessage = check(cudaGetLastError(), "lanzamiento de scanBlockKoggeStone");
    if (errorMessage.empty() && blockCount > 1) {
        std::int32_t* accumulatedBlockTotal = scratch + blockCount;
        errorMessage = scanInclusiveOnDevice(blockTotal, accumulatedBlockTotal, blockCount, scratch + 2 * blockCount);  // paso 2
        if (errorMessage.empty()) {
            addPreviousBlocksTotal<<<static_cast<unsigned>(blockCount), kScanBlockSize>>>(output, accumulatedBlockTotal, count);  // paso 3
            errorMessage = check(cudaGetLastError(), "lanzamiento de addPreviousBlocksTotal");
        }
    }
    return errorMessage;
}

}  // namespace accel
