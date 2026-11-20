#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

#include "accel/OpenOrders.h"

namespace accel {

namespace {

constexpr int kScanBlockSize = 512;  // hilos por bloque = elementos por tramo (medido mas rapido que 1024)
constexpr int kPeakBlockSize = 256;
constexpr std::size_t kMaxEvents = std::size_t{1} << 31;

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

// Reduccion max del pico: cada hilo recorre varios eventos, el bloque reduce en compartida y un solo
// hilo por bloque publica con un atomico. deviceKey es peakKey (OpenOrders.h) para el dispositivo.
__device__ unsigned long long deviceKey(std::int32_t openOrders, unsigned event) {
    const unsigned long long nonNegative = openOrders > 0 ? static_cast<unsigned long long>(openOrders) : 0ull;
    return (nonNegative << 32) | (0xFFFFFFFFu - event);
}

__global__ void peakOfRunningTotal(const std::int32_t* openOrders, std::size_t count, unsigned long long* globalPeakKey) {
    __shared__ unsigned long long partialPeak[kPeakBlockSize];
    const unsigned positionInBlock = threadIdx.x;
    const std::size_t totalThreads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    unsigned long long threadPeak = 0;  // menor que cualquier clave real
    for (std::size_t event = static_cast<std::size_t>(blockIdx.x) * blockDim.x + positionInBlock; event < count;
         event += totalThreads) {
        threadPeak = max(threadPeak, deviceKey(openOrders[event], static_cast<unsigned>(event)));
    }
    partialPeak[positionInBlock] = threadPeak;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {  // los activos quedan juntos
        if (positionInBlock < stride) {
            partialPeak[positionInBlock] = max(partialPeak[positionInBlock], partialPeak[positionInBlock + stride]);
        }
        __syncthreads();
    }
    if (positionInBlock == 0) {
        atomicMax(globalPeakKey, partialPeak[0]);  // un atomico por bloque
    }
}

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

}  // namespace

DeviceOpenOrders::~DeviceOpenOrders() { release(); }

void DeviceOpenOrders::release() {
    if (startEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
        startEvent_ = nullptr;
    }
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
        stopEvent_ = nullptr;
    }
    cudaFree(deviceChange_);
    cudaFree(deviceOpenOrders_);
    cudaFree(deviceScratch_);
    cudaFree(devicePeakKey_);
    deviceChange_ = nullptr;
    deviceOpenOrders_ = nullptr;
    deviceScratch_ = nullptr;
    devicePeakKey_ = nullptr;
    eventCapacity_ = 0;
    uploadedCount_ = 0;
    cudaGetLastError();
}

std::string DeviceOpenOrders::allocate(std::size_t eventCapacity) {
    release();
    if (eventCapacity > kMaxEvents) {
        return "DeviceOpenOrders: demasiados eventos";
    }
    const std::size_t eventBytes = std::max<std::size_t>(eventCapacity, 1) * sizeof(std::int32_t);
    const std::size_t scratchBytes = scanScratchCount(eventCapacity) * sizeof(std::int32_t);
    std::string errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceChange_), eventBytes),
                                     ("cudaMalloc de " + std::to_string(eventBytes) + " bytes").c_str());
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceOpenOrders_), eventBytes), "cudaMalloc de los abiertos");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceScratch_), scratchBytes), "cudaMalloc del auxiliar");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&devicePeakKey_), sizeof(unsigned long long)), "cudaMalloc del pico");
    cudaEvent_t startCreated = nullptr;
    cudaEvent_t stopCreated = nullptr;
    if (errorMessage.empty()) errorMessage = check(cudaEventCreate(&startCreated), "cudaEventCreate");
    if (errorMessage.empty()) {
        startEvent_ = startCreated;
        errorMessage = check(cudaEventCreate(&stopCreated), "cudaEventCreate");
    }
    if (errorMessage.empty()) {
        stopEvent_ = stopCreated;
        eventCapacity_ = eventCapacity;
        return "";
    }
    release();
    return errorMessage;
}

std::string DeviceOpenOrders::upload(const std::int32_t* hostChange, std::size_t count, double* elapsedMs) {
    if (deviceChange_ == nullptr || count > eventCapacity_) {
        return "upload: memoria no reservada o demasiado chica";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && count > 0) {
        errorMessage = check(cudaMemcpy(deviceChange_, hostChange, count * sizeof(std::int32_t), cudaMemcpyHostToDevice),
                             "cudaMemcpy H2D");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty()) {
        uploadedCount_ = count;
        if (elapsedMs != nullptr) *elapsedMs = elapsed;
    }
    return errorMessage;
}

std::string DeviceOpenOrders::run(double* elapsedMs) {
    if (deviceChange_ == nullptr) {
        return "run: memoria no reservada";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaMemset(devicePeakKey_, 0, sizeof(unsigned long long)), "cudaMemset");
    int device = 0;
    int multiprocessorCount = 0;
    int blocksPerMultiprocessor = 0;
    if (errorMessage.empty()) errorMessage = check(cudaGetDevice(&device), "cudaGetDevice");
    if (errorMessage.empty()) {
        errorMessage = check(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device),
                             "cudaDeviceGetAttribute");
    }
    if (errorMessage.empty()) {
        errorMessage = check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor, peakOfRunningTotal,
                                                                           kPeakBlockSize, 0),
                             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && uploadedCount_ > 0) {
        errorMessage = scanInclusiveOnDevice(deviceChange_, deviceOpenOrders_, uploadedCount_, deviceScratch_);
        if (errorMessage.empty()) {
            const std::size_t blockCap = static_cast<std::size_t>(std::max(1, multiprocessorCount * blocksPerMultiprocessor));
            const std::size_t peakBlocks = std::min(blockCap, ceilDivide(uploadedCount_, kPeakBlockSize));
            peakOfRunningTotal<<<static_cast<unsigned>(peakBlocks), kPeakBlockSize>>>(deviceOpenOrders_, uploadedCount_, devicePeakKey_);
            errorMessage = check(cudaGetLastError(), "lanzamiento de peakOfRunningTotal");
        }
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

std::string DeviceOpenOrders::download(std::int32_t* hostOpenOrders, unsigned long long* hostPeakKey, double* elapsedMs) {
    if (deviceOpenOrders_ == nullptr) {
        return "download: memoria no reservada";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && uploadedCount_ > 0) {
        errorMessage = check(cudaMemcpy(hostOpenOrders, deviceOpenOrders_, uploadedCount_ * sizeof(std::int32_t),
                                        cudaMemcpyDeviceToHost),
                             "cudaMemcpy D2H de los abiertos");
    }
    if (errorMessage.empty()) {
        errorMessage = check(cudaMemcpy(hostPeakKey, devicePeakKey_, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
                             "cudaMemcpy D2H del pico");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

}  // namespace accel
