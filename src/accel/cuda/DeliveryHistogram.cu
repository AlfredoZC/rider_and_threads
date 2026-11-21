#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

#include "accel/DeliveryHistogram.h"

namespace accel {

namespace {

constexpr int kBlockSize = 256;

// Convierte un cudaError_t en mensaje ("" si no hubo error) y limpia el error pegajoso.
std::string check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return "";
    }
    const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
    cudaGetLastError();
    return message;
}

// Version ingenua: un hilo por entrega y un atomico en memoria GLOBAL por entrega.
__global__ void histogramGlobalAtomicsKernel(const std::int32_t* deliveryTimesMs, std::size_t deliveryCount,
                                             unsigned* globalBins) {
    const std::size_t deliveryIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (deliveryIndex < deliveryCount) {
        atomicAdd(&globalBins[deliveryMinuteBin(deliveryTimesMs[deliveryIndex])], 1u);
    }
}

// Version privatizada: cada bloque cuenta en su propia copia del histograma en memoria compartida
// y al final la suma al global. Bucle grid-stride: hilos vecinos leen entregas vecinas.
__global__ void histogramPrivatizedKernel(const std::int32_t* deliveryTimesMs, std::size_t deliveryCount,
                                          unsigned* globalBins) {
    __shared__ unsigned blockBins[kDeliveryHistogramBins];

    // 1) Copia privada en cero (los hilos se reparten las barras).
    for (int binIndex = threadIdx.x; binIndex < kDeliveryHistogramBins; binIndex += blockDim.x) {
        blockBins[binIndex] = 0;
    }
    __syncthreads();  // nadie suma antes de que todo este en cero

    // 2) Contar con atomicos en memoria compartida (mucho mas baratos que los globales).
    const std::size_t totalThreads = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t deliveryIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         deliveryIndex < deliveryCount; deliveryIndex += totalThreads) {
        atomicAdd(&blockBins[deliveryMinuteBin(deliveryTimesMs[deliveryIndex])], 1u);
    }
    __syncthreads();  // el bloque termino de contar

    // 3) Volcar: un atomico global por barra no vacia, no uno por entrega.
    for (int binIndex = threadIdx.x; binIndex < kDeliveryHistogramBins; binIndex += blockDim.x) {
        if (blockBins[binIndex] > 0) {
            atomicAdd(&globalBins[binIndex], blockBins[binIndex]);
        }
    }
}

// Bloques para llenar la GPU con el kernel privatizado (ocupacion real x multiprocesadores).
std::string privatizedBlockCap(int* blockCap) {
    int device = 0;
    int multiprocessorCount = 0;
    int blocksPerMultiprocessor = 0;
    std::string errorMessage = check(cudaGetDevice(&device), "cudaGetDevice");
    if (errorMessage.empty()) {
        errorMessage = check(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device),
                             "cudaDeviceGetAttribute");
    }
    if (errorMessage.empty()) {
        errorMessage = check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMultiprocessor,
                                                                           histogramPrivatizedKernel, kBlockSize, 0),
                             "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    }
    if (errorMessage.empty()) {
        *blockCap = std::max(1, multiprocessorCount * blocksPerMultiprocessor);
    }
    return errorMessage;
}

}  // namespace

DeviceHistogram::~DeviceHistogram() { release(); }

void DeviceHistogram::release() {
    if (startEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
        startEvent_ = nullptr;
    }
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
        stopEvent_ = nullptr;
    }
    if (deviceTimes_ != nullptr) {
        cudaFree(deviceTimes_);
        deviceTimes_ = nullptr;
    }
    if (deviceBins_ != nullptr) {
        cudaFree(deviceBins_);
        deviceBins_ = nullptr;
    }
    timeCapacity_ = 0;
    uploadedCount_ = 0;
    cudaGetLastError();
}

std::string DeviceHistogram::allocate(std::size_t timeCapacity) {
    release();
    if (timeCapacity > std::size_t{1} << 31) {
        return "DeviceHistogram: demasiados tiempos";
    }
    const std::size_t timeBytes = std::max<std::size_t>(timeCapacity, 1) * sizeof(std::int32_t);
    std::string errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceTimes_), timeBytes),
                                     ("cudaMalloc de " + std::to_string(timeBytes) + " bytes").c_str());
    if (errorMessage.empty()) {
        errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceBins_),
                                        static_cast<std::size_t>(kDeliveryHistogramBins) * sizeof(unsigned)),
                             "cudaMalloc de las barras");
    }
    cudaEvent_t startCreated = nullptr;
    cudaEvent_t stopCreated = nullptr;
    if (errorMessage.empty()) errorMessage = check(cudaEventCreate(&startCreated), "cudaEventCreate");
    if (errorMessage.empty()) {
        startEvent_ = startCreated;
        errorMessage = check(cudaEventCreate(&stopCreated), "cudaEventCreate");
    }
    if (errorMessage.empty()) {
        stopEvent_ = stopCreated;
        timeCapacity_ = timeCapacity;
        return "";
    }
    release();
    return errorMessage;
}

std::string DeviceHistogram::upload(const std::int32_t* hostTimesMs, std::size_t count, double* elapsedMs) {
    if (deviceTimes_ == nullptr || count > timeCapacity_) {
        return "upload: memoria no reservada o demasiado chica";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && count > 0) {
        errorMessage = check(cudaMemcpy(deviceTimes_, hostTimesMs, count * sizeof(std::int32_t), cudaMemcpyHostToDevice),
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

std::string DeviceHistogram::run(HistogramKernel kernel, double* elapsedMs) {
    if (deviceTimes_ == nullptr || deviceBins_ == nullptr) {
        return "run: memoria no reservada";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    int blockCount = 1;
    const std::size_t neededBlocks = (uploadedCount_ + kBlockSize - 1) / kBlockSize;
    std::string errorMessage = check(
        cudaMemset(deviceBins_, 0, static_cast<std::size_t>(kDeliveryHistogramBins) * sizeof(unsigned)), "cudaMemset");
    if (errorMessage.empty() && kernel == HistogramKernel::privatized) {
        int blockCap = 1;
        errorMessage = privatizedBlockCap(&blockCap);
        blockCount = static_cast<int>(std::min<std::size_t>(std::max<std::size_t>(neededBlocks, 1), blockCap));
    } else {
        blockCount = static_cast<int>(std::max<std::size_t>(neededBlocks, 1));
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && uploadedCount_ > 0) {
        if (kernel == HistogramKernel::privatized) {
            histogramPrivatizedKernel<<<blockCount, kBlockSize>>>(deviceTimes_, uploadedCount_, deviceBins_);
        } else {
            histogramGlobalAtomicsKernel<<<blockCount, kBlockSize>>>(deviceTimes_, uploadedCount_, deviceBins_);
        }
        errorMessage = check(cudaGetLastError(), "lanzamiento del kernel del histograma");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

std::string DeviceHistogram::download(unsigned* hostBins, double* elapsedMs) {
    if (deviceBins_ == nullptr) {
        return "download: memoria no reservada";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) {
        errorMessage = check(cudaMemcpy(hostBins, deviceBins_, static_cast<std::size_t>(kDeliveryHistogramBins) * sizeof(unsigned),
                                        cudaMemcpyDeviceToHost),
                             "cudaMemcpy D2H");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

}  // namespace accel
