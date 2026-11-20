#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

#include "accel/Calib.h"

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

// Copia source -> destination en palabras de 16 B con un bucle grid-stride; los (bytes % 16) ultimos bytes los
// copia un solo hilo. cudaMalloc devuelve memoria alineada a 256 B, asi que uint4 es seguro.
__global__ void calib_copy_kernel(const uint4* sourceWords, uint4* destinationWords, std::size_t words,
                                  const unsigned char* sourceBytes,
                                  unsigned char* destinationBytes, std::size_t bytes) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t wordIndex = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; wordIndex < words; wordIndex += stride) {
        destinationWords[wordIndex] = sourceWords[wordIndex];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (std::size_t byteIndex = words * 16; byteIndex < bytes; ++byteIndex) {
            destinationBytes[byteIndex] = sourceBytes[byteIndex];
        }
    }
}

// Un solo hilo suma 1: el "kernel vacio" cuyo costo es el del lanzamiento. El contador permite
// comprobar que los N lanzamientos se ejecutaron todos.
__global__ void calib_launch_kernel(unsigned long long* counter) { *counter += 1; }

}  // namespace

DeviceBuffer::~DeviceBuffer() { release(); }

void DeviceBuffer::release() {
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
    bytes_ = 0;
    cudaGetLastError();
}

std::string DeviceBuffer::allocate(std::size_t bytes) {
    release();
    if (bytes == 0) {
        return "DeviceBuffer: tamano 0";
    }
    std::string errorMessage = check(cudaMalloc(&devicePointer_, bytes), ("cudaMalloc de " + std::to_string(bytes) + " bytes").c_str());
    if (!errorMessage.empty()) {
        devicePointer_ = nullptr;
        return errorMessage;
    }
    bytes_ = bytes;
    cudaEvent_t startCreated = nullptr;
    cudaEvent_t stopCreated = nullptr;
    errorMessage = check(cudaEventCreate(&startCreated), "cudaEventCreate");
    if (errorMessage.empty()) {
        startEvent_ = startCreated;
        errorMessage = check(cudaEventCreate(&stopCreated), "cudaEventCreate");
    }
    if (errorMessage.empty()) {
        stopEvent_ = stopCreated;
        return "";
    }
    release();
    return errorMessage;
}

std::string DeviceBuffer::upload(const void* host, std::size_t bytes, double* elapsedMs) {
    if (devicePointer_ == nullptr || bytes > bytes_) {
        return "upload: buffer no reservado o demasiado chico";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaMemcpy(devicePointer_, host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

std::string DeviceBuffer::download(void* host, std::size_t bytes, double* elapsedMs) {
    if (devicePointer_ == nullptr || bytes > bytes_) {
        return "download: buffer no reservado o demasiado chico";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaMemcpy(host, devicePointer_, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

std::string DeviceBuffer::copyTo(DeviceBuffer& destination, double* elapsedMs) {
    if (devicePointer_ == nullptr || destination.devicePointer_ == nullptr || destination.bytes_ < bytes_) {
        return "copyTo: buffers no reservados o destino demasiado chico";
    }
    const std::size_t words = bytes_ / 16;
    const std::size_t wanted = (std::max<std::size_t>(words, 1) + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>(wanted, 65535));
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) {
        calib_copy_kernel<<<blocks, 256>>>(static_cast<const uint4*>(devicePointer_), static_cast<uint4*>(destination.devicePointer_), words,
                                           static_cast<const unsigned char*>(devicePointer_),
                                           static_cast<unsigned char*>(destination.devicePointer_), bytes_);
        errorMessage = check(cudaGetLastError(), "lanzamiento de calib_copy_kernel");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

std::string DeviceBuffer::clear() {
    if (devicePointer_ == nullptr) {
        return "clear: buffer no reservado";
    }
    return check(cudaMemset(devicePointer_, 0, bytes_), "cudaMemset");
}

std::string deviceSynchronize() { return check(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

std::string gpuMemory(std::size_t* freeBytes, std::size_t* totalBytes) {
    std::size_t freeMemory = 0;
    std::size_t totalMemory = 0;
    const std::string errorMessage = check(cudaMemGetInfo(&freeMemory, &totalMemory), "cudaMemGetInfo");
    if (errorMessage.empty()) {
        *freeBytes = freeMemory;
        *totalBytes = totalMemory;
    }
    return errorMessage;
}

std::string launchCounterKernel(DeviceBuffer& counter, std::size_t count, bool syncEach, double* eventMs) {
    if (counter.devicePointer_ == nullptr || counter.bytes_ < sizeof(unsigned long long)) {
        return "launchCounterKernel: contador no reservado";
    }
    auto* counterOnDevice = static_cast<unsigned long long*>(counter.devicePointer_);
    const cudaEvent_t start = static_cast<cudaEvent_t>(counter.startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(counter.stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    for (std::size_t launchIndex = 0; errorMessage.empty() && launchIndex < count; ++launchIndex) {
        calib_launch_kernel<<<1, 1>>>(counterOnDevice);
        errorMessage = check(cudaGetLastError(), "lanzamiento de calib_launch_kernel");
        if (errorMessage.empty() && syncEach) errorMessage = check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && eventMs != nullptr) *eventMs = elapsed;
    return errorMessage;
}

}  // namespace accel
