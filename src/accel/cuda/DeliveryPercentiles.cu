#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "accel/DeliveryPercentiles.h"
#include "accel/cuda/Scan.cuh"

namespace accel {

namespace {

constexpr int kRadixBlockSize = 256;
constexpr std::size_t kMaxKeys = std::size_t{1} << 31;

// Convierte un cudaError_t en mensaje ("" si no hubo error) y limpia el error pegajoso.
std::string check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return "";
    }
    const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
    cudaGetLastError();
    return message;
}

// Marcar: isZero[i] = 1 si la clave i tiene este bit en 0.
__global__ void markZeroBit(const std::uint32_t* keys, std::size_t count, int bit, std::int32_t* isZero) {
    const std::size_t position = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position < count) {
        isZero[position] = ((keys[position] >> bit) & 1u) == 0 ? 1 : 0;
    }
}

// Reubicar: con el scan de isZero cada hilo sabe, sin mirar a nadie mas, a donde va su clave.
// zerosUpTo es el scan INCLUSIVO; el EXCLUSIVO sale restando el propio valor.
__global__ void radixScatter(const std::uint32_t* keysIn, std::uint32_t* keysOut, const std::int32_t* isZero,
                             const std::int32_t* zerosUpTo, std::size_t count) {
    const std::size_t position = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (position >= count) {
        return;
    }
    const std::size_t totalZeros = static_cast<std::size_t>(zerosUpTo[count - 1]);       // todos leen la misma celda
    const std::size_t zerosBefore = static_cast<std::size_t>(zerosUpTo[position] - isZero[position]);  // scan exclusivo
    const std::size_t onesBefore = position - zerosBefore;
    const std::size_t destination = isZero[position] ? zerosBefore              // ceros: adelante, en su orden
                                                     : totalZeros + onesBefore;  // unos: despues de todos los ceros
    keysOut[destination] = keysIn[position];
}

}  // namespace

DeviceRadixSort::~DeviceRadixSort() { release(); }

void DeviceRadixSort::release() {
    if (startEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
        startEvent_ = nullptr;
    }
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
        stopEvent_ = nullptr;
    }
    cudaFree(deviceKeys_[0]);
    cudaFree(deviceKeys_[1]);
    cudaFree(deviceKeys_[2]);
    cudaFree(deviceIsZero_);
    cudaFree(deviceZerosUpTo_);
    cudaFree(deviceScratch_);
    deviceKeys_[0] = nullptr;
    deviceKeys_[1] = nullptr;
    deviceKeys_[2] = nullptr;
    deviceIsZero_ = nullptr;
    deviceZerosUpTo_ = nullptr;
    deviceScratch_ = nullptr;
    keyCapacity_ = 0;
    uploadedCount_ = 0;
    bitCount_ = 0;
    sortedBuffer_ = 0;
    cudaGetLastError();
}

std::string DeviceRadixSort::allocate(std::size_t keyCapacity) {
    release();
    if (keyCapacity > kMaxKeys) {
        return "DeviceRadixSort: demasiadas claves";
    }
    const std::size_t keyBytes = std::max<std::size_t>(keyCapacity, 1) * sizeof(std::uint32_t);
    const std::size_t scratchBytes = scanScratchCount(keyCapacity) * sizeof(std::int32_t);
    std::string errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceKeys_[0]), keyBytes),
                                     ("cudaMalloc de " + std::to_string(keyBytes) + " bytes").c_str());
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceKeys_[1]), keyBytes), "cudaMalloc de las claves 2");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceKeys_[2]), keyBytes), "cudaMalloc de las claves 3");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceIsZero_), keyBytes), "cudaMalloc de las marcas");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceZerosUpTo_), keyBytes), "cudaMalloc del scan");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceScratch_), scratchBytes), "cudaMalloc del auxiliar");
    cudaEvent_t startCreated = nullptr;
    cudaEvent_t stopCreated = nullptr;
    if (errorMessage.empty()) errorMessage = check(cudaEventCreate(&startCreated), "cudaEventCreate");
    if (errorMessage.empty()) {
        startEvent_ = startCreated;
        errorMessage = check(cudaEventCreate(&stopCreated), "cudaEventCreate");
    }
    if (errorMessage.empty()) {
        stopEvent_ = stopCreated;
        keyCapacity_ = keyCapacity;
        return "";
    }
    release();
    return errorMessage;
}

std::string DeviceRadixSort::upload(const std::uint32_t* hostKeys, std::size_t count, double* elapsedMs) {
    if (deviceKeys_[0] == nullptr || count > keyCapacity_) {
        return "upload: memoria no reservada o demasiado chica";
    }
    const int bitCount = count == 0 ? 0 : significantBits(*std::max_element(hostKeys, hostKeys + count));
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && count > 0) {
        errorMessage = check(cudaMemcpy(deviceKeys_[0], hostKeys, count * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
                             "cudaMemcpy H2D");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty()) {
        uploadedCount_ = count;
        bitCount_ = bitCount;
        sortedBuffer_ = 0;
        if (elapsedMs != nullptr) *elapsedMs = elapsed;
    }
    return errorMessage;
}

std::string DeviceRadixSort::run(double* elapsedMs) {
    if (deviceKeys_[0] == nullptr) {
        return "run: memoria no reservada";
    }
    // Cada corrida parte de las claves originales (buffer 0, intacto): la primera pasada las lee y escribe en
    // el buffer 1; las siguientes alternan 1 y 2. Asi repetir run() mide siempre el mismo trabajo.
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    std::uint32_t* keys = deviceKeys_[0];
    std::uint32_t* otherKeys = deviceKeys_[1];
    int sortedBuffer = 0;
    const std::size_t blockCount = (uploadedCount_ + kRadixBlockSize - 1) / kRadixBlockSize;
    for (int bit = 0; errorMessage.empty() && uploadedCount_ > 0 && bit < bitCount_; ++bit) {
        markZeroBit<<<static_cast<unsigned>(blockCount), kRadixBlockSize>>>(keys, uploadedCount_, bit, deviceIsZero_);
        errorMessage = check(cudaGetLastError(), "lanzamiento de markZeroBit");
        if (errorMessage.empty()) errorMessage = scanInclusiveOnDevice(deviceIsZero_, deviceZerosUpTo_, uploadedCount_, deviceScratch_);
        if (errorMessage.empty()) {
            radixScatter<<<static_cast<unsigned>(blockCount), kRadixBlockSize>>>(keys, otherKeys, deviceIsZero_,
                                                                                deviceZerosUpTo_, uploadedCount_);
            errorMessage = check(cudaGetLastError(), "lanzamiento de radixScatter");
        }
        sortedBuffer = bit % 2 == 0 ? 1 : 2;  // la pasada 0 escribio en el buffer 1, la 1 en el 2, ...
        keys = otherKeys;                     // la salida de esta pasada es la entrada de la siguiente
        otherKeys = deviceKeys_[bit % 2 == 0 ? 2 : 1];
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty()) {
        sortedBuffer_ = sortedBuffer;
        if (elapsedMs != nullptr) *elapsedMs = elapsed;
    }
    return errorMessage;
}

std::string DeviceRadixSort::download(std::uint32_t* hostSortedKeys, double* elapsedMs) {
    if (deviceKeys_[0] == nullptr) {
        return "download: memoria no reservada";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && uploadedCount_ > 0) {
        errorMessage = check(cudaMemcpy(hostSortedKeys, deviceKeys_[sortedBuffer_], uploadedCount_ * sizeof(std::uint32_t),
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
