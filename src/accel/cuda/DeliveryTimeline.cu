#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

#include "accel/DeliveryTimeline.h"

namespace accel {

namespace {

constexpr int kMergeBlockSize = 256;
constexpr int kOutputsPerThread = 4;  // medido en el estudio: entre 1 y 64 salidas por hilo, 4 da el menor tiempo
constexpr std::size_t kMaxRecords = std::size_t{1} << 30;

// Convierte un cudaError_t en mensaje ("" si no hubo error) y limpia el error pegajoso.
std::string check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return "";
    }
    const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
    cudaGetLastError();
    return message;
}

// Una ronda del arbol: cada hilo produce `outputsPerThread` posiciones consecutivas de la salida, sea cual
// sea el par al que pertenezcan. Un lanzamiento fusiona TODOS los pares.
__global__ void mergeRound(const DeliveryRecord* roundInput, DeliveryRecord* roundOutput, const int* listStart,
                           int listCount, int totalCount, int outputsPerThread) {
    const long long thread = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long firstOutput = thread * outputsPerThread;
    const int outputBegin = static_cast<int>(firstOutput < totalCount ? firstOutput : totalCount);
    const int outputEnd = minInt(outputBegin + outputsPerThread, totalCount);
    mergeRoundRange(roundInput, roundOutput, listStart, listCount, outputBegin, outputEnd);
}

// Un hilo por franja: donde empieza su franja y donde la siguiente; la resta es la cuenta. Sin atomicos.
__global__ void deliveriesPerBucket(const DeliveryRecord* timeline, int count, int bucketMs, int bucketCount,
                                    int* perBucket) {
    const int bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket < bucketCount) {
        perBucket[bucket] = firstAtOrAfter(timeline, count, static_cast<long long>(bucket + 1) * bucketMs) -
                            firstAtOrAfter(timeline, count, static_cast<long long>(bucket) * bucketMs);
    }
}

}  // namespace

DeviceTimeline::~DeviceTimeline() { release(); }

void DeviceTimeline::release() {
    if (startEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
        startEvent_ = nullptr;
    }
    if (stopEvent_ != nullptr) {
        cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
        stopEvent_ = nullptr;
    }
    for (DeliveryRecord*& buffer : deviceRecords_) {
        cudaFree(buffer);
        buffer = nullptr;
    }
    cudaFree(deviceListStarts_);
    cudaFree(devicePerBucket_);
    deviceListStarts_ = nullptr;
    devicePerBucket_ = nullptr;
    recordCapacity_ = 0;
    listStartCapacity_ = 0;
    bucketCapacity_ = 0;
    uploadedCount_ = 0;
    listCountPerRound_.clear();
    bucketCount_ = 0;
    sortedBuffer_ = nullptr;
    cudaGetLastError();
}

std::string DeviceTimeline::allocate(std::size_t recordCapacity, const TimelinePlan& plan) {
    release();
    if (recordCapacity > kMaxRecords) {
        return "DeviceTimeline: demasiados registros";
    }
    const std::size_t recordBytes = std::max<std::size_t>(recordCapacity, 1) * sizeof(DeliveryRecord);
    const std::size_t listStartBytes = std::max<std::size_t>(plan.allListStarts.size(), 1) * sizeof(int);
    const std::size_t bucketBytes = std::max<std::size_t>(static_cast<std::size_t>(plan.bucketCount), 1) * sizeof(int);
    std::string errorMessage;
    for (DeliveryRecord*& buffer : deviceRecords_) {
        if (errorMessage.empty()) {
            errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&buffer), recordBytes),
                                 ("cudaMalloc de " + std::to_string(recordBytes) + " bytes").c_str());
        }
    }
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&deviceListStarts_), listStartBytes), "cudaMalloc de los comienzos de lista");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(reinterpret_cast<void**>(&devicePerBucket_), bucketBytes), "cudaMalloc de las cuentas");
    cudaEvent_t startCreated = nullptr;
    cudaEvent_t stopCreated = nullptr;
    if (errorMessage.empty()) errorMessage = check(cudaEventCreate(&startCreated), "cudaEventCreate");
    if (errorMessage.empty()) {
        startEvent_ = startCreated;
        errorMessage = check(cudaEventCreate(&stopCreated), "cudaEventCreate");
    }
    if (errorMessage.empty()) {
        stopEvent_ = stopCreated;
        recordCapacity_ = recordCapacity;
        listStartCapacity_ = plan.allListStarts.size();
        bucketCapacity_ = static_cast<std::size_t>(plan.bucketCount);
        return "";
    }
    release();
    return errorMessage;
}

std::string DeviceTimeline::upload(const DeliveryRecord* hostRecords, std::size_t count, const TimelinePlan& plan,
                                   double* elapsedMs) {
    if (deviceRecords_[0] == nullptr || count > recordCapacity_ || plan.allListStarts.size() > listStartCapacity_ ||
        static_cast<std::size_t>(plan.bucketCount) > bucketCapacity_) {
        return "upload: memoria no reservada o demasiado chica";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && count > 0) {
        errorMessage = check(cudaMemcpy(deviceRecords_[0], hostRecords, count * sizeof(DeliveryRecord), cudaMemcpyHostToDevice),
                             "cudaMemcpy H2D de los registros");
    }
    if (errorMessage.empty() && !plan.allListStarts.empty()) {
        errorMessage = check(cudaMemcpy(deviceListStarts_, plan.allListStarts.data(), plan.allListStarts.size() * sizeof(int),
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy H2D de los comienzos de lista");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty()) {
        uploadedCount_ = count;
        listCountPerRound_ = plan.listCountPerRound;
        bucketMs_ = plan.bucketMs;
        bucketCount_ = plan.bucketCount;
        sortedBuffer_ = deviceRecords_[0];
        if (elapsedMs != nullptr) *elapsedMs = elapsed;
    }
    return errorMessage;
}

std::string DeviceTimeline::run(double* elapsedMs) {
    if (deviceRecords_[0] == nullptr) {
        return "run: memoria no reservada";
    }
    // Cada corrida parte de los registros originales (buffer 0, intacto): la primera ronda los lee y escribe en el
    // buffer 1; las siguientes alternan 1 y 2. Cada lanzamiento espera al anterior: es la barrera entre rondas.
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    DeliveryRecord* roundInput = deviceRecords_[0];
    DeliveryRecord* roundOutput = deviceRecords_[1];
    const int totalCount = static_cast<int>(uploadedCount_);
    const std::size_t outputThreads = (uploadedCount_ + kOutputsPerThread - 1) / kOutputsPerThread;
    const unsigned blockCount = static_cast<unsigned>((outputThreads + kMergeBlockSize - 1) / kMergeBlockSize);
    const int* listStart = deviceListStarts_;
    DeliveryRecord* sorted = deviceRecords_[0];
    for (std::size_t round = 0; errorMessage.empty() && uploadedCount_ > 0 && round < listCountPerRound_.size(); ++round) {
        const int listCount = listCountPerRound_[round];
        mergeRound<<<blockCount, kMergeBlockSize>>>(roundInput, roundOutput, listStart, listCount, totalCount, kOutputsPerThread);
        errorMessage = check(cudaGetLastError(), "lanzamiento de mergeRound");
        sorted = roundOutput;
        roundInput = roundOutput;
        roundOutput = deviceRecords_[round % 2 == 0 ? 2 : 1];  // la ronda 0 escribio en el 1, la 1 en el 2, ...
        listStart += listCount + 1;                            // los comienzos de lista de la ronda siguiente
    }
    if (errorMessage.empty() && uploadedCount_ > 0 && bucketCount_ > 0) {
        const unsigned bucketBlocks = static_cast<unsigned>((static_cast<std::size_t>(bucketCount_) + kMergeBlockSize - 1) / kMergeBlockSize);
        deliveriesPerBucket<<<bucketBlocks, kMergeBlockSize>>>(sorted, totalCount, bucketMs_, bucketCount_, devicePerBucket_);
        errorMessage = check(cudaGetLastError(), "lanzamiento de deliveriesPerBucket");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty()) {
        sortedBuffer_ = sorted;
        if (elapsedMs != nullptr) *elapsedMs = elapsed;
    }
    return errorMessage;
}

std::string DeviceTimeline::download(DeliveryRecord* hostTimeline, int* hostPerBucket, double* elapsedMs) {
    if (deviceRecords_[0] == nullptr || sortedBuffer_ == nullptr) {
        return "download: memoria no reservada";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty() && uploadedCount_ > 0) {
        errorMessage = check(cudaMemcpy(hostTimeline, sortedBuffer_, uploadedCount_ * sizeof(DeliveryRecord), cudaMemcpyDeviceToHost),
                             "cudaMemcpy D2H de la linea de tiempo");
    }
    if (errorMessage.empty() && uploadedCount_ > 0 && bucketCount_ > 0) {
        errorMessage = check(cudaMemcpy(hostPerBucket, devicePerBucket_, static_cast<std::size_t>(bucketCount_) * sizeof(int),
                                        cudaMemcpyDeviceToHost),
                             "cudaMemcpy D2H de las cuentas");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

}  // namespace accel
