// Percentiles p50, p90 y p99 del tiempo de entrega con radix sort de 1 bit por pasada.
//
//   cpu1        radix sort secuencial (una pasada por bit del maximo) + lectura de los percentiles
//   cpuN        radix sort con el pool propio: ceros por franja, scan de las franjas y reubicacion
//   gpu_kernel  radix sort en la GPU (marcar + scan + reubicar por pasada), con cudaEvent
//   gpu_e2e     subir las claves, ordenar y traer el arreglo ordenado (steady_clock + cudaEvent)
//
// size = cantidad de tiempos de entrega (misma distribucion sintetica que delivery_histogram). Correcto =
// el arreglo ordenado y los tres percentiles identicos a los de std::sort (referencia calculada en prepare).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <new>
#include <string>
#include <vector>

#include "accel/Calib.h"
#include "accel/DeliveryPercentiles.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"
#include "bench/DeliveryTimes.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

// Host: claves, referencia y copia de trabajo; GPU: 5 buffers de claves/marcas. Con 5e7 claves son 1 GB en la GPU.
constexpr std::size_t kMaxKeys = 50000000;

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

bool samePercentiles(const accel::DeliveryPercentiles& left, const accel::DeliveryPercentiles& right) {
    return left.p50Ms == right.p50Ms && left.p90Ms == right.p90Ms && left.p99Ms == right.p99Ms;
}

class DeliveryPercentilesBench : public BenchApp {
public:
    std::string name() const override { return "delivery_percentiles"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {1000, 10000, 100000, 1000000, 10000000};
    }

    std::size_t maxSize() const override { return kMaxKeys; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1, Version::CpuN};
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) override {
        releaseAll();
        if (size == 0 || size > kMaxKeys) {
            return "delivery_percentiles: tamano fuera de rango (1 .. " + std::to_string(kMaxKeys) + ")";
        }
        try {
            std::vector<std::int32_t> deliveryTimesMs(size, 0);
            fillDeliveryTimes(deliveryTimesMs, seed);
            keys_.assign(deliveryTimesMs.begin(), deliveryTimesMs.end());  // los tiempos generados son >= 0
            referenceSorted_ = keys_;
        } catch (const std::bad_alloc&) {
            releaseAll();
            return "delivery_percentiles: sin memoria en el host para " + std::to_string(size) + " tiempos";
        }
        std::sort(referenceSorted_.begin(), referenceSorted_.end());  // referencia independiente del radix
        referencePercentiles_ = accel::percentilesFromSorted(referenceSorted_);
        if (context.gpu.available) {
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            std::string errorMessage = accel::gpuMemory(&freeBytes, &totalBytes);
            const std::size_t neededBytes = size * sizeof(std::uint32_t) * 5 + size / 100 * sizeof(std::int32_t) + (1u << 20);
            if (errorMessage.empty() && neededBytes > freeBytes / 4 * 3) {
                errorMessage = "sin memoria en la GPU: se necesitan " + std::to_string(neededBytes) + " bytes y quedan " +
                               std::to_string(freeBytes) + " libres";
            }
            if (errorMessage.empty()) errorMessage = device_.allocate(size);
            if (errorMessage.empty()) errorMessage = device_.upload(keys_.data(), size, nullptr);  // datos residentes
            if (!errorMessage.empty()) {
                releaseAll();
                return "delivery_percentiles: " + errorMessage;
            }
            gpuReady_ = true;
        }
        size_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (size_ == 0) {
            return fail("delivery_percentiles: run sin prepare valido");
        }
        std::vector<std::uint32_t> sorted;
        accel::DeliveryPercentiles percentiles;
        switch (variant.version) {
            case Version::Cpu1: {
                std::vector<std::uint32_t> input = keys_;  // la copia de la entrada queda fuera de la medicion
                const auto startTime = Clock::now();
                sorted = accel::radixSortCpu1(std::move(input));
                percentiles = accel::percentilesFromSorted(sorted);
                runResult.timing.ms = elapsedMs(startTime, Clock::now());
                break;
            }
            case Version::CpuN: {
                accel::ThreadPool& pool = context.pool(variant.threads);
                std::vector<std::uint32_t> input = keys_;
                const auto startTime = Clock::now();
                sorted = accel::radixSortCpuN(std::move(input), pool);
                percentiles = accel::percentilesFromSorted(sorted);
                runResult.timing.ms = elapsedMs(startTime, Clock::now());
                break;
            }
            case Version::GpuKernel: {
                if (!gpuReady_) return fail("delivery_percentiles: GPU no disponible");
                double kernelMs = 0.0;
                std::string errorMessage = device_.run(&kernelMs);
                sorted.assign(size_, 0);
                // Verificacion fuera de la region medida: el arreglo ordenado vuelve al host.
                if (errorMessage.empty()) errorMessage = device_.download(sorted.data(), nullptr);
                if (!errorMessage.empty()) return fail("delivery_percentiles gpu_kernel: " + errorMessage);
                percentiles = accel::percentilesFromSorted(sorted);
                runResult.timing.ms = kernelMs;
                runResult.timing.kernelMs = kernelMs;
                break;
            }
            case Version::GpuE2e: {
                if (!gpuReady_) return fail("delivery_percentiles: GPU no disponible");
                double hostToDeviceMs = 0.0;
                double kernelMs = 0.0;
                double deviceToHostMs = 0.0;
                sorted.assign(size_, 0);
                const auto startTime = Clock::now();
                std::string errorMessage = device_.upload(keys_.data(), size_, &hostToDeviceMs);
                if (errorMessage.empty()) errorMessage = device_.run(&kernelMs);
                if (errorMessage.empty()) errorMessage = device_.download(sorted.data(), &deviceToHostMs);
                if (errorMessage.empty()) percentiles = accel::percentilesFromSorted(sorted);
                const auto stopTime = Clock::now();
                if (!errorMessage.empty()) return fail("delivery_percentiles gpu_e2e: " + errorMessage);
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                runResult.timing.h2dMs = hostToDeviceMs;
                runResult.timing.kernelMs = kernelMs;
                runResult.timing.d2hMs = deviceToHostMs;
                break;
            }
        }
        runResult.correct = sorted == referenceSorted_ && samePercentiles(percentiles, referencePercentiles_);
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        char description[120];
        const double millionsPerSecond =
            stats.medianMs > 0.0 ? static_cast<double>(stats.size) / (stats.medianMs * 1e-3) / 1e6 : 0.0;
        std::snprintf(description, sizeof description, "%.1f M keys/s sorted", millionsPerSecond);
        return description;
    }

private:
    static RunResult fail(const std::string& message) {
        RunResult runResult;
        runResult.ok = false;
        runResult.error = message;
        return runResult;
    }

    void releaseAll() {
        size_ = 0;
        gpuReady_ = false;
        keys_.clear();
        keys_.shrink_to_fit();
        referenceSorted_.clear();
        referenceSorted_.shrink_to_fit();
        device_.release();
    }

    std::size_t size_ = 0;
    std::vector<std::uint32_t> keys_;
    std::vector<std::uint32_t> referenceSorted_;
    accel::DeliveryPercentiles referencePercentiles_;
    accel::DeviceRadixSort device_;
    bool gpuReady_ = false;
};

BENCH_REGISTER_APP("delivery_percentiles", DeliveryPercentilesBench);

}  // namespace

}  // namespace bench
