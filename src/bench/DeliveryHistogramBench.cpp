// Histograma de tiempos de entrega por minuto (atomicos y privatizacion).
//
//   delivery_histogram  cpu1, cpuN (histograma por trozo + suma final), gpu_kernel y gpu_e2e con el
//                       kernel PRIVATIZADO (copia por bloque en memoria compartida)
//   histogram_naive     solo GPU, con el kernel INGENUO (un atomico global por entrega): comparado con
//                       delivery_histogram muestra el costo de la contencion en las mismas 61 barras
//
// size = cantidad de tiempos de entrega. Los tiempos son sinteticos y deterministas segun la semilla:
// log-normal con mediana de 28 min (casi todo entre 10 y 60 min, cola larga hacia horas) y cada 1024
// entregas un valor exacto en un borde de minuto, para ejercitar los bordes de las barras y el desborde.
// Correcto = barras identicas a las de la referencia (cpu1 calculada en prepare) y suma = cantidad.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include "accel/Calib.h"
#include "accel/DeliveryHistogram.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"
#include "bench/DeliveryTimes.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxDeliveries = 100000000;  // 400 MB de tiempos en el host y otros tantos en la GPU

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

class DeliveryHistogramBench : public BenchApp {
public:
    DeliveryHistogramBench(std::string appName, accel::HistogramKernel kernel, bool gpuOnly)
        : appName_(std::move(appName)), kernel_(kernel), gpuOnly_(gpuOnly) {}

    std::string name() const override { return appName_; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {1000, 10000, 100000, 1000000, 10000000};
    }

    std::size_t maxSize() const override { return kMaxDeliveries; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions;
        if (!gpuOnly_) {
            supportedVersions = {Version::Cpu1, Version::CpuN};
        }
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        if (supportedVersions.empty()) {
            supportedVersions.push_back(Version::Cpu1);  // histogram_naive sin GPU: solo la referencia
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) override {
        releaseAll();
        if (size == 0 || size > kMaxDeliveries) {
            return appName_ + ": tamano fuera de rango (1 .. " + std::to_string(kMaxDeliveries) + ")";
        }
        try {
            deliveryTimesMs_.assign(size, 0);
        } catch (const std::bad_alloc&) {
            releaseAll();
            return appName_ + ": sin memoria en el host para " + std::to_string(size) + " tiempos";
        }
        fillDeliveryTimes(deliveryTimesMs_, seed);
        referenceBins_ = accel::deliveryTimeHistogramCpu1(deliveryTimesMs_);
        if (context.gpu.available) {
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            std::string errorMessage = accel::gpuMemory(&freeBytes, &totalBytes);
            if (errorMessage.empty() && size * sizeof(std::int32_t) > freeBytes / 2) {
                errorMessage = "sin memoria en la GPU: se necesitan " + std::to_string(size * sizeof(std::int32_t)) +
                               " bytes y quedan " + std::to_string(freeBytes) + " libres";
            }
            if (errorMessage.empty()) errorMessage = device_.allocate(size);
            if (errorMessage.empty()) errorMessage = device_.upload(deliveryTimesMs_.data(), size, nullptr);  // datos residentes
            if (!errorMessage.empty()) {
                releaseAll();
                return appName_ + ": " + errorMessage;
            }
            gpuReady_ = true;
        }
        size_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (size_ == 0) {
            return fail(appName_ + ": run sin prepare valido");
        }
        std::vector<unsigned> bins;
        switch (variant.version) {
            case Version::Cpu1: {
                const auto startTime = Clock::now();
                bins = accel::deliveryTimeHistogramCpu1(deliveryTimesMs_);
                runResult.timing.ms = elapsedMs(startTime, Clock::now());
                break;
            }
            case Version::CpuN: {
                accel::ThreadPool& pool = context.pool(variant.threads);
                const auto startTime = Clock::now();
                bins = accel::deliveryTimeHistogramCpuN(deliveryTimesMs_, pool);
                runResult.timing.ms = elapsedMs(startTime, Clock::now());
                break;
            }
            case Version::GpuKernel: {
                if (!gpuReady_) return fail(appName_ + ": GPU no disponible");
                double kernelMs = 0.0;
                std::string errorMessage = device_.run(kernel_, &kernelMs);
                bins.assign(accel::kDeliveryHistogramBins, 0);
                // Verificacion fuera de la region medida: las barras vuelven al host.
                if (errorMessage.empty()) errorMessage = device_.download(bins.data(), nullptr);
                if (!errorMessage.empty()) return fail(appName_ + " gpu_kernel: " + errorMessage);
                runResult.timing.ms = kernelMs;
                runResult.timing.kernelMs = kernelMs;
                break;
            }
            case Version::GpuE2e: {
                if (!gpuReady_) return fail(appName_ + ": GPU no disponible");
                double hostToDeviceMs = 0.0;
                double kernelMs = 0.0;
                double deviceToHostMs = 0.0;
                bins.assign(accel::kDeliveryHistogramBins, 0);
                const auto startTime = Clock::now();
                std::string errorMessage = device_.upload(deliveryTimesMs_.data(), size_, &hostToDeviceMs);
                if (errorMessage.empty()) errorMessage = device_.run(kernel_, &kernelMs);
                if (errorMessage.empty()) errorMessage = device_.download(bins.data(), &deviceToHostMs);
                const auto stopTime = Clock::now();
                if (!errorMessage.empty()) return fail(appName_ + " gpu_e2e: " + errorMessage);
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                runResult.timing.h2dMs = hostToDeviceMs;
                runResult.timing.kernelMs = kernelMs;
                runResult.timing.d2hMs = deviceToHostMs;
                break;
            }
        }
        const unsigned binTotal = std::accumulate(bins.begin(), bins.end(), 0u);
        runResult.correct = bins == referenceBins_ && binTotal == size_;
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        char description[120];
        const double millionsPerSecond =
            stats.medianMs > 0.0 ? static_cast<double>(stats.size) / (stats.medianMs * 1e-3) / 1e6 : 0.0;
        std::snprintf(description, sizeof description, "%.1f M deliveries/s", millionsPerSecond);
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
        deliveryTimesMs_.clear();
        deliveryTimesMs_.shrink_to_fit();
        referenceBins_.clear();
        device_.release();
    }

    std::string appName_;
    accel::HistogramKernel kernel_;
    bool gpuOnly_;
    std::size_t size_ = 0;
    std::vector<std::int32_t> deliveryTimesMs_;
    std::vector<unsigned> referenceBins_;
    accel::DeviceHistogram device_;
    bool gpuReady_ = false;
};

class DeliveryHistogramPrivatizedBench : public DeliveryHistogramBench {
public:
    DeliveryHistogramPrivatizedBench()
        : DeliveryHistogramBench("delivery_histogram", accel::HistogramKernel::privatized, false) {}
};

class HistogramNaiveBench : public DeliveryHistogramBench {
public:
    HistogramNaiveBench() : DeliveryHistogramBench("histogram_naive", accel::HistogramKernel::globalAtomics, true) {}
};

BENCH_REGISTER_APP("delivery_histogram", DeliveryHistogramPrivatizedBench);
BENCH_REGISTER_APP("histogram_naive", HistogramNaiveBench);

}  // namespace

}  // namespace bench
