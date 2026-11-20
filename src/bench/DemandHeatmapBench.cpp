// Aplicacion 05: mapa de calor de la demanda (convolucion 2D, filtro gaussiano 7 x 7 en memoria constante).
//
//   cpu1        un hilo
//   cpuN        franjas de filas con ThreadPool::parallelFor
//   gpu_kernel  solo el kernel (cudaEvent); la entrada ya esta en la GPU
//   gpu_e2e     subir + kernel + bajar (steady_clock con sincronizacion; h2d/kernel/d2h con cudaEvent);
//               los buffers se reservan una vez por tamano, fuera de la region medida
//
// size = lado de la grilla (size x size celdas). Entrada: conteos sinteticos enteros 0..3 con la semilla.
// Correcto = |d| / (1 + |ref|) <= kDemandTolerance contra cpu1 (float: la GPU fusiona a*b+c en una FMA).
// `demand_heatmap` mide el kernel PRINCIPAL (filtro en memoria constante); `demand_heatmap_tiles` la variante
// opcional con tile de memoria compartida y halo.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <new>
#include <string>
#include <vector>

#include "accel/DemandHeatmap.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxGridSideBench = 8192;
constexpr int kBenchFilterRadius = 3;

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

// splitmix64: conteos deterministas segun la semilla, sin depender de la libreria.
void fillSyntheticCounts(std::vector<float>& counts, std::uint32_t seed) {
    std::uint64_t state = seed;
    for (float& count : counts) {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t mixed = state;
        mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
        mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
        mixed ^= mixed >> 31;
        count = static_cast<float>(mixed % 4);
    }
}

class DemandHeatmapBench : public BenchApp {
public:
    DemandHeatmapBench(std::string appName, accel::DemandKernel gpuKernel)
        : appName_(std::move(appName)), gpuKernel_(gpuKernel) {}

    std::string name() const override { return appName_; }

    // 100 no es multiplo del bloque de 16: ejercita los hilos sobrantes del borde.
    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {16, 32, 64, 100, 128, 256, 512, 1024, 2048};
    }

    std::size_t maxSize() const override { return kMaxGridSideBench; }

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
        if (size == 0 || size > kMaxGridSideBench) {
            return name() + ": tamano fuera de rango (lado 1 .. " + std::to_string(kMaxGridSideBench) + ")";
        }
        const std::size_t cellCount = size * size;
        try {
            ordersPerCell_.assign(cellCount, 0.0f);
            demandHeat_.assign(cellCount, 0.0f);
            fillSyntheticCounts(ordersPerCell_, seed);
            filterWeights_ = accel::makeGaussianWeights(kBenchFilterRadius, accel::kDefaultFilterSigmaCells);
            gridSide_ = static_cast<int>(size);
            reference_ = accel::smoothDemandCpu1(ordersPerCell_, gridSide_, gridSide_, filterWeights_, kBenchFilterRadius);
        } catch (const std::bad_alloc&) {
            releaseAll();
            return name() + ": sin memoria en el host para una grilla de " + std::to_string(size) + " x " +
                   std::to_string(size);
        }
        if (context.gpu.available) {
            std::string errorMessage = deviceGrid_.allocate(gridSide_, gridSide_);
            if (errorMessage.empty()) errorMessage = deviceGrid_.setFilter(filterWeights_, kBenchFilterRadius);
            if (errorMessage.empty()) errorMessage = deviceGrid_.upload(ordersPerCell_, nullptr);  // queda en la GPU
            if (!errorMessage.empty()) {
                releaseAll();
                return name() + ": " + errorMessage;
            }
            gpuReady_ = true;
        }
        size_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (size_ == 0) {
            return fail(name() + ": run sin prepare valido");
        }
        std::fill(demandHeat_.begin(), demandHeat_.end(), 0.0f);
        switch (variant.version) {
            case Version::Cpu1: {
                const auto startTime = Clock::now();
                demandHeat_ = accel::smoothDemandCpu1(ordersPerCell_, gridSide_, gridSide_, filterWeights_, kBenchFilterRadius);
                const auto stopTime = Clock::now();
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                break;
            }
            case Version::CpuN: {
                accel::ThreadPool& pool = context.pool(variant.threads);
                const auto startTime = Clock::now();
                demandHeat_ = accel::smoothDemandCpuN(pool, ordersPerCell_, gridSide_, gridSide_, filterWeights_,
                                                      kBenchFilterRadius);
                const auto stopTime = Clock::now();
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                break;
            }
            case Version::GpuKernel: {
                if (!gpuReady_) return fail(name() + ": GPU no disponible");
                double kernelMs = 0.0;
                std::string errorMessage = deviceGrid_.launch(gpuKernel_, &kernelMs);
                if (!errorMessage.empty()) return fail(name() + " gpu_kernel: " + errorMessage);
                runResult.timing.ms = kernelMs;
                runResult.timing.kernelMs = kernelMs;
                // Verificacion fuera de la region medida.
                errorMessage = deviceGrid_.download(demandHeat_, nullptr);
                if (!errorMessage.empty()) return fail(name() + " gpu_kernel (verificacion): " + errorMessage);
                break;
            }
            case Version::GpuE2e: {
                if (!gpuReady_) return fail(name() + ": GPU no disponible");
                double hostToDeviceMs = 0.0;
                double kernelMs = 0.0;
                double deviceToHostMs = 0.0;
                const auto startTime = Clock::now();
                std::string errorMessage = deviceGrid_.upload(ordersPerCell_, &hostToDeviceMs);
                if (errorMessage.empty()) errorMessage = deviceGrid_.launch(gpuKernel_, &kernelMs);
                if (errorMessage.empty()) errorMessage = deviceGrid_.download(demandHeat_, &deviceToHostMs);
                const auto stopTime = Clock::now();  // download es sincrono: el kernel y las copias ya terminaron
                if (!errorMessage.empty()) return fail(name() + " gpu_e2e: " + errorMessage);
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                runResult.timing.h2dMs = hostToDeviceMs;
                runResult.timing.kernelMs = kernelMs;
                runResult.timing.d2hMs = deviceToHostMs;
                break;
            }
        }
        runResult.correct = demandHeat_.size() == reference_.size() &&
                            accel::maxRelativeDifference(reference_, demandHeat_) <= accel::kDemandTolerance;
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        char description[200];
        const double cells = static_cast<double>(stats.size) * static_cast<double>(stats.size);
        const double macs = cells * (2 * kBenchFilterRadius + 1) * (2 * kBenchFilterRadius + 1);
        const double mcellsPerSecond = stats.medianMs > 0.0 ? cells / (stats.medianMs * 1e-3) / 1e6 : 0.0;
        const double gmacsPerSecond = stats.medianMs > 0.0 ? macs / (stats.medianMs * 1e-3) / 1e9 : 0.0;
        std::snprintf(description, sizeof description, "%.1f Mcells/s (%.2f GMAC/s, filtro 7x7)", mcellsPerSecond,
                      gmacsPerSecond);
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
        gridSide_ = 0;
        deviceGrid_.release();
        for (std::vector<float>* buffer : {&ordersPerCell_, &demandHeat_, &reference_, &filterWeights_}) {
            buffer->clear();
            buffer->shrink_to_fit();
        }
    }

    std::string appName_;
    accel::DemandKernel gpuKernel_;
    std::size_t size_ = 0;
    int gridSide_ = 0;
    std::vector<float> ordersPerCell_;
    std::vector<float> demandHeat_;
    std::vector<float> reference_;
    std::vector<float> filterWeights_;
    accel::DemandHeatGpu deviceGrid_;
    bool gpuReady_ = false;
};

class DemandHeatmapConstantBench : public DemandHeatmapBench {
public:
    DemandHeatmapConstantBench() : DemandHeatmapBench("demand_heatmap", accel::DemandKernel::constantFilter) {}
};

class DemandHeatmapTiledBench : public DemandHeatmapBench {
public:
    DemandHeatmapTiledBench() : DemandHeatmapBench("demand_heatmap_tiles", accel::DemandKernel::tiledHalo) {}
};

BENCH_REGISTER_APP("demand_heatmap", DemandHeatmapConstantBench);
BENCH_REGISTER_APP("demand_heatmap_tiles", DemandHeatmapTiledBench);

}  // namespace

}  // namespace bench
