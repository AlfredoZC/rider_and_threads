// Aplicacion 06: campo de cobertura de la flota (stencil de 5 puntos, barridos de Jacobi con doble buffer).
//
//   cpu1        un hilo, kSweepCount barridos
//   cpuN        por barrido, franjas de filas con ThreadPool::parallelFor (la vuelta es la barrera)
//   gpu_kernel  barrido 0 (D2D) + kSweepCount lanzamientos, uno por barrido, medidos con cudaEvent; la entrada ya
//               esta en la GPU
//   gpu_e2e     subir + barridos + bajar (steady_clock; h2d/kernel/d2h con cudaEvent); los buffers se reservan una
//               vez por tamano, fuera de la region medida
//
// size = lado de la grilla (size x size celdas). Entrada: fuentes sinteticas (~1 de cada 16 celdas interiores, con
// 1..3 couriers) con la semilla; el borde en 0. Todas las versiones hacen la MISMA cantidad de barridos.
// Correcto = |d| / (1 + |ref|) <= kCoverageTolerance contra cpu1 (en la practica identico: nada que fusionar en FMA).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <new>
#include <string>
#include <vector>

#include "accel/CoverageField.h"
#include "accel/DemandHeatmap.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxGridSideBench = 4096;
constexpr int kSweepCount = 200;

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

// splitmix64: fuentes deterministas segun la semilla, sin depender de la libreria.
std::vector<float> syntheticCouriers(int gridSide, std::uint32_t seed) {
    std::vector<float> couriersPerCell(static_cast<std::size_t>(gridSide) * static_cast<std::size_t>(gridSide), 0.0f);
    std::uint64_t state = seed;
    for (int cellY = 1; cellY < gridSide - 1; ++cellY) {
        for (int cellX = 1; cellX < gridSide - 1; ++cellX) {
            state += 0x9E3779B97F4A7C15ull;
            std::uint64_t mixed = state;
            mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
            mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
            mixed ^= mixed >> 31;
            if (mixed % 16 == 0) {
                couriersPerCell[static_cast<std::size_t>(cellY) * gridSide + cellX] = 1.0f + static_cast<float>((mixed >> 8) % 3);
            }
        }
    }
    return couriersPerCell;
}

class CoverageFieldBench : public BenchApp {
public:
    std::string name() const override { return "coverage_field"; }

    // 100 no es multiplo del bloque de 16: ejercita los hilos sobrantes del borde.
    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {8, 16, 32, 64, 100, 128, 256, 512, 1024};
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
        try {
            gridSide_ = static_cast<int>(size);
            couriersPerCell_ = syntheticCouriers(gridSide_, seed);
            coverage_.assign(couriersPerCell_.size(), 0.0f);
            reference_ = accel::coverageFieldCpu1(couriersPerCell_, gridSide_, gridSide_, kSweepCount);
        } catch (const std::bad_alloc&) {
            releaseAll();
            return name() + ": sin memoria en el host para una grilla de " + std::to_string(size) + " x " + std::to_string(size);
        }
        if (context.gpu.available) {
            std::string errorMessage = deviceGrid_.allocate(gridSide_, gridSide_);
            if (errorMessage.empty()) errorMessage = deviceGrid_.upload(couriersPerCell_, nullptr);  // queda en la GPU
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
        std::fill(coverage_.begin(), coverage_.end(), 0.0f);
        switch (variant.version) {
            case Version::Cpu1: {
                const auto startTime = Clock::now();
                coverage_ = accel::coverageFieldCpu1(couriersPerCell_, gridSide_, gridSide_, kSweepCount);
                const auto stopTime = Clock::now();
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                break;
            }
            case Version::CpuN: {
                accel::ThreadPool& pool = context.pool(variant.threads);
                const auto startTime = Clock::now();
                coverage_ = accel::coverageFieldCpuN(pool, couriersPerCell_, gridSide_, gridSide_, kSweepCount);
                const auto stopTime = Clock::now();
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                break;
            }
            case Version::GpuKernel: {
                if (!gpuReady_) return fail(name() + ": GPU no disponible");
                double kernelMs = 0.0;
                std::string errorMessage = deviceGrid_.run(kSweepCount, &kernelMs);
                if (!errorMessage.empty()) return fail(name() + " gpu_kernel: " + errorMessage);
                runResult.timing.ms = kernelMs;
                runResult.timing.kernelMs = kernelMs;
                // Verificacion fuera de la region medida.
                errorMessage = deviceGrid_.download(coverage_, nullptr);
                if (!errorMessage.empty()) return fail(name() + " gpu_kernel (verificacion): " + errorMessage);
                break;
            }
            case Version::GpuE2e: {
                if (!gpuReady_) return fail(name() + ": GPU no disponible");
                double hostToDeviceMs = 0.0;
                double kernelMs = 0.0;
                double deviceToHostMs = 0.0;
                const auto startTime = Clock::now();
                std::string errorMessage = deviceGrid_.upload(couriersPerCell_, &hostToDeviceMs);
                if (errorMessage.empty()) errorMessage = deviceGrid_.run(kSweepCount, &kernelMs);
                if (errorMessage.empty()) errorMessage = deviceGrid_.download(coverage_, &deviceToHostMs);
                const auto stopTime = Clock::now();  // download es sincrono: los barridos y las copias ya terminaron
                if (!errorMessage.empty()) return fail(name() + " gpu_e2e: " + errorMessage);
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                runResult.timing.h2dMs = hostToDeviceMs;
                runResult.timing.kernelMs = kernelMs;
                runResult.timing.d2hMs = deviceToHostMs;
                break;
            }
        }
        runResult.correct = coverage_.size() == reference_.size() &&
                            accel::maxRelativeDifference(reference_, coverage_) <= accel::kCoverageTolerance;
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        char description[200];
        const double updates = static_cast<double>(stats.size) * static_cast<double>(stats.size) * kSweepCount;
        const double mupdatesPerSecond = stats.medianMs > 0.0 ? updates / (stats.medianMs * 1e-3) / 1e6 : 0.0;
        const double microsecondsPerSweep = stats.medianMs * 1e3 / kSweepCount;
        std::snprintf(description, sizeof description, "%.1f Mcell-updates/s (%.1f us per sweep, %d sweeps)",
                      mupdatesPerSecond, microsecondsPerSweep, kSweepCount);
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
        for (std::vector<float>* buffer : {&couriersPerCell_, &coverage_, &reference_}) {
            buffer->clear();
            buffer->shrink_to_fit();
        }
    }

    std::size_t size_ = 0;
    int gridSide_ = 0;
    std::vector<float> couriersPerCell_;
    std::vector<float> coverage_;
    std::vector<float> reference_;
    accel::CoverageGpu deviceGrid_;
    bool gpuReady_ = false;
};

BENCH_REGISTER_APP("coverage_field", CoverageFieldBench);

}  // namespace

}  // namespace bench
