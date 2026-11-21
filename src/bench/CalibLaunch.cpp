// Calibracion de lanzamientos: cuanto cuesta lanzar un kernel que casi no hace nada (un hilo suma 1
// a un contador). Es el piso de latencia por debajo del cual ninguna aplicacion en GPU puede bajar.
//
//   cpu1        N llamadas a una funcion trivial (por puntero volatil, sin inline)
//   cpuN        N despachos de un parallelFor del pool propio (la latencia de despertar hilos)
//   gpu_kernel  N lanzamientos seguidos, una sola espera al final (cudaEvent de punta a punta)
//   gpu_e2e     N lanzamientos, cada uno seguido de cudaDeviceSynchronize (steady_clock)
//
// size = N. Correcto = el contador vale exactamente N (se ejecuto cada lanzamiento).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "accel/Calib.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxLaunches = 1000000;

void incrementOnce(std::uint64_t* launchesCounted) { *launchesCounted += 1; }

class CalibLaunch : public BenchApp {
public:
    std::string name() const override { return "calib_launch"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override { return {1, 10, 100, 1000, 10000}; }

    std::size_t maxSize() const override { return kMaxLaunches; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1, Version::CpuN};
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t, BenchContext& context) override {
        if (size == 0 || size > kMaxLaunches) {
            return "calib_launch: tamano fuera de rango (1 .. " + std::to_string(kMaxLaunches) + " lanzamientos)";
        }
        size_ = 0;
        gpuReady_ = false;
        if (context.gpu.available) {
            const std::string errorMessage = launchCounter_.allocate(sizeof(unsigned long long));
            if (!errorMessage.empty()) {
                return "calib_launch: " + errorMessage;
            }
            gpuReady_ = true;
        }
        size_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (size_ == 0) {
            return fail("calib_launch: run sin prepare valido");
        }
        std::uint64_t executedLaunches = 0;
        switch (variant.version) {
            case Version::Cpu1: {
                void (*volatile fn)(std::uint64_t*) = &incrementOnce;
                std::uint64_t launchesCounted = 0;
                const auto startTime = Clock::now();
                for (std::size_t launchIndex = 0; launchIndex < size_; ++launchIndex) {
                    fn(&launchesCounted);
                }
                const auto stopTime = Clock::now();
                runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                executedLaunches = launchesCounted;
                break;
            }
            case Version::CpuN: {
                accel::ThreadPool& pool = context.pool(variant.threads);
                std::atomic<std::uint64_t> launchesCounted{0};
                const auto startTime = Clock::now();
                for (std::size_t launchIndex = 0; launchIndex < size_; ++launchIndex) {
                    // Un despacho por iteracion: despierta a todos los hilos; un solo elemento de trabajo.
                    pool.parallelFor(1, [&launchesCounted](std::size_t chunkBegin, std::size_t chunkEnd) {
                        launchesCounted.fetch_add(chunkEnd - chunkBegin, std::memory_order_relaxed);
                    });
                }
                const auto stopTime = Clock::now();
                runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                executedLaunches = launchesCounted.load();
                break;
            }
            case Version::GpuKernel:
            case Version::GpuE2e: {
                if (!gpuReady_) return fail("calib_launch: GPU no disponible");
                const bool e2e = variant.version == Version::GpuE2e;
                std::string errorMessage = launchCounter_.clear();
                if (!errorMessage.empty()) return fail("calib_launch: " + errorMessage);
                double eventMs = 0.0;
                const auto startTime = Clock::now();
                errorMessage = accel::launchCounterKernel(launchCounter_, size_, e2e, &eventMs);
                if (errorMessage.empty()) errorMessage = accel::deviceSynchronize();
                const auto stopTime = Clock::now();
                if (!errorMessage.empty()) return fail("calib_launch " + std::string(versionName(variant.version)) + ": " + errorMessage);
                if (e2e) {
                    runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                } else {
                    runResult.timing.ms = eventMs;
                    runResult.timing.kernelMs = eventMs;
                }
                unsigned long long counterOnHost = 0;
                errorMessage = launchCounter_.download(&counterOnHost, sizeof counterOnHost, nullptr);
                if (!errorMessage.empty()) return fail("calib_launch (verificacion): " + errorMessage);
                executedLaunches = counterOnHost;
                break;
            }
        }
        runResult.correct = executedLaunches == size_;
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        char description[120];
        std::snprintf(description, sizeof description, "%.3f us per launch/dispatch", stats.medianMs * 1000.0 / static_cast<double>(stats.size));
        return description;
    }

private:
    static RunResult fail(const std::string& message) {
        RunResult runResult;
        runResult.ok = false;
        runResult.error = message;
        return runResult;
    }

    std::size_t size_ = 0;
    accel::DeviceBuffer launchCounter_;
    bool gpuReady_ = false;
};

BENCH_REGISTER_APP("calib_launch", CalibLaunch);

}  // namespace

}  // namespace bench
