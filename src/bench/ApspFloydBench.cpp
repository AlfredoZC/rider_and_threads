// Aplicacion 03: tabla de tiempos entre todos los pares (Floyd-Warshall). Entrada: un subgrafo conexo de
// la ciudad real (CitySubgraph) con `size` nodos; la region es determinista, asi que la semilla no la cambia.
//
//   cpu1        Floyd-Warshall con 1 hilo
//   cpuN        Floyd-Warshall con las filas repartidas por ThreadPool::parallelFor en cada pivote
//   gpu_kernel  solo los kernels por tiles (cudaEvent); la matriz ya esta en la GPU, el resultado se baja
//               despues para verificarlo, fuera de la region medida
//   gpu_e2e     relleno + H2D + kernels + D2H + quitar el relleno (steady_clock); cudaMalloc queda fuera
//
// Correcto = tabla identica (enteros) a la de Dijkstra desde cada nodo con 1 hilo, que el test
// test_travel_time_table.cpp demuestra igual a Floyd-Warshall de 1 hilo.
#include <chrono>
#include <cstdio>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include "CityGraph.h"
#include "accel/Calib.h"
#include "accel/TravelTimeTable.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"
#include "bench/BenchGraph.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxNodes = 8192;  // 256 MiB de tabla

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

class ApspFloydBench : public BenchApp {
public:
    std::string name() const override { return "apsp_floyd"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {25, 100, 400, 1000, 2000, 3741};
    }

    std::size_t maxSize() const override { return kMaxNodes; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1, Version::CpuN};
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t, BenchContext& context) override {
        releaseAll();
        if (size == 0 || size > kMaxNodes) {
            return "apsp_floyd: tamano fuera de rango (1 .. " + std::to_string(kMaxNodes) + " nodos)";
        }
        try {
            const Config region = cityRegionOrSynthetic(context, size);
            const CityGraph graph(region);
            initialCells_ = accel::buildInitialCells(graph);
            reference_ = accel::dijkstraAllPairsSingleThread(graph);
            if (context.gpu.available) {
                std::size_t freeBytes = 0;
                std::size_t totalBytes = 0;
                std::string errorMessage = accel::gpuMemory(&freeBytes, &totalBytes);
                const std::size_t paddedCount = accel::paddedNodeCount(size, accel::kApspTileSize);
                const std::size_t tableBytes = paddedCount * paddedCount * sizeof(std::int32_t);
                if (errorMessage.empty() && tableBytes > freeBytes / 4 * 3) {
                    errorMessage = "sin memoria en la GPU: la tabla necesita " + std::to_string(tableBytes) +
                                   " bytes y quedan " + std::to_string(freeBytes) + " libres";
                }
                if (errorMessage.empty()) errorMessage = device_.allocate(paddedCount);
                if (!errorMessage.empty()) {
                    releaseAll();
                    return "apsp_floyd: " + errorMessage;
                }
                paddedCount_ = paddedCount;
                gpuReady_ = true;
            }
        } catch (const std::bad_alloc&) {
            releaseAll();
            return "apsp_floyd: sin memoria en el host";
        } catch (const std::exception& error) {
            releaseAll();
            return std::string("apsp_floyd: ") + error.what();
        }
        nodeCount_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (nodeCount_ == 0) return fail("apsp_floyd: run sin prepare valido");
        std::vector<std::int32_t> result;
        try {
            switch (variant.version) {
                case Version::Cpu1: {
                    result = initialCells_;
                    const auto startTime = Clock::now();
                    accel::floydWarshallSingleThread(result, nodeCount_);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    break;
                }
                case Version::CpuN: {
                    accel::ThreadPool& pool = context.pool(variant.threads);
                    result = initialCells_;
                    const auto startTime = Clock::now();
                    accel::floydWarshallMultiThread(result, nodeCount_, pool);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    break;
                }
                case Version::GpuKernel: {
                    if (!gpuReady_) return fail("apsp_floyd: GPU no disponible");
                    std::vector<std::int32_t> padded = accel::padCells(initialCells_, nodeCount_, paddedCount_);
                    std::string errorMessage = device_.upload(padded.data(), nullptr);
                    double kernelMs = 0.0;
                    if (errorMessage.empty()) errorMessage = device_.runTiled(accel::kApspTileSize, &kernelMs);
                    if (!errorMessage.empty()) return fail("apsp_floyd gpu_kernel: " + errorMessage);
                    runResult.timing.ms = kernelMs;
                    runResult.timing.kernelMs = kernelMs;
                    // Verificacion fuera de la region medida.
                    errorMessage = device_.download(padded.data(), nullptr);
                    if (!errorMessage.empty()) return fail("apsp_floyd gpu_kernel (verificacion): " + errorMessage);
                    result = accel::unpadCells(padded, paddedCount_, nodeCount_);
                    break;
                }
                case Version::GpuE2e: {
                    if (!gpuReady_) return fail("apsp_floyd: GPU no disponible");
                    accel::ApspGpuTiming timing;
                    std::vector<std::int32_t> padded;
                    const auto startTime = Clock::now();
                    padded = accel::padCells(initialCells_, nodeCount_, paddedCount_);
                    std::string errorMessage = device_.upload(padded.data(), &timing.h2dMs);
                    if (errorMessage.empty()) errorMessage = device_.runTiled(accel::kApspTileSize, &timing.kernelMs);
                    if (errorMessage.empty()) errorMessage = device_.download(padded.data(), &timing.d2hMs);
                    if (errorMessage.empty()) errorMessage = accel::deviceSynchronize();
                    if (!errorMessage.empty()) return fail("apsp_floyd gpu_e2e: " + errorMessage);
                    result = accel::unpadCells(padded, paddedCount_, nodeCount_);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    runResult.timing.h2dMs = timing.h2dMs;
                    runResult.timing.kernelMs = timing.kernelMs;
                    runResult.timing.d2hMs = timing.d2hMs;
                    break;
                }
            }
        } catch (const std::exception& error) {
            return fail(std::string("apsp_floyd: ") + error.what());
        }
        runResult.correct = (result == reference_);
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        if (stats.medianMs <= 0.0) return "";
        const double pairs = static_cast<double>(stats.size) * static_cast<double>(stats.size);
        char description[120];
        std::snprintf(description, sizeof description, "%.1f M pares/s (tabla de %zu x %zu)",
                      pairs / (stats.medianMs * 1e3), stats.size, stats.size);
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
        nodeCount_ = 0;
        paddedCount_ = 0;
        gpuReady_ = false;
        initialCells_.clear();
        initialCells_.shrink_to_fit();
        reference_.clear();
        reference_.shrink_to_fit();
        device_.release();
    }

    std::size_t nodeCount_ = 0;
    std::size_t paddedCount_ = 0;
    std::vector<std::int32_t> initialCells_;
    std::vector<std::int32_t> reference_;
    accel::ApspDevice device_;
    bool gpuReady_ = false;
};

BENCH_REGISTER_APP("apsp_floyd", ApspFloydBench);

}  // namespace

}  // namespace bench
