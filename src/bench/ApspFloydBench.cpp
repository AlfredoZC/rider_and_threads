// Tabla de tiempos entre todos los pares (Floyd-Warshall con tiles en memoria compartida).
//
//   cpu1        Floyd-Warshall de 1 hilo
//   cpuN        Floyd-Warshall con las filas de cada pasada repartidas en franjas (ThreadPool::parallelFor)
//   gpu_kernel  kernels con tiles de 32 x 32 en memoria compartida, 3 fases por tile pivote (cudaEvent, solo kernels)
//   gpu_e2e     todo el envoltorio (relleno a multiplo de 32, cudaMalloc, H2D, kernels, D2H, quitar relleno);
//               h2d_ms / kernel_ms / d2h_ms con cudaEvent
//
// size = nodos del subgrafo de la ciudad real (25 .. 3741; sin ciudad, un grafo sintetico). Correcto = la tabla
// es IGUAL, celda por celda (enteros), a la de Dijkstra desde cada nodo, un algoritmo distinto que no comparte
// codigo con Floyd-Warshall; se calcula con 1 hilo en prepare. La matriz inicial se arma en prepare, fuera de la region medida.
#include <chrono>
#include <exception>

#include "accel/TravelTimeTable.h"
#include "bench/ApspProblem.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

class ApspFloyd : public BenchApp {
public:
    std::string name() const override { return "apsp_floyd"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override { return apspDefaultSizes(); }

    std::size_t maxSize() const override { return kApspMaxSize; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1, Version::CpuN};
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) override {
        return prepareApspProblem("apsp_floyd", size, seed, context, ApspReference::Dijkstra, problem_);
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        if (problem_.nodeCount == 0) {
            return failedRun("apsp_floyd: run sin prepare valido");
        }
        RunResult runResult;
        try {
            switch (variant.version) {
                case Version::Cpu1: {
                    problem_.resultCells = problem_.initialCells;  // fuera de la region medida
                    const auto startTime = Clock::now();
                    accel::floydWarshallSingleThread(problem_.resultCells, problem_.nodeCount);
                    const auto stopTime = Clock::now();
                    runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                    break;
                }
                case Version::CpuN: {
                    accel::ThreadPool& pool = context.pool(variant.threads);
                    problem_.resultCells = problem_.initialCells;
                    const auto startTime = Clock::now();
                    accel::floydWarshallMultiThread(problem_.resultCells, problem_.nodeCount, pool);
                    const auto stopTime = Clock::now();
                    runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                    break;
                }
                case Version::GpuKernel:
                case Version::GpuE2e:
                    return runGpuFloyd(problem_, accel::floydWarshallTiledGpu, variant.version, "apsp_floyd");
            }
        } catch (const std::exception& caught) {
            return failedRun(std::string("apsp_floyd ") + versionName(variant.version) + ": " + caught.what());
        }
        markCorrectness(problem_, runResult);
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override { return describeRelaxations(stats); }

private:
    ApspProblem problem_;
};

BENCH_REGISTER_APP("apsp_floyd", ApspFloyd);

}  // namespace

}  // namespace bench
