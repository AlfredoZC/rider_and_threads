// El "antes" del tiling: Floyd-Warshall en la GPU SIN memoria compartida, un
// lanzamiento por cada nodo intermedio y un hilo por celda, todo en memoria global. Se mide contra apsp_floyd
// (tiles de 32 x 32) en los mismos tamanos para mostrar cuanto aporta el tiling.
//
//   cpu1        Dijkstra desde cada nodo, 1 hilo: es LA REFERENCIA de la app (la misma que usa apsp_floyd). La app es
//               de GPU y bench_calib exige que toda app registrada tenga cpu1 y cpuN que corran sin GPU y sin
//               ciudad; las versiones de CPU son entonces las de la mejor CPU (Dijkstra) y las GPU se comparan
//               contra la tabla de cpu1.
//   cpuN        Dijkstra desde cada nodo con N hilos (un Router por hilo)
//   gpu_kernel  todos los lanzamientos ingenuos (cudaEvent, solo kernels)
//   gpu_e2e     envoltorio completo (cudaMalloc, H2D, kernels, D2H); h2d_ms / kernel_ms / d2h_ms con cudaEvent
//
// size = nodos del subgrafo.
// Correcto = tabla igual, celda por celda, a la de Dijkstra desde cada nodo.
#include <chrono>
#include <exception>

#include "accel/TravelTimeTable.h"
#include "bench/ApspProblem.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

class ApspFloydNaive : public BenchApp {
public:
    std::string name() const override { return "apsp_floyd_naive"; }

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
        return prepareApspProblem("apsp_floyd_naive", size, seed, context, ApspReference::Dijkstra, problem_);
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        if (problem_.nodeCount == 0) {
            return failedRun("apsp_floyd_naive: run sin prepare valido");
        }
        RunResult runResult;
        switch (variant.version) {
            case Version::Cpu1: {
                try {
                    problem_.resultCells.assign(problem_.resultCells.size(), 0);  // fuera de la region medida
                    const auto startTime = Clock::now();
                    accel::dijkstraFromEveryNode(*problem_.graph, problem_.resultCells);
                    const auto stopTime = Clock::now();
                    runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                } catch (const std::exception& caught) {
                    return failedRun(std::string("apsp_floyd_naive cpu1: ") + caught.what());
                }
                break;
            }
            case Version::GpuKernel:
            case Version::GpuE2e:
                return runGpuFloyd(problem_, accel::floydWarshallNaiveGpu, variant.version, "apsp_floyd_naive");
            case Version::CpuN: {
                try {
                    accel::ThreadPool& pool = context.pool(variant.threads);
                    problem_.resultCells.assign(problem_.resultCells.size(), 0);
                    const auto startTime = Clock::now();
                    accel::dijkstraFromEveryNodeParallel(*problem_.graph, problem_.resultCells, pool);
                    const auto stopTime = Clock::now();
                    runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                } catch (const std::exception& caught) {
                    return failedRun(std::string("apsp_floyd_naive cpuN: ") + caught.what());
                }
                break;
            }
        }
        markCorrectness(problem_, runResult);
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        // cpu1 y cpuN son Dijkstra (n^2 celdas por segundo); las GPU son Floyd-Warshall (relajaciones por segundo).
        return (stats.version == Version::Cpu1 || stats.version == Version::CpuN) ? describeCells(stats) : describeRelaxations(stats);
    }

private:
    ApspProblem problem_;
};

BENCH_REGISTER_APP("apsp_floyd_naive", ApspFloydNaive);

}  // namespace

}  // namespace bench
