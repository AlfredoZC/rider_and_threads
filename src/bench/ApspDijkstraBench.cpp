// La mejor CPU para la tabla de tiempos: Dijkstra desde cada nodo. El grafo de calles es ralo
// (~2,7 arcos por nodo), asi que hacer n Dijkstra cuesta del orden de n * (m + n) * log n, unas 85 veces menos
// que las n^3 relajaciones de Floyd-Warshall. Por eso la GPU se compara contra esta version, no solo contra apsp_floyd cpuN.
//
//   cpu1   Dijkstra desde cada nodo con 1 hilo (reutiliza Router::travelTimesFrom)
//   cpuN   las filas de origen repartidas en franjas con ThreadPool::parallelFor, un Router por hilo
//
// No hay version de GPU: esta app es la referencia de CPU del veredicto. size = nodos del subgrafo de la ciudad.
// Correcto = tabla igual, celda por celda, a la de Floyd-Warshall de 1 hilo (algoritmo independiente), calculada en prepare.
#include <chrono>
#include <exception>

#include "accel/TravelTimeTable.h"
#include "bench/ApspProblem.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

class ApspDijkstra : public BenchApp {
public:
    std::string name() const override { return "apsp_dijkstra"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override { return apspDefaultSizes(); }

    std::size_t maxSize() const override { return kApspMaxSize; }

    std::vector<Version> versions(bool) const override { return {Version::Cpu1, Version::CpuN}; }

    std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) override {
        return prepareApspProblem("apsp_dijkstra", size, seed, context, ApspReference::FloydWarshall, problem_);
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        if (problem_.nodeCount == 0) {
            return failedRun("apsp_dijkstra: run sin prepare valido");
        }
        RunResult runResult;
        try {
            problem_.resultCells.assign(problem_.resultCells.size(), 0);  // fuera de la region medida
            switch (variant.version) {
                case Version::Cpu1: {
                    const auto startTime = Clock::now();
                    accel::dijkstraFromEveryNode(*problem_.graph, problem_.resultCells);
                    const auto stopTime = Clock::now();
                    runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                    break;
                }
                case Version::CpuN: {
                    accel::ThreadPool& pool = context.pool(variant.threads);
                    const auto startTime = Clock::now();
                    accel::dijkstraFromEveryNodeParallel(*problem_.graph, problem_.resultCells, pool);
                    const auto stopTime = Clock::now();
                    runResult.timing.ms = std::chrono::duration<double, std::milli>(stopTime - startTime).count();
                    break;
                }
                case Version::GpuKernel:
                case Version::GpuE2e:
                    return failedRun("apsp_dijkstra: no tiene versiones de GPU");
            }
        } catch (const std::exception& caught) {
            return failedRun(std::string("apsp_dijkstra ") + versionName(variant.version) + ": " + caught.what());
        }
        markCorrectness(problem_, runResult);
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override { return describeCells(stats); }

private:
    ApspProblem problem_;
};

BENCH_REGISTER_APP("apsp_dijkstra", ApspDijkstra);

}  // namespace

}  // namespace bench
