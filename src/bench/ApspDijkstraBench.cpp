// Aplicacion 03 (mejor CPU honesta): la misma tabla de todos los pares calculada con Dijkstra desde cada
// nodo. Solo CPU: sirve de comparacion justa para Floyd-Warshall en GPU (apsp_floyd).
//
//   cpu1  Dijkstra desde cada nodo con 1 hilo
//   cpuN  Dijkstra desde cada nodo, las fuentes repartidas con ThreadPool::parallelFor
//
// Correcto = tabla identica a la de Floyd-Warshall con 1 hilo.
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "CityGraph.h"
#include "accel/TravelTimeTable.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"
#include "bench/BenchGraph.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxNodes = 8192;

class ApspDijkstraBench : public BenchApp {
public:
    std::string name() const override { return "apsp_dijkstra"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {25, 100, 400, 1000, 2000, 3741};
    }

    std::size_t maxSize() const override { return kMaxNodes; }

    std::vector<Version> versions(bool) const override { return {Version::Cpu1, Version::CpuN}; }

    std::string prepare(std::size_t size, std::uint32_t, BenchContext& context) override {
        clearState();
        if (size == 0 || size > kMaxNodes) {
            return "apsp_dijkstra: tamano fuera de rango (1 .. " + std::to_string(kMaxNodes) + " nodos)";
        }
        try {
            const Config region = cityRegionOrSynthetic(context, size);
            graph_ = std::make_unique<CityGraph>(region);
            reference_ = accel::buildInitialCells(*graph_);
            accel::floydWarshallSingleThread(reference_, size);
        } catch (const std::bad_alloc&) {
            clearState();
            return "apsp_dijkstra: sin memoria en el host";
        } catch (const std::exception& error) {
            clearState();
            return std::string("apsp_dijkstra: ") + error.what();
        }
        nodeCount_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (nodeCount_ == 0) {
            runResult.ok = false;
            runResult.error = "apsp_dijkstra: run sin prepare valido";
            return runResult;
        }
        std::vector<std::int32_t> result;
        try {
            const auto startTime = Clock::now();
            if (variant.version == Version::CpuN) {
                result = accel::dijkstraAllPairsMultiThread(*graph_, context.pool(variant.threads));
            } else {
                result = accel::dijkstraAllPairsSingleThread(*graph_);
            }
            runResult.timing.ms = std::chrono::duration<double, std::milli>(Clock::now() - startTime).count();
        } catch (const std::exception& error) {
            runResult.ok = false;
            runResult.error = std::string("apsp_dijkstra: ") + error.what();
            return runResult;
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
    void clearState() {
        nodeCount_ = 0;
        graph_.reset();
        reference_.clear();
    }

    std::size_t nodeCount_ = 0;
    std::unique_ptr<CityGraph> graph_;
    std::vector<std::int32_t> reference_;
};

BENCH_REGISTER_APP("apsp_dijkstra", ApspDijkstraBench);

}  // namespace

}  // namespace bench
