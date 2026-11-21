// Aplicacion 10: filtrar couriers candidatos (filter / compactacion). size = cantidad de couriers.
// La flota se ubica en nodos reales de la ciudad (semilla fija; ~70 % disponibles); la tabla de tiempos
// es la de la aplicacion 03 (Dijkstra desde cada nodo) y el radio es el percentil 25 de los tiempos
// al primer restaurante, asi pasa una parte chica pero no vacia de la flota.
//
//   cpu1        un hilo recorre la flota
//   cpuN        una lista por franja en ThreadPool::parallelFor, concatenadas en orden de franja
//   gpu_kernel  solo el kernel (cudaEvent): la tabla y la flota ya estan en la GPU
//   gpu_e2e     copia la tabla (56 MB para la ciudad completa) y la flota, kernel y trae los candidatos
//               (steady_clock; h2d_ms = las dos copias, d2h_ms = contador + lista)
//
// Dos aplicaciones registradas: `candidate_filter` usa el kernel con lista privada por bloque y
// `candidate_filter_naive` el de un atomico global por candidato (mide la contencion). Correcto = el
// MISMO CONJUNTO de candidatos que cpu1 (en la GPU el orden no esta fijado, asi que se ordena antes de comparar).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "CityGraph.h"
#include "accel/CandidateFilter.h"
#include "accel/Calib.h"
#include "bench/BenchApp.h"
#include "bench/BenchGraph.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxCouriers = 2000000;
constexpr std::size_t kSyntheticNodes = 400;  // nodos del grafo sintetico cuando no hay ciudad

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

class CandidateFilterBench : public BenchApp {
public:
    CandidateFilterBench(std::string appName, bool naiveKernel) : appName_(std::move(appName)), naive_(naiveKernel) {}

    std::string name() const override { return appName_; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {100, 500, 2000, 10000, 50000, 200000};
    }

    std::size_t maxSize() const override { return kMaxCouriers; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1};
        if (!naive_) supportedVersions.push_back(Version::CpuN);
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) override {
        fleetSize_ = 0;
        reference_.clear();
        if (size == 0 || size > kMaxCouriers) {
            return name() + ": tamano fuera de rango (1 .. " + std::to_string(kMaxCouriers) + " couriers)";
        }
        try {
            if (!table_) {
                const std::string errorMessage = buildTable(context);
                if (!errorMessage.empty()) return errorMessage;
            }
            std::mt19937 generator(seed);
            fleet_.node.clear();
            fleet_.available.clear();
            for (std::size_t courier = 0; courier < size; ++courier) {
                fleet_.node.push_back(static_cast<std::int32_t>(generator() % table_->nodeCount()));
                fleet_.available.push_back(generator() % 10 < 7 ? 1 : 0);
            }
            reference_ = accel::filterCandidatesSingleThread(fleet_, *table_, restaurantNode_, maxTravelMs_);
            if (context.gpu.available) {
                std::string errorMessage;
                if (!tableOnDevice_) {
                    errorMessage = device_.uploadTable(table_->cells().data(), table_->nodeCount(), nullptr);
                    if (errorMessage.empty()) tableOnDevice_ = true;
                }
                if (errorMessage.empty()) errorMessage = device_.reserveFleet(size);
                if (errorMessage.empty()) errorMessage = device_.uploadFleet(fleet_, nullptr);
                if (!errorMessage.empty()) return name() + ": " + errorMessage;
                gpuReady_ = true;
            }
        } catch (const std::bad_alloc&) {
            return name() + ": sin memoria en el host";
        } catch (const std::exception& error) {
            return name() + ": " + error.what();
        }
        fleetSize_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (fleetSize_ == 0) return fail(name() + ": run sin prepare valido");
        std::vector<std::int32_t> result;
        try {
            switch (variant.version) {
                case Version::Cpu1: {
                    const auto startTime = Clock::now();
                    result = accel::filterCandidatesSingleThread(fleet_, *table_, restaurantNode_, maxTravelMs_);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    break;
                }
                case Version::CpuN: {
                    accel::ThreadPool& pool = context.pool(variant.threads);
                    const auto startTime = Clock::now();
                    result = accel::filterCandidatesMultiThread(fleet_, *table_, restaurantNode_, maxTravelMs_, pool);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    break;
                }
                case Version::GpuKernel: {
                    if (!gpuReady_) return fail(name() + ": GPU no disponible");
                    double kernelMs = 0.0;
                    std::string errorMessage = device_.runFilter(naive_, restaurantNode_, maxTravelMs_, &kernelMs);
                    if (!errorMessage.empty()) return fail(name() + " gpu_kernel: " + errorMessage);
                    runResult.timing.ms = kernelMs;
                    runResult.timing.kernelMs = kernelMs;
                    errorMessage = device_.download(result, nullptr);  // verificacion, fuera de la region medida
                    if (!errorMessage.empty()) return fail(name() + " gpu_kernel (verificacion): " + errorMessage);
                    break;
                }
                case Version::GpuE2e: {
                    if (!gpuReady_) return fail(name() + ": GPU no disponible");
                    double tableMs = 0.0;
                    double fleetMs = 0.0;
                    double kernelMs = 0.0;
                    double downloadMs = 0.0;
                    const auto startTime = Clock::now();
                    std::string errorMessage = device_.uploadTable(table_->cells().data(), table_->nodeCount(), &tableMs);
                    if (errorMessage.empty()) errorMessage = device_.uploadFleet(fleet_, &fleetMs);
                    if (errorMessage.empty()) {
                        errorMessage = device_.runFilter(naive_, restaurantNode_, maxTravelMs_, &kernelMs);
                    }
                    if (errorMessage.empty()) errorMessage = device_.download(result, &downloadMs);
                    if (errorMessage.empty()) errorMessage = accel::deviceSynchronize();
                    if (!errorMessage.empty()) return fail(name() + " gpu_e2e: " + errorMessage);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    runResult.timing.h2dMs = tableMs + fleetMs;
                    runResult.timing.kernelMs = kernelMs;
                    runResult.timing.d2hMs = downloadMs;
                    break;
                }
            }
        } catch (const std::exception& error) {
            return fail(name() + ": " + error.what());
        }
        std::sort(result.begin(), result.end());  // el conjunto, no el orden
        runResult.correct = (result == reference_);
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        if (stats.medianMs <= 0.0) return "";
        char description[120];
        std::snprintf(description, sizeof description, "%.1f M couriers/s (%zu couriers, %zu candidatos)",
                      static_cast<double>(stats.size) / (stats.medianMs * 1e3), stats.size, reference_.size());
        return description;
    }

private:
    static RunResult fail(const std::string& message) {
        RunResult runResult;
        runResult.ok = false;
        runResult.error = message;
        return runResult;
    }

    // La tabla de toda la ciudad (la de la aplicacion 03), el restaurante y el radio.
    std::string buildTable(BenchContext& context) {
        // Con ciudad: toda la ciudad. Sin ciudad (contexto vacio, p. ej. en los tests): grafo sintetico.
        const Config region = context.config.nodes.empty() ? cityRegionOrSynthetic(context, kSyntheticNodes) : context.config;
        const CityGraph graph(region);
        if (graph.nodeCount() == 0 || region.restaurants.empty()) {
            return name() + ": la ciudad no tiene nodos o restaurantes";
        }
        const NodeIndex restaurant = graph.indexOf(region.restaurants.front().node);
        if (restaurant == kNoNode) return name() + ": restaurante fuera del grafo";
        accel::ThreadPool& pool = context.pool(std::max(1u, std::thread::hardware_concurrency()));
        table_ = accel::buildTravelTimeTableCpu(graph, pool);
        restaurantNode_ = static_cast<std::int32_t>(restaurant);
        // Percentil 25 de los tiempos (nodo -> restaurante) que si llegan.
        std::vector<std::int32_t> times;
        for (std::size_t node = 0; node < table_->nodeCount(); ++node) {
            const std::int64_t travelMs = table_->travelTimeMs(static_cast<NodeIndex>(node), restaurant);
            if (travelMs != accel::TravelTimeTable::kNoRoute) times.push_back(static_cast<std::int32_t>(travelMs));
        }
        if (times.empty()) return name() + ": nadie llega al restaurante";
        std::sort(times.begin(), times.end());
        maxTravelMs_ = times[times.size() / 4];
        return "";
    }

    std::string appName_;
    bool naive_;
    std::shared_ptr<const accel::TravelTimeTable> table_;
    std::int32_t restaurantNode_ = 0;
    std::int32_t maxTravelMs_ = 0;
    accel::FleetColumns fleet_;
    std::vector<std::int32_t> reference_;
    std::size_t fleetSize_ = 0;
    accel::CandidateFilterDevice device_;
    bool tableOnDevice_ = false;
    bool gpuReady_ = false;
};

class CandidateFilterBlockBench : public CandidateFilterBench {
public:
    CandidateFilterBlockBench() : CandidateFilterBench("candidate_filter", false) {}
};

class CandidateFilterNaiveBench : public CandidateFilterBench {
public:
    CandidateFilterNaiveBench() : CandidateFilterBench("candidate_filter_naive", true) {}
};

BENCH_REGISTER_APP("candidate_filter", CandidateFilterBlockBench);
BENCH_REGISTER_APP("candidate_filter_naive", CandidateFilterNaiveBench);

}  // namespace

}  // namespace bench
