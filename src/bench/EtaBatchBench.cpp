// Aplicacion 04: ETA en lote, una matriz pedidos x couriers. size = cantidad de couriers (columnas); los
// pedidos (filas) salen de la tabla 50 -> 10, 150 -> 30, 500 -> 100, 2000 -> 300, 5000 -> 600 y, para
// cualquier otro tamano (p. ej. 777, que no es multiplo del bloque de 256), size / 8 pedidos (minimo 1).
// Los nodos son reales de la ciudad (o del grafo sintetico sin ciudad), con semilla fija.
//
//   cpu1        un hilo llena la matriz
//   cpuN        las columnas (couriers) en franjas de ThreadPool::parallelFor
//   gpu_kernel  solo el kernel (cudaEvent): la tabla, los couriers y los pedidos ya estan en la GPU
//   gpu_e2e     copia la tabla (56 MB para la ciudad completa), couriers y pedidos, kernel y trae la matriz
//               (steady_clock; h2d_ms = las copias, d2h_ms = la matriz)
//
// Dos aplicaciones registradas con la MISMA formula: `eta_batch_soa` lee los couriers como estructura de
// arreglos (lecturas coalescidas) y `eta_batch_aos` como registros de 32 bytes (cada hilo usa 8 de 32).
// Correcto = matriz identica (enteros) a la de cpu1. `describe` informa los bytes utiles y movidos (modelo)
// y los GB/s efectivos.
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
#include "accel/Calib.h"
#include "accel/EtaBatch.h"
#include "bench/BenchApp.h"
#include "bench/BenchGraph.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxCouriers = 200000;
constexpr std::size_t kSyntheticNodes = 400;  // nodos del grafo sintetico cuando no hay ciudad

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

std::size_t orderCountFor(std::size_t courierCount) {
    switch (courierCount) {
        case 50: return 10;
        case 150: return 30;
        case 500: return 100;
        case 2000: return 300;
        case 5000: return 600;
        default: return std::max<std::size_t>(1, courierCount / 8);
    }
}

class EtaBatchBench : public BenchApp {
public:
    EtaBatchBench(std::string appName, bool aosLayout) : appName_(std::move(appName)), aos_(aosLayout) {}

    std::string name() const override { return appName_; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {50, 150, 500, 777, 2000, 5000};
    }

    std::size_t maxSize() const override { return kMaxCouriers; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1};
        if (!aos_) supportedVersions.push_back(Version::CpuN);
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) override {
        ready_ = false;
        reference_.clear();
        if (size == 0 || size > kMaxCouriers) {
            return name() + ": tamano fuera de rango (1 .. " + std::to_string(kMaxCouriers) + " couriers)";
        }
        try {
            if (!table_) {
                const Config region = context.config.nodes.empty() ? cityRegionOrSynthetic(context, kSyntheticNodes)
                                                                   : context.config;
                const CityGraph graph(region);
                if (graph.nodeCount() == 0) return name() + ": la ciudad no tiene nodos";
                table_ = accel::buildTravelTimeTableCpu(graph, context.pool(std::max(1u, std::thread::hardware_concurrency())));
            }
            const std::size_t nodeCount = table_->nodeCount();
            const std::size_t orderCount = orderCountFor(size);
            std::mt19937 generator(seed);
            couriers_ = accel::CourierColumns{};
            orders_ = accel::OrderColumns{};
            for (std::size_t courier = 0; courier < size; ++courier) {
                couriers_.node.push_back(static_cast<std::int32_t>(generator() % nodeCount));
                couriers_.freeAtMs.push_back(static_cast<std::int32_t>(generator() % 30000));
            }
            for (std::size_t order = 0; order < orderCount; ++order) {
                orders_.restaurantNode.push_back(static_cast<std::int32_t>(generator() % nodeCount));
                orders_.customerNode.push_back(static_cast<std::int32_t>(generator() % nodeCount));
                orders_.readyAtMs.push_back(static_cast<std::int32_t>(generator() % 40000));
            }
            records_ = accel::toRecords(couriers_);
            reference_ = accel::computeEtaSingleThread(couriers_, orders_, *table_);
            if (context.gpu.available) {
                std::string errorMessage;
                if (!tableOnDevice_) {
                    errorMessage = device_.uploadTable(table_->cells().data(), nodeCount, nullptr);
                    if (errorMessage.empty()) tableOnDevice_ = true;
                }
                if (errorMessage.empty()) {
                    errorMessage = aos_ ? device_.uploadCouriersAoS(records_, nullptr)
                                        : device_.uploadCouriersSoA(couriers_, nullptr);
                }
                if (errorMessage.empty()) errorMessage = device_.uploadOrders(orders_, nullptr);
                if (!errorMessage.empty()) return name() + ": " + errorMessage;
                gpuReady_ = true;
            }
        } catch (const std::bad_alloc&) {
            return name() + ": sin memoria en el host";
        } catch (const std::exception& error) {
            return name() + ": " + error.what();
        }
        ready_ = true;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (!ready_) return fail(name() + ": run sin prepare valido");
        std::vector<std::int32_t> result;
        try {
            switch (variant.version) {
                case Version::Cpu1: {
                    const auto startTime = Clock::now();
                    result = accel::computeEtaSingleThread(couriers_, orders_, *table_);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    break;
                }
                case Version::CpuN: {
                    accel::ThreadPool& pool = context.pool(variant.threads);
                    const auto startTime = Clock::now();
                    result = accel::computeEtaMultiThread(couriers_, orders_, *table_, pool);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    break;
                }
                case Version::GpuKernel: {
                    if (!gpuReady_) return fail(name() + ": GPU no disponible");
                    double kernelMs = 0.0;
                    std::string errorMessage = device_.runEta(aos_, &kernelMs);
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
                    double couriersMs = 0.0;
                    double ordersMs = 0.0;
                    double kernelMs = 0.0;
                    double downloadMs = 0.0;
                    const auto startTime = Clock::now();
                    std::string errorMessage = device_.uploadTable(table_->cells().data(), table_->nodeCount(), &tableMs);
                    if (errorMessage.empty()) {
                        errorMessage = aos_ ? device_.uploadCouriersAoS(records_, &couriersMs)
                                            : device_.uploadCouriersSoA(couriers_, &couriersMs);
                    }
                    if (errorMessage.empty()) errorMessage = device_.uploadOrders(orders_, &ordersMs);
                    if (errorMessage.empty()) errorMessage = device_.runEta(aos_, &kernelMs);
                    if (errorMessage.empty()) errorMessage = device_.download(result, &downloadMs);
                    if (errorMessage.empty()) errorMessage = accel::deviceSynchronize();
                    if (!errorMessage.empty()) return fail(name() + " gpu_e2e: " + errorMessage);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    runResult.timing.h2dMs = tableMs + couriersMs + ordersMs;
                    runResult.timing.kernelMs = kernelMs;
                    runResult.timing.d2hMs = downloadMs;
                    break;
                }
            }
        } catch (const std::exception& error) {
            return fail(name() + ": " + error.what());
        }
        runResult.correct = (result == reference_);
        return runResult;
    }

    // Bytes del modelo de trafico por par: lee los campos del courier, escribe 4 bytes de ETA (la tabla y los
    // pedidos se reutilizan y no cuentan). Utiles = 8 (nodo + libre-desde) + 4. Movidos: SoA = utiles; AoS = el
    // registro entero (32) + 4, porque cada hilo hace pasar por el bus el registro de 32 bytes para usar 8.
    std::string describe(const SummaryStats& stats) const override {
        if (stats.medianMs <= 0.0) return "";
        const double pairs = static_cast<double>(stats.size) * static_cast<double>(orderCountFor(stats.size));
        const double usefulBytes = pairs * 12.0;
        const double movedBytes = pairs * (aos_ ? 36.0 : 12.0);
        const double seconds = stats.medianMs * 1e-3;
        char description[200];
        std::snprintf(description, sizeof description, "%zu x %zu pares, utiles %.1f MB (%.2f GB/s), movidos (modelo) %.1f MB (%.2f GB/s)",
                      orderCountFor(stats.size), stats.size, usefulBytes / 1e6, usefulBytes / seconds / 1e9,
                      movedBytes / 1e6, movedBytes / seconds / 1e9);
        return description;
    }

private:
    static RunResult fail(const std::string& message) {
        RunResult runResult;
        runResult.ok = false;
        runResult.error = message;
        return runResult;
    }

    std::string appName_;
    bool aos_;
    std::shared_ptr<const accel::TravelTimeTable> table_;
    accel::CourierColumns couriers_;
    std::vector<accel::CourierRecord> records_;
    accel::OrderColumns orders_;
    std::vector<std::int32_t> reference_;
    accel::EtaDevice device_;
    bool tableOnDevice_ = false;
    bool gpuReady_ = false;
    bool ready_ = false;
};

class EtaBatchSoaBench : public EtaBatchBench {
public:
    EtaBatchSoaBench() : EtaBatchBench("eta_batch_soa", false) {}
};

class EtaBatchAosBench : public EtaBatchBench {
public:
    EtaBatchAosBench() : EtaBatchBench("eta_batch_aos", true) {}
};

BENCH_REGISTER_APP("eta_batch_soa", EtaBatchSoaBench);
BENCH_REGISTER_APP("eta_batch_aos", EtaBatchAosBench);

}  // namespace

}  // namespace bench
