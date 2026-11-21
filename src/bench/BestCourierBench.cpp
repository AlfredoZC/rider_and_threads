// Aplicacion 08: elegir el mejor courier de cada pedido (argmin por fila de la matriz de ETA). size = cantidad
// de couriers (columnas); los pedidos (filas) salen de 50 -> 10, 150 -> 30, 500 -> 100, 2000 -> 300,
// 5000 -> 600 y, para cualquier otro tamano (p. ej. 777, que no es multiplo del bloque de 256), size / 8
// pedidos (minimo 1). La matriz de ETA es sintetica con semilla fija: valores chicos (muchos empates), ~5 %
// inalcanzables, una de cada cuatro filas con todos los valores iguales (empate total) y, en las demas, tres
// columnas forzadas al minimo 0 (empate triple). Los ids de los couriers no estan ordenados.
//
//   cpu1        un hilo recorre las filas
//   cpuN        cada hilo reduce su franja de columnas y despues se combinan los parciales
//   gpu_kernel  solo el kernel (cudaEvent): la matriz y los ids ya estan en la GPU
//   gpu_e2e     copia la matriz y los ids, kernel y trae una clave por pedido (steady_clock)
//
// Dos aplicaciones: `best_courier` con el arbol de reduccion convergente y `best_courier_divergent` con el que
// empareja vecinos (mide la diferencia). Correcto = el mismo courier y la misma ETA que cpu1 en cada pedido.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "accel/BestCourier.h"
#include "accel/Calib.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxCouriers = 200000;

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

class BestCourierBench : public BenchApp {
public:
    BestCourierBench(std::string appName, bool divergentTree) : appName_(std::move(appName)), divergent_(divergentTree) {}

    std::string name() const override { return appName_; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {50, 150, 500, 777, 2000, 5000};
    }

    std::size_t maxSize() const override { return kMaxCouriers; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1};
        if (!divergent_) supportedVersions.push_back(Version::CpuN);
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) override {
        ready_ = false;
        if (size == 0 || size > kMaxCouriers) {
            return name() + ": tamano fuera de rango (1 .. " + std::to_string(kMaxCouriers) + " couriers)";
        }
        try {
            orderCount_ = orderCountFor(size);
            std::mt19937 generator(seed);
            courierIds_.clear();
            for (std::size_t column = 0; column < size; ++column) {
                courierIds_.push_back(static_cast<std::int32_t>(1 + (column * 7919) % 1000003));  // distintos, desordenados
            }
            eta_.assign(orderCount_ * size, 0);
            for (std::size_t order = 0; order < orderCount_; ++order) {
                for (std::size_t column = 0; column < size; ++column) {
                    eta_[order * size + column] =
                        order % 4 == 3 ? 7  // fila de empate total
                                       : (generator() % 20 == 0 ? accel::kUnreachableMs
                                                                : static_cast<std::int32_t>(1 + generator() % 5000));
                }
                if (order % 4 != 3) {
                    for (int tie = 0; tie < 3; ++tie) eta_[order * size + generator() % size] = 0;  // empate triple en el minimo
                }
            }
            reference_ = accel::bestCourierSingleThread(eta_, courierIds_, orderCount_);
            if (context.gpu.available) {
                std::string errorMessage = device_.uploadEta(eta_.data(), orderCount_, size, nullptr);
                if (errorMessage.empty()) errorMessage = device_.uploadCourierIds(courierIds_.data(), size, nullptr);
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
        std::vector<accel::BestChoice> result;
        try {
            switch (variant.version) {
                case Version::Cpu1: {
                    const auto startTime = Clock::now();
                    result = accel::bestCourierSingleThread(eta_, courierIds_, orderCount_);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    break;
                }
                case Version::CpuN: {
                    accel::ThreadPool& pool = context.pool(variant.threads);
                    const auto startTime = Clock::now();
                    result = accel::bestCourierMultiThread(eta_, courierIds_, orderCount_, pool);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    break;
                }
                case Version::GpuKernel: {
                    if (!gpuReady_) return fail(name() + ": GPU no disponible");
                    double kernelMs = 0.0;
                    std::string errorMessage = device_.run(nullptr, orderCount_, courierIds_.size(), divergent_, &kernelMs);
                    if (!errorMessage.empty()) return fail(name() + " gpu_kernel: " + errorMessage);
                    runResult.timing.ms = kernelMs;
                    runResult.timing.kernelMs = kernelMs;
                    std::vector<unsigned long long> keys;
                    errorMessage = device_.download(keys, nullptr);  // verificacion, fuera de la region medida
                    if (!errorMessage.empty()) return fail(name() + " gpu_kernel (verificacion): " + errorMessage);
                    result = decode(keys);
                    break;
                }
                case Version::GpuE2e: {
                    if (!gpuReady_) return fail(name() + ": GPU no disponible");
                    double etaMs = 0.0;
                    double idsMs = 0.0;
                    double kernelMs = 0.0;
                    double downloadMs = 0.0;
                    std::vector<unsigned long long> keys;
                    const auto startTime = Clock::now();
                    std::string errorMessage = device_.uploadEta(eta_.data(), orderCount_, courierIds_.size(), &etaMs);
                    if (errorMessage.empty()) errorMessage = device_.uploadCourierIds(courierIds_.data(), courierIds_.size(), &idsMs);
                    if (errorMessage.empty()) errorMessage = device_.run(nullptr, orderCount_, courierIds_.size(), divergent_, &kernelMs);
                    if (errorMessage.empty()) errorMessage = device_.download(keys, &downloadMs);
                    if (errorMessage.empty()) errorMessage = accel::deviceSynchronize();
                    if (!errorMessage.empty()) return fail(name() + " gpu_e2e: " + errorMessage);
                    result = decode(keys);
                    runResult.timing.ms = elapsedMs(startTime, Clock::now());
                    runResult.timing.h2dMs = etaMs + idsMs;
                    runResult.timing.kernelMs = kernelMs;
                    runResult.timing.d2hMs = downloadMs;
                    break;
                }
            }
        } catch (const std::exception& error) {
            return fail(name() + ": " + error.what());
        }
        runResult.correct = same(result, reference_);
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        if (stats.medianMs <= 0.0) return "";
        // Bytes que lee la reduccion: la matriz de ETA (4 B por celda) y los ids (4 B por columna).
        const double cells = static_cast<double>(stats.size) * static_cast<double>(orderCountFor(stats.size));
        const double readBytes = cells * 4.0;
        char description[160];
        std::snprintf(description, sizeof description, "%zu x %zu celdas, lee %.2f MB (%.2f GB/s)", orderCountFor(stats.size),
                      stats.size, readBytes / 1e6, readBytes / (stats.medianMs * 1e-3) / 1e9);
        return description;
    }

private:
    static RunResult fail(const std::string& message) {
        RunResult runResult;
        runResult.ok = false;
        runResult.error = message;
        return runResult;
    }

    static std::vector<accel::BestChoice> decode(const std::vector<unsigned long long>& keys) {
        std::vector<accel::BestChoice> choices;
        choices.reserve(keys.size());
        for (const unsigned long long key : keys) choices.push_back(accel::decodeBestKey(key));
        return choices;
    }

    static bool same(const std::vector<accel::BestChoice>& left, const std::vector<accel::BestChoice>& right) {
        if (left.size() != right.size()) return false;
        for (std::size_t order = 0; order < left.size(); ++order) {
            if (left[order].courierId != right[order].courierId || left[order].etaMs != right[order].etaMs) return false;
        }
        return true;
    }

    std::string appName_;
    bool divergent_;
    std::size_t orderCount_ = 0;
    std::vector<std::int32_t> courierIds_;
    std::vector<std::int32_t> eta_;
    std::vector<accel::BestChoice> reference_;
    accel::BestCourierDevice device_;
    bool gpuReady_ = false;
    bool ready_ = false;
};

class BestCourierConvergentBench : public BestCourierBench {
public:
    BestCourierConvergentBench() : BestCourierBench("best_courier", false) {}
};

class BestCourierDivergentBench : public BestCourierBench {
public:
    BestCourierDivergentBench() : BestCourierBench("best_courier_divergent", true) {}
};

BENCH_REGISTER_APP("best_courier", BestCourierConvergentBench);
BENCH_REGISTER_APP("best_courier_divergent", BestCourierDivergentBench);

}  // namespace

}  // namespace bench
