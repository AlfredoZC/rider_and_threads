// Pedidos en curso a lo largo del tiempo y su pico (scan inclusivo + reduccion max).
//
//   cpu1        suma acumulada y maximo en una pasada
//   cpuN        dos pasadas paralelas (totales por franja, offsets, scan por franja) con el pool propio
//   gpu_kernel  scan jerarquico Kogge-Stone (bloques de 512) + reduccion max del pico (cudaEvent)
//   gpu_e2e     subir los cambios, scan + pico, traer la curva y la clave del pico (steady_clock + cudaEvent)
//
// size = cantidad de eventos (+1 pedido creado, -1 entregado o rechazado). Los eventos son una secuencia
// determinista segun la semilla que sube y baja por olas y nunca deja la cuenta en negativo. Correcto =
// la curva de abiertos, el pico y el evento del pico identicos a los de la referencia (cpu1 en prepare).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <new>
#include <string>
#include <vector>

#include "accel/Calib.h"
#include "accel/OpenOrders.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxEvents = 100000000;  // 400 MB de cambios; la GPU guarda el doble mas el auxiliar

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

// splitmix64: uniforme en [0, 1), determinista segun el estado.
double nextUniform(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t mixed = state;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
    mixed ^= mixed >> 31;
    return static_cast<double>(mixed >> 11) / 9007199254740992.0;
}

// Caminata con olas: la probabilidad de un +1 oscila entre 0.1 y 0.9 (horas pico y horas flojas); con 0
// pedidos abiertos siempre toca un +1, asi que la cuenta nunca es negativa.
void fillChanges(std::vector<std::int32_t>& change, std::uint32_t seed) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kWavePeriodEvents = 50000.0;
    std::uint64_t state = seed;
    std::int32_t openOrders = 0;
    for (std::size_t eventIndex = 0; eventIndex < change.size(); ++eventIndex) {
        const double probabilityUp = 0.5 + 0.4 * std::sin(2.0 * kPi * static_cast<double>(eventIndex) / kWavePeriodEvents);
        const bool isCreation = openOrders == 0 || nextUniform(state) < probabilityUp;
        change[eventIndex] = isCreation ? 1 : -1;
        openOrders += change[eventIndex];
    }
}

class OrdersInProgress : public BenchApp {
public:
    std::string name() const override { return "orders_in_progress"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {1000, 10000, 100000, 1000000, 10000000};
    }

    std::size_t maxSize() const override { return kMaxEvents; }

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
        if (size == 0 || size > kMaxEvents) {
            return "orders_in_progress: tamano fuera de rango (1 .. " + std::to_string(kMaxEvents) + ")";
        }
        try {
            change_.assign(size, 0);
        } catch (const std::bad_alloc&) {
            releaseAll();
            return "orders_in_progress: sin memoria en el host para " + std::to_string(size) + " eventos";
        }
        fillChanges(change_, seed);
        reference_ = accel::openOrdersCpu1(change_);
        if (context.gpu.available) {
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            std::string errorMessage = accel::gpuMemory(&freeBytes, &totalBytes);
            const std::size_t neededBytes = size * sizeof(std::int32_t) * 2 + size / 100 * sizeof(std::int32_t) + (1u << 20);
            if (errorMessage.empty() && neededBytes > freeBytes / 4 * 3) {
                errorMessage = "sin memoria en la GPU: se necesitan " + std::to_string(neededBytes) + " bytes y quedan " +
                               std::to_string(freeBytes) + " libres";
            }
            if (errorMessage.empty()) errorMessage = device_.allocate(size);
            if (errorMessage.empty()) errorMessage = device_.upload(change_.data(), size, nullptr);  // datos residentes
            if (!errorMessage.empty()) {
                releaseAll();
                return "orders_in_progress: " + errorMessage;
            }
            gpuReady_ = true;
        }
        size_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (size_ == 0) {
            return fail("orders_in_progress: run sin prepare valido");
        }
        accel::OpenOrdersResult result;
        switch (variant.version) {
            case Version::Cpu1: {
                const auto startTime = Clock::now();
                result = accel::openOrdersCpu1(change_);
                runResult.timing.ms = elapsedMs(startTime, Clock::now());
                break;
            }
            case Version::CpuN: {
                accel::ThreadPool& pool = context.pool(variant.threads);
                const auto startTime = Clock::now();
                result = accel::openOrdersCpuN(change_, pool);
                runResult.timing.ms = elapsedMs(startTime, Clock::now());
                break;
            }
            case Version::GpuKernel: {
                if (!gpuReady_) return fail("orders_in_progress: GPU no disponible");
                double kernelMs = 0.0;
                std::string errorMessage = device_.run(&kernelMs);
                // Verificacion fuera de la region medida: el resultado vuelve al host.
                if (errorMessage.empty()) errorMessage = downloadResult(result, nullptr);
                if (!errorMessage.empty()) return fail("orders_in_progress gpu_kernel: " + errorMessage);
                runResult.timing.ms = kernelMs;
                runResult.timing.kernelMs = kernelMs;
                break;
            }
            case Version::GpuE2e: {
                if (!gpuReady_) return fail("orders_in_progress: GPU no disponible");
                double hostToDeviceMs = 0.0;
                double kernelMs = 0.0;
                double deviceToHostMs = 0.0;
                const auto startTime = Clock::now();
                std::string errorMessage = device_.upload(change_.data(), size_, &hostToDeviceMs);
                if (errorMessage.empty()) errorMessage = device_.run(&kernelMs);
                if (errorMessage.empty()) errorMessage = downloadResult(result, &deviceToHostMs);
                const auto stopTime = Clock::now();
                if (!errorMessage.empty()) return fail("orders_in_progress gpu_e2e: " + errorMessage);
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                runResult.timing.h2dMs = hostToDeviceMs;
                runResult.timing.kernelMs = kernelMs;
                runResult.timing.d2hMs = deviceToHostMs;
                break;
            }
        }
        runResult.correct = result.openOrders == reference_.openOrders &&
                            result.peakOpenOrders == reference_.peakOpenOrders && result.peakEvent == reference_.peakEvent;
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        char description[120];
        const double millionsPerSecond =
            stats.medianMs > 0.0 ? static_cast<double>(stats.size) / (stats.medianMs * 1e-3) / 1e6 : 0.0;
        std::snprintf(description, sizeof description, "%.1f M events/s", millionsPerSecond);
        return description;
    }

private:
    static RunResult fail(const std::string& message) {
        RunResult runResult;
        runResult.ok = false;
        runResult.error = message;
        return runResult;
    }

    std::string downloadResult(accel::OpenOrdersResult& result, double* elapsedMs) {
        result.openOrders.assign(size_, 0);
        unsigned long long peakKeyOnHost = 0;
        const std::string errorMessage = device_.download(result.openOrders.data(), &peakKeyOnHost, elapsedMs);
        if (errorMessage.empty()) {
            accel::applyPeakKey(peakKeyOnHost, result);
        }
        return errorMessage;
    }

    void releaseAll() {
        size_ = 0;
        gpuReady_ = false;
        change_.clear();
        change_.shrink_to_fit();
        reference_ = accel::OpenOrdersResult();
        device_.release();
    }

    std::size_t size_ = 0;
    std::vector<std::int32_t> change_;
    accel::OpenOrdersResult reference_;
    accel::DeviceOpenOrders device_;
    bool gpuReady_ = false;
};

BENCH_REGISTER_APP("orders_in_progress", OrdersInProgress);

}  // namespace

}  // namespace bench
