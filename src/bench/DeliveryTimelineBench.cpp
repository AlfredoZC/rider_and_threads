// Linea de tiempo global de entregas: fusion de las listas ordenadas de 2.000 couriers (merge con co-rank).
//
//   cpu1        arbol de fusiones de a pares, cada una secuencial, + entregas por hora contando
//   cpuN        igual, pero cada ronda reparte la salida entre los hilos del pool propio (co-rank)
//   gpu_kernel  un lanzamiento por ronda (todos los pares a la vez) + un hilo por hora, con cudaEvent
//   gpu_e2e     subir registros y comienzos de lista, fusionar, contar y traer la linea de tiempo y las cuentas
//
// size = cantidad total de entregas, repartidas en 2.000 couriers (algunos sin ninguna entrega; el reparto
// es desigual). Los tiempos van de 0 a 24 h simuladas redondeados al segundo, para que haya muchos empates
// entre couriers. Correcto = la linea de tiempo (tiempo Y orden de empates, comparando el pedido) y las
// entregas por hora identicas a las de std::stable_sort sobre todas las listas concatenadas.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <new>
#include <string>
#include <vector>

#include "accel/Calib.h"
#include "accel/DeliveryTimeline.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kCourierCount = 2000;
constexpr std::size_t kMaxDeliveries = 30000000;  // host: 4 copias de 240 MB; GPU: 3 buffers de 240 MB
constexpr std::int32_t kDayMs = 24 * accel::kMillisecondsPerHour;

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

// Reparte `size` entregas en kCourierCount listas (desiguales; los couriers con indice terminado en 9 no tienen
// ninguna) y arma cada lista ordenada por tiempo. Los pedidos se numeran de 1 en adelante a lo largo de las listas.
void fillLists(std::size_t size, std::uint32_t seed, std::vector<accel::DeliveryRecord>& records, std::vector<int>& listStart) {
    std::uint64_t state = seed;
    std::vector<std::size_t> countPerCourier(kCourierCount, 0);
    for (std::size_t delivery = 0; delivery < size; ++delivery) {
        const double skewed = nextUniform(state);
        std::size_t courier = static_cast<std::size_t>(skewed * skewed * static_cast<double>(kCourierCount));
        if (courier >= kCourierCount) courier = kCourierCount - 1;
        if (courier % 10 == 9) courier -= 1;  // sin entregas
        ++countPerCourier[courier];
    }
    records.assign(size, accel::DeliveryRecord());
    listStart.assign(1, 0);
    std::size_t next = 0;
    for (std::size_t courier = 0; courier < kCourierCount; ++courier) {
        const std::size_t listBegin = next;
        for (std::size_t item = 0; item < countPerCourier[courier]; ++item) {
            const std::int32_t seconds = static_cast<std::int32_t>(nextUniform(state) * (kDayMs / 1000));
            records[next++].deliveredAtMs = seconds * 1000;
        }
        std::sort(records.begin() + static_cast<std::ptrdiff_t>(listBegin), records.begin() + static_cast<std::ptrdiff_t>(next),
                  [](const accel::DeliveryRecord& left, const accel::DeliveryRecord& right) { return left.deliveredAtMs < right.deliveredAtMs; });
        listStart.push_back(static_cast<int>(next));
    }
    for (std::size_t position = 0; position < size; ++position) {
        records[position].orderId = static_cast<std::int32_t>(position + 1);
    }
}

bool sameTimeline(const std::vector<accel::DeliveryRecord>& left, const std::vector<accel::DeliveryRecord>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t position = 0; position < left.size(); ++position) {
        if (left[position].deliveredAtMs != right[position].deliveredAtMs || left[position].orderId != right[position].orderId) {
            return false;
        }
    }
    return true;
}

class DeliveryTimelineBench : public BenchApp {
public:
    std::string name() const override { return "delivery_timeline"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override {
        return {1000, 10000, 100000, 1000000, 10000000};
    }

    std::size_t maxSize() const override { return kMaxDeliveries; }

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
        if (size == 0 || size > kMaxDeliveries) {
            return "delivery_timeline: tamano fuera de rango (1 .. " + std::to_string(kMaxDeliveries) + ")";
        }
        try {
            fillLists(size, seed, records_, listStart_);
            referenceTimeline_ = records_;
        } catch (const std::bad_alloc&) {
            releaseAll();
            return "delivery_timeline: sin memoria en el host para " + std::to_string(size) + " entregas";
        }
        // Referencia independiente de la fusion: orden estable por tiempo de todas las listas concatenadas.
        std::stable_sort(referenceTimeline_.begin(), referenceTimeline_.end(),
                         [](const accel::DeliveryRecord& left, const accel::DeliveryRecord& right) {
                             return left.deliveredAtMs < right.deliveredAtMs;
                         });
        referencePerHour_ = accel::deliveriesPerBucketCpu(referenceTimeline_, accel::kMillisecondsPerHour);
        plan_ = accel::makeTimelinePlan(records_, listStart_, accel::kMillisecondsPerHour);
        if (context.gpu.available) {
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            std::string errorMessage = accel::gpuMemory(&freeBytes, &totalBytes);
            const std::size_t neededBytes = size * sizeof(accel::DeliveryRecord) * 3 + (4u << 20);
            if (errorMessage.empty() && neededBytes > freeBytes / 4 * 3) {
                errorMessage = "sin memoria en la GPU: se necesitan " + std::to_string(neededBytes) + " bytes y quedan " +
                               std::to_string(freeBytes) + " libres";
            }
            if (errorMessage.empty()) errorMessage = device_.allocate(size, plan_);
            if (errorMessage.empty()) errorMessage = device_.upload(records_.data(), size, plan_, nullptr);  // datos residentes
            if (!errorMessage.empty()) {
                releaseAll();
                return "delivery_timeline: " + errorMessage;
            }
            gpuReady_ = true;
        }
        size_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (size_ == 0) {
            return fail("delivery_timeline: run sin prepare valido");
        }
        accel::DeliveryTimelineResult result;
        switch (variant.version) {
            case Version::Cpu1: {
                std::vector<accel::DeliveryRecord> input = records_;  // la copia de la entrada queda fuera de la medicion
                std::vector<int> listStart = listStart_;
                const auto startTime = Clock::now();
                result.timeline = accel::mergeCourierListsCpu1(std::move(input), std::move(listStart));
                result.deliveriesPerBucket = accel::deliveriesPerBucketCpu(result.timeline, accel::kMillisecondsPerHour);
                runResult.timing.ms = elapsedMs(startTime, Clock::now());
                break;
            }
            case Version::CpuN: {
                accel::ThreadPool& pool = context.pool(variant.threads);
                std::vector<accel::DeliveryRecord> input = records_;
                std::vector<int> listStart = listStart_;
                const auto startTime = Clock::now();
                result.timeline = accel::mergeCourierListsCpuN(std::move(input), std::move(listStart), pool);
                result.deliveriesPerBucket = accel::deliveriesPerBucketCpu(result.timeline, accel::kMillisecondsPerHour);
                runResult.timing.ms = elapsedMs(startTime, Clock::now());
                break;
            }
            case Version::GpuKernel: {
                if (!gpuReady_) return fail("delivery_timeline: GPU no disponible");
                double kernelMs = 0.0;
                std::string errorMessage = device_.run(&kernelMs);
                result.timeline.assign(size_, accel::DeliveryRecord());
                result.deliveriesPerBucket.assign(static_cast<std::size_t>(plan_.bucketCount), 0);
                // Verificacion fuera de la region medida: el resultado vuelve al host.
                if (errorMessage.empty()) errorMessage = device_.download(result.timeline.data(), result.deliveriesPerBucket.data(), nullptr);
                if (!errorMessage.empty()) return fail("delivery_timeline gpu_kernel: " + errorMessage);
                runResult.timing.ms = kernelMs;
                runResult.timing.kernelMs = kernelMs;
                break;
            }
            case Version::GpuE2e: {
                if (!gpuReady_) return fail("delivery_timeline: GPU no disponible");
                double hostToDeviceMs = 0.0;
                double kernelMs = 0.0;
                double deviceToHostMs = 0.0;
                result.timeline.assign(size_, accel::DeliveryRecord());
                const auto startTime = Clock::now();
                const accel::TimelinePlan plan = accel::makeTimelinePlan(records_, listStart_, accel::kMillisecondsPerHour);
                result.deliveriesPerBucket.assign(static_cast<std::size_t>(plan.bucketCount), 0);
                std::string errorMessage = device_.upload(records_.data(), size_, plan, &hostToDeviceMs);
                if (errorMessage.empty()) errorMessage = device_.run(&kernelMs);
                if (errorMessage.empty()) errorMessage = device_.download(result.timeline.data(), result.deliveriesPerBucket.data(), &deviceToHostMs);
                const auto stopTime = Clock::now();
                if (!errorMessage.empty()) return fail("delivery_timeline gpu_e2e: " + errorMessage);
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                runResult.timing.h2dMs = hostToDeviceMs;
                runResult.timing.kernelMs = kernelMs;
                runResult.timing.d2hMs = deviceToHostMs;
                break;
            }
        }
        runResult.correct = sameTimeline(result.timeline, referenceTimeline_) && result.deliveriesPerBucket == referencePerHour_;
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        char description[120];
        const double millionsPerSecond =
            stats.medianMs > 0.0 ? static_cast<double>(stats.size) / (stats.medianMs * 1e-3) / 1e6 : 0.0;
        std::snprintf(description, sizeof description, "%.1f M deliveries/s merged", millionsPerSecond);
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
        size_ = 0;
        gpuReady_ = false;
        records_.clear();
        records_.shrink_to_fit();
        referenceTimeline_.clear();
        referenceTimeline_.shrink_to_fit();
        listStart_.clear();
        referencePerHour_.clear();
        plan_ = accel::TimelinePlan();
        device_.release();
    }

    std::size_t size_ = 0;
    std::vector<accel::DeliveryRecord> records_;
    std::vector<int> listStart_;
    std::vector<accel::DeliveryRecord> referenceTimeline_;
    std::vector<int> referencePerHour_;
    accel::TimelinePlan plan_;
    accel::DeviceTimeline device_;
    bool gpuReady_ = false;
};

BENCH_REGISTER_APP("delivery_timeline", DeliveryTimelineBench);

}  // namespace

}  // namespace bench
