// Calibracion de copias: cuanto cuesta mover bytes. Sirve para el analisis de la seccion 2.5 (cuando
// los overheads de copia dominan) y para el techo de ancho de banda del roofline.
//
//   cpu1        memcpy de un hilo (referencia)
//   cpuN        memcpy particionado entre N hilos del pool propio
//   gpu_kernel  copia dispositivo -> dispositivo con un kernel (cudaEvent): ancho de banda de la VRAM
//   gpu_e2e     ida y vuelta H2D + D2H con memoria paginable (steady_clock + sync; h2d/d2h con cudaEvent)
//
// size = bytes. Correcto = el buffer de destino es identico al de origen.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "accel/Calib.h"
#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

namespace bench {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kMaxBytes = std::size_t{1} << 31;  // 2 GiB

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

// splitmix64: datos pseudoaleatorios deterministas segun la semilla, sin depender de la libreria.
void fillRandom(std::vector<unsigned char>& buffer, std::uint32_t seed) {
    std::uint64_t state = seed;
    std::size_t byteIndex = 0;
    for (; byteIndex + 8 <= buffer.size(); byteIndex += 8) {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t mixed = state;
        mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
        mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
        mixed ^= mixed >> 31;
        std::memcpy(buffer.data() + byteIndex, &mixed, 8);
    }
    for (; byteIndex < buffer.size(); ++byteIndex) {
        buffer[byteIndex] = static_cast<unsigned char>(state >> (8 * (byteIndex % 8)));
    }
}

class CalibCopy : public BenchApp {
public:
    std::string name() const override { return "calib_copy"; }

    std::vector<std::size_t> defaultSizes(const accel::GpuInfo& gpu) const override {
        std::vector<std::size_t> sizes{std::size_t{4} << 10, std::size_t{64} << 10, std::size_t{1} << 20,
                                       std::size_t{16} << 20, std::size_t{64} << 20, std::size_t{256} << 20};
        if (gpu.available) {
            // Dos buffers en el dispositivo: se deja un cuarto de la memoria libre como margen.
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            if (accel::gpuMemory(&freeBytes, &totalBytes).empty()) {
                const std::size_t budget = freeBytes / 4 * 3;
                while (sizes.size() > 5 && sizes.back() > budget / 2) {
                    sizes.pop_back();
                }
            }
        }
        return sizes;
    }

    std::size_t maxSize() const override { return kMaxBytes; }

    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1, Version::CpuN};
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }

    std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) override {
        // Todo lo anterior se libera primero: la memoria libre de la GPU que se mira abajo no debe
        // incluir los buffers del tamano anterior, y un prepare que falla no deja nada reservado.
        releaseAll();
        if (size == 0 || size > kMaxBytes) {
            return "calib_copy: tamano fuera de rango (1 .. " + std::to_string(kMaxBytes) + " bytes)";
        }
        try {
            sourceBytes_.assign(size, 0);
            destinationBytes_.assign(size, 0);
        } catch (const std::bad_alloc&) {
            releaseAll();
            return "calib_copy: sin memoria en el host para 2 buffers de " + std::to_string(size) + " bytes";
        }
        fillRandom(sourceBytes_, seed);
        if (context.gpu.available) {
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            std::string errorMessage = accel::gpuMemory(&freeBytes, &totalBytes);
            if (!errorMessage.empty()) {
                releaseAll();
                return "calib_copy: " + errorMessage;
            }
            if (size > freeBytes / 2) {
                releaseAll();
                return "calib_copy: sin memoria en la GPU: se necesitan 2 buffers de " + std::to_string(size) +
                       " bytes y quedan " + std::to_string(freeBytes) + " libres";
            }
            errorMessage = sourceOnDevice_.allocate(size);
            if (errorMessage.empty()) errorMessage = destinationOnDevice_.allocate(size);
            if (errorMessage.empty()) errorMessage = sourceOnDevice_.upload(sourceBytes_.data(), size, nullptr);  // la fuente queda en la GPU
            if (!errorMessage.empty()) {
                releaseAll();
                return "calib_copy: " + errorMessage;
            }
            gpuReady_ = true;
        }
        size_ = size;
        return "";
    }

    RunResult run(const Variant& variant, BenchContext& context) override {
        RunResult runResult;
        if (size_ == 0) {
            return fail("calib_copy: run sin prepare valido");
        }
        switch (variant.version) {
            case Version::Cpu1: {
                std::memset(destinationBytes_.data(), 0, size_);
                const auto startTime = Clock::now();
                std::memcpy(destinationBytes_.data(), sourceBytes_.data(), size_);
                const auto stopTime = Clock::now();
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                break;
            }
            case Version::CpuN: {
                accel::ThreadPool& pool = context.pool(variant.threads);
                std::memset(destinationBytes_.data(), 0, size_);
                unsigned char* destinationData = destinationBytes_.data();
                const unsigned char* sourceData = sourceBytes_.data();
                const auto startTime = Clock::now();
                pool.parallelFor(size_, [destinationData, sourceData](std::size_t chunkBegin, std::size_t chunkEnd) {
                    std::memcpy(destinationData + chunkBegin, sourceData + chunkBegin, chunkEnd - chunkBegin);
                });
                const auto stopTime = Clock::now();
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                break;
            }
            case Version::GpuE2e: {
                if (!gpuReady_) return fail("calib_copy: GPU no disponible");
                std::memset(destinationBytes_.data(), 0, size_);
                double hostToDeviceMs = 0.0;
                double deviceToHostMs = 0.0;
                const auto startTime = Clock::now();
                std::string errorMessage = sourceOnDevice_.upload(sourceBytes_.data(), size_, &hostToDeviceMs);
                if (errorMessage.empty()) errorMessage = sourceOnDevice_.download(destinationBytes_.data(), size_, &deviceToHostMs);
                if (errorMessage.empty()) errorMessage = accel::deviceSynchronize();
                const auto stopTime = Clock::now();
                if (!errorMessage.empty()) return fail("calib_copy gpu_e2e: " + errorMessage);
                runResult.timing.ms = elapsedMs(startTime, stopTime);
                runResult.timing.h2dMs = hostToDeviceMs;
                runResult.timing.d2hMs = deviceToHostMs;
                break;
            }
            case Version::GpuKernel: {
                if (!gpuReady_) return fail("calib_copy: GPU no disponible");
                std::string errorMessage = destinationOnDevice_.clear();
                double kernelMs = 0.0;
                if (errorMessage.empty()) errorMessage = sourceOnDevice_.copyTo(destinationOnDevice_, &kernelMs);
                if (!errorMessage.empty()) return fail("calib_copy gpu_kernel: " + errorMessage);
                runResult.timing.ms = kernelMs;
                runResult.timing.kernelMs = kernelMs;
                // Verificacion fuera de la region medida: el resultado vuelve al host.
                std::memset(destinationBytes_.data(), 0, size_);
                errorMessage = destinationOnDevice_.download(destinationBytes_.data(), size_, nullptr);
                if (!errorMessage.empty()) return fail("calib_copy gpu_kernel (verificacion): " + errorMessage);
                break;
            }
        }
        runResult.correct = std::memcmp(destinationBytes_.data(), sourceBytes_.data(), size_) == 0;
        return runResult;
    }

    std::string describe(const SummaryStats& stats) const override {
        char description[200];
        auto gbs = [](double bytes, double elapsedMs) { return elapsedMs > 0.0 ? bytes / (elapsedMs * 1e-3) / 1e9 : 0.0; };
        const double bytes = static_cast<double>(stats.size);
        switch (stats.version) {
            case Version::Cpu1:
            case Version::CpuN:
                std::snprintf(description, sizeof description, "%.2f GB/stats copied (memcpy)", gbs(bytes, stats.medianMs));
                break;
            case Version::GpuKernel:
                std::snprintf(description, sizeof description, "%.2f GB/stats copied = %.2f GB/stats of VRAM traffic (read+write)",
                              gbs(bytes, stats.medianMs), gbs(2.0 * bytes, stats.medianMs));
                break;
            case Version::GpuE2e:
                std::snprintf(description, sizeof description, "H2D %.2f GB/stats, D2H %.2f GB/stats, round trip %.2f GB/stats (pageable)",
                              stats.medianH2dMs ? gbs(bytes, *stats.medianH2dMs) : 0.0,
                              stats.medianD2hMs ? gbs(bytes, *stats.medianD2hMs) : 0.0, gbs(2.0 * bytes, stats.medianMs));
                break;
        }
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
        sourceBytes_.clear();
        sourceBytes_.shrink_to_fit();
        destinationBytes_.clear();
        destinationBytes_.shrink_to_fit();
        sourceOnDevice_.release();
        destinationOnDevice_.release();
    }

    std::size_t size_ = 0;
    std::vector<unsigned char> sourceBytes_;
    std::vector<unsigned char> destinationBytes_;
    accel::DeviceBuffer sourceOnDevice_;
    accel::DeviceBuffer destinationOnDevice_;
    bool gpuReady_ = false;
};

BENCH_REGISTER_APP("calib_copy", CalibCopy);

}  // namespace

}  // namespace bench
