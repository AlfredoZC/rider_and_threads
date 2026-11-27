#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "Config.h"
#include "accel/GpuInfo.h"
#include "accel/ThreadPool.h"

namespace bench {

// Las cuatro versiones del enunciado (seccion 2.1). Una aplicacion puede omitir las que no aplican.
enum class Version { Cpu1, CpuN, GpuKernel, GpuE2e };

const char* versionName(Version version);
bool parseVersion(std::string_view text, Version& parsedVersion);
inline bool isGpu(Version version) { return version == Version::GpuKernel || version == Version::GpuE2e; }

// Una version concreta a medir. `threads`: 1 para cpu1 y las GPU, N para cpuN.
struct Variant {
    Version version = Version::Cpu1;
    unsigned threads = 1;
};

struct Timing {
    double ms = 0.0;                 // tiempo de la version (el reloj de cada una esta en el ADR 0003)
    std::optional<double> h2dMs;     // cudaEvent
    std::optional<double> kernelMs;  // cudaEvent
    std::optional<double> d2hMs;     // cudaEvent
};

struct RunResult {
    bool ok = true;       // false: no se pudo ejecutar (p. ej. sin memoria); `error` dice por que
    std::string error;
    Timing timing;
    bool correct = false;  // resultado igual al de la referencia (cpu1)
};

// Medianas de las repeticiones de una (tamano, version, hilos), para que la aplicacion derive metricas.
struct SummaryStats {
    std::size_t size = 0;
    Version version = Version::Cpu1;
    unsigned threads = 1;
    double medianMs = 0.0;
    std::optional<double> medianH2dMs;
    std::optional<double> medianKernelMs;
    std::optional<double> medianD2hMs;
};

// Lo que el runner le da a cada aplicacion: el config, la GPU y pools de CPU ya creados
// (crear hilos dentro de la region medida falsearia los tiempos).
class BenchContext {
public:
    Config config;
    accel::GpuInfo gpu;

    accel::ThreadPool& pool(unsigned threads) {
        auto& slot = pools_[threads];
        if (!slot) {
            slot = std::make_unique<accel::ThreadPool>(threads);
        }
        return *slot;
    }

private:
    std::map<unsigned, std::unique_ptr<accel::ThreadPool>> pools_;
};

// Contrato de una aplicacion medible. Una aplicacion nueva solo agrega su archivo en src/bench/ y
// se registra con BENCH_REGISTER_APP (BenchRegistry.h).
class BenchApp {
public:
    virtual ~BenchApp() = default;

    virtual std::string name() const = 0;  // identificador generado por el codigo, va al CSV
    // Tamanos del barrido (al menos 5, dos ordenes de magnitud). Puede depender de la GPU (memoria libre).
    virtual std::vector<std::size_t> defaultSizes(const accel::GpuInfo& gpu) const = 0;
    virtual std::size_t maxSize() const = 0;  // cota para --size / --sizes
    // Versiones que implementa; las GPU solo se listan si gpuAvailable.
    virtual std::vector<Version> versions(bool gpuAvailable) const = 0;
    // Construye la entrada con la semilla (misma para todas las versiones) y la referencia.
    // Devuelve "" si todo bien o un mensaje de error (tamano invalido, sin memoria...).
    virtual std::string prepare(std::size_t size, std::uint32_t seed, BenchContext& context) = 0;
    // Ejecuta una vez, mide y compara contra la referencia. No lanza ni aborta.
    virtual RunResult run(const Variant& variant, BenchContext& context) = 0;
    // Metrica derivada para el resumen (p. ej. "12.3 GB/s"), calculada con las medianas; "" si no hay.
    virtual std::string describe(const SummaryStats& stats) const {
        (void)stats;
        return "";
    }
};

}  // namespace bench
