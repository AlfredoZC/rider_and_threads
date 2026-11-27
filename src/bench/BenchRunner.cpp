#include "bench/BenchRunner.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <map>
#include <ostream>
#include <sstream>
#include <tuple>

namespace bench {

const char* versionName(Version version) {
    switch (version) {
        case Version::Cpu1: return "cpu1";
        case Version::CpuN: return "cpuN";
        case Version::GpuKernel: return "gpu_kernel";
        case Version::GpuE2e: return "gpu_e2e";
    }
    return "unknown";
}

bool parseVersion(std::string_view text, Version& parsedVersion) {
    for (Version candidate : {Version::Cpu1, Version::CpuN, Version::GpuKernel, Version::GpuE2e}) {
        if (text == versionName(candidate)) {
            parsedVersion = candidate;
            return true;
        }
    }
    return false;
}

std::vector<Variant> variantsFor(const BenchApp& app, bool gpuAvailable, const std::vector<unsigned>& threadSweep) {
    std::vector<Variant> variants;
    for (Version version : app.versions(gpuAvailable)) {
        if (isGpu(version) && !gpuAvailable) {
            continue;
        }
        if (version == Version::CpuN) {
            for (unsigned threadCount : threadSweep) {
                variants.push_back(Variant{version, threadCount});
            }
        } else {
            variants.push_back(Variant{version, 1});
        }
    }
    return variants;
}

std::vector<Variant> rotatedOrder(const std::vector<Variant>& baseOrder, std::size_t repetition) {
    std::vector<Variant> rotated;
    if (baseOrder.empty()) {
        return rotated;
    }
    rotated.reserve(baseOrder.size());
    const std::size_t shift = repetition % baseOrder.size();
    for (std::size_t position = 0; position < baseOrder.size(); ++position) {
        rotated.push_back(baseOrder[(position + shift) % baseOrder.size()]);
    }
    return rotated;
}

namespace {

Row makeRow(const std::string& appName, const Variant& variant, std::size_t size, std::size_t repetition,
            std::uint32_t seed, const RunResult& runResult) {
    Row row;
    row.application = appName;
    row.version = versionName(variant.version);
    row.size = size;
    row.rep = repetition;
    row.ms = runResult.timing.ms;
    row.correct = runResult.correct;
    row.threads = variant.threads;
    row.seed = seed;
    row.h2dMs = runResult.timing.h2dMs;
    row.kernelMs = runResult.timing.kernelMs;
    row.d2hMs = runResult.timing.d2hMs;
    return row;
}

// Las aplicaciones no lanzan por contrato, pero un pool que no puede crear hilos o un bad_alloc
// no deben tumbar el proceso: se convierten en una corrida con error.
RunResult safeRun(BenchApp& app, const Variant& variant, BenchContext& context) {
    try {
        return app.run(variant, context);
    } catch (const std::exception& caught) {
        RunResult failed;
        failed.ok = false;
        failed.error = std::string("excepcion: ") + caught.what();
        return failed;
    }
}

std::string safePrepare(BenchApp& app, std::size_t size, std::uint32_t seed, BenchContext& context) {
    try {
        return app.prepare(size, seed, context);
    } catch (const std::exception& caught) {
        return std::string("excepcion: ") + caught.what();
    }
}

// Crea los pools de las versiones cpuN antes de calentar y medir. Si el sistema no puede crear los
// hilos (limite de procesos o de memoria), esa version se descarta con un error claro.
std::vector<Variant> withPools(const std::vector<Variant>& variants, BenchContext& context, std::ostream& log,
                               std::size_t& errorCount) {
    std::vector<Variant> usable;
    for (const Variant& variant : variants) {
        if (variant.version == Version::CpuN) {
            try {
                context.pool(variant.threads);
            } catch (const std::exception& caught) {
                log << "ERROR no se pudo crear un pool de " << variant.threads << " hilos: " << caught.what() << '\n';
                ++errorCount;
                continue;
            }
        }
        usable.push_back(variant);
    }
    return usable;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t count = values.size();
    return count % 2 == 1 ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) / 2.0;
}

// Tiempos de todas las repeticiones de una (version, hilos).
struct Samples {
    std::vector<double> totalMs;
    std::vector<double> h2dMs;
    std::vector<double> kernelMs;
    std::vector<double> d2hMs;
};

}  // namespace

BenchOutcome runBench(const BenchOptions& options, const std::vector<BenchApp*>& apps, BenchContext& context,
                      RowSink& sink, std::ostream& log) {
    BenchOutcome outcome;
    for (BenchApp* app : apps) {
        std::vector<std::size_t> sizes = options.sizes;
        if (sizes.empty()) {
            sizes = app->defaultSizes(context.gpu);
        }
        const std::vector<Variant> baseOrder =
            withPools(variantsFor(*app, context.gpu.available, options.threads), context, log, outcome.errors);
        for (std::size_t size : sizes) {
            const std::string errorMessage = safePrepare(*app, size, options.seed, context);
            if (!errorMessage.empty()) {
                log << "[" << app->name() << "] size=" << size << ": " << errorMessage << '\n';
                ++outcome.errors;
                continue;
            }
            log << "[" << app->name() << "] size=" << size << " (" << baseOrder.size() << " versiones x "
                << options.reps << " repeticiones)\n";

            // Calentamiento: una corrida de cada version, sin registrar (la primera llamada a CUDA
            // crea el contexto; las de CPU llenan caches y paginas).
            for (const Variant& variant : baseOrder) {
                const RunResult warmup = safeRun(*app, variant, context);
                if (!warmup.ok) {
                    log << "  calentamiento " << versionName(variant.version) << " x" << variant.threads << ": "
                        << warmup.error << '\n';
                }
            }

            std::map<std::pair<std::string, unsigned>, Samples> samplesByVariant;
            for (std::size_t repetition = 0; repetition < options.reps; ++repetition) {
                for (const Variant& variant : rotatedOrder(baseOrder, repetition)) {
                    const RunResult runResult = safeRun(*app, variant, context);
                    if (!runResult.ok) {
                        log << "  ERROR " << app->name() << " " << versionName(variant.version) << " x"
                            << variant.threads << " size=" << size << " rep=" << repetition << ": "
                            << runResult.error << '\n';
                        ++outcome.errors;
                        continue;
                    }
                    sink.write(makeRow(app->name(), variant, size, repetition, options.seed, runResult));
                    ++outcome.rows;
                    if (!runResult.correct) {
                        ++outcome.incorrect;
                        log << "  INCORRECTO " << app->name() << " " << versionName(variant.version) << " x"
                            << variant.threads << " size=" << size << " rep=" << repetition << '\n';
                    }
                    Samples& samples = samplesByVariant[{versionName(variant.version), variant.threads}];
                    samples.totalMs.push_back(runResult.timing.ms);
                    if (runResult.timing.h2dMs) samples.h2dMs.push_back(*runResult.timing.h2dMs);
                    if (runResult.timing.kernelMs) samples.kernelMs.push_back(*runResult.timing.kernelMs);
                    if (runResult.timing.d2hMs) samples.d2hMs.push_back(*runResult.timing.d2hMs);
                }
            }
            for (const Variant& variant : baseOrder) {
                const auto found = samplesByVariant.find({versionName(variant.version), variant.threads});
                if (found == samplesByVariant.end() || found->second.totalMs.empty()) {
                    continue;
                }
                const Samples& samples = found->second;
                SummaryLine line;
                line.application = app->name();
                line.size = size;
                line.version = versionName(variant.version);
                line.threads = variant.threads;
                line.samples = samples.totalMs.size();
                line.medianMs = median(samples.totalMs);
                line.minMs = *std::min_element(samples.totalMs.begin(), samples.totalMs.end());
                line.maxMs = *std::max_element(samples.totalMs.begin(), samples.totalMs.end());
                SummaryStats stats;
                stats.size = size;
                stats.version = variant.version;
                stats.threads = variant.threads;
                stats.medianMs = line.medianMs;
                if (!samples.h2dMs.empty()) stats.medianH2dMs = median(samples.h2dMs);
                if (!samples.kernelMs.empty()) stats.medianKernelMs = median(samples.kernelMs);
                if (!samples.d2hMs.empty()) stats.medianD2hMs = median(samples.d2hMs);
                line.note = app->describe(stats);
                outcome.summary.push_back(line);
            }
        }
    }
    return outcome;
}

std::string formatSummary(const std::vector<SummaryLine>& lines) {
    std::ostringstream table;
    char formattedLine[256];
    std::snprintf(formattedLine, sizeof formattedLine, "%-14s %12s %-11s %7s %4s %12s %12s %12s  %s\n", "application",
                  "size", "version", "threads", "n", "median_ms", "min_ms", "max_ms", "derived (from the median)");
    table << formattedLine;
    for (const SummaryLine& line : lines) {
        std::snprintf(formattedLine, sizeof formattedLine, "%-14s %12zu %-11s %7u %4zu %12.6f %12.6f %12.6f  %s\n",
                      line.application.c_str(), line.size, line.version.c_str(), line.threads, line.samples,
                      line.medianMs, line.minMs, line.maxMs, line.note.c_str());
        table << formattedLine;
    }
    return table.str();
}

BenchOutcome runOnce(BenchApp& app, std::size_t size, std::uint32_t seed, const std::vector<Variant>& variants,
                     BenchContext& context, std::ostream& output, std::ostream& log) {
    BenchOutcome outcome;
    const std::string errorMessage = safePrepare(app, size, seed, context);
    if (!errorMessage.empty()) {
        log << "[" << app.name() << "] size=" << size << ": " << errorMessage << '\n';
        ++outcome.errors;
        return outcome;
    }
    for (const Variant& variant : withPools(variants, context, log, outcome.errors)) {
        const RunResult runResult = safeRun(app, variant, context);
        if (!runResult.ok) {
            log << "ERROR " << app.name() << " " << versionName(variant.version) << ": " << runResult.error << '\n';
            ++outcome.errors;
            continue;
        }
        ++outcome.rows;
        if (!runResult.correct) {
            ++outcome.incorrect;
        }
        output << csvRow(makeRow(app.name(), variant, size, 0, seed, runResult)) << '\n';
    }
    return outcome;
}

}  // namespace bench
