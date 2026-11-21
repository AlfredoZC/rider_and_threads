#include "bench/BenchApp.h"
#include "bench/BenchRunner.h"
#include "bench/BuildInfo.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

using bench::Variant;
using bench::Version;

// Aplicacion falsa: registra el orden exacto de las llamadas y puede fallar a pedido.
class FakeApp : public bench::BenchApp {
public:
    std::string name() const override { return "fake"; }
    std::vector<std::size_t> defaultSizes(const accel::GpuInfo&) const override { return {10, 20}; }
    std::size_t maxSize() const override { return 1000; }
    std::vector<Version> versions(bool gpuAvailable) const override {
        std::vector<Version> supportedVersions{Version::Cpu1, Version::CpuN};
        if (gpuAvailable) {
            supportedVersions.push_back(Version::GpuKernel);
            supportedVersions.push_back(Version::GpuE2e);
        }
        return supportedVersions;
    }
    std::string prepare(std::size_t size, std::uint32_t seed, bench::BenchContext&) override {
        prepared.push_back({size, seed});
        return "";
    }
    bench::RunResult run(const Variant& variant, bench::BenchContext&) override {
        calls.push_back(variant);
        if (throwVersion && variant.version == *throwVersion) throw std::runtime_error("pool roto");
        bench::RunResult runResult;
        runResult.timing.ms = 1.0 + static_cast<double>(calls.size());
        runResult.correct = !(wrongVersion && variant.version == *wrongVersion);
        if (failVersion && variant.version == *failVersion) {
            runResult.ok = false;
            runResult.error = "fallo simulado";
        }
        return runResult;
    }
    std::vector<std::pair<std::size_t, std::uint32_t>> prepared;
    std::vector<Variant> calls;
    const Version* wrongVersion = nullptr;
    const Version* failVersion = nullptr;
    const Version* throwVersion = nullptr;
};

class MemorySink : public bench::RowSink {
public:
    void write(const bench::Row& row) override { rows.push_back(row); }
    std::vector<bench::Row> rows;
};

bench::BenchOptions makeOptions(unsigned reps, std::vector<unsigned> threads) {
    bench::BenchOptions options;
    options.reps = reps;
    options.threads = std::move(threads);
    options.seed = 777;
    return options;
}

}  // namespace

int main() {
    // variantsFor: cpu1, una entrada cpuN por cada valor del barrido y las GPU si hay GPU.
    {
        FakeApp app;
        const auto cpuOnly = bench::variantsFor(app, false, {1, 2, 4});
        CHECK(cpuOnly.size() == 4);
        CHECK(cpuOnly[0].version == Version::Cpu1 && cpuOnly[0].threads == 1);
        CHECK(cpuOnly[1].version == Version::CpuN && cpuOnly[1].threads == 1);
        CHECK(cpuOnly[3].version == Version::CpuN && cpuOnly[3].threads == 4);
        const auto withGpu = bench::variantsFor(app, true, {2});
        CHECK(withGpu.size() == 4);
        CHECK(withGpu[2].version == Version::GpuKernel && withGpu[3].version == Version::GpuE2e);
    }

    // rotatedOrder: rota a la izquierda `repetition` posiciones (modulo n) y conserva todos los elementos.
    {
        const std::vector<Variant> baseOrder{{Version::Cpu1, 1}, {Version::CpuN, 2}, {Version::CpuN, 4},
                                             {Version::GpuE2e, 1}};
        for (std::size_t repetition = 0; repetition < 9; ++repetition) {
            const auto rotated = bench::rotatedOrder(baseOrder, repetition);
            CHECK(rotated.size() == baseOrder.size());
            for (std::size_t position = 0; position < baseOrder.size(); ++position) {
                const Variant& expected = baseOrder[(position + repetition) % baseOrder.size()];
                CHECK(rotated[position].version == expected.version && rotated[position].threads == expected.threads);
            }
        }
        // Cada variante ocupa cada posicion exactamente una vez en n repeticiones consecutivas.
        for (std::size_t position = 0; position < baseOrder.size(); ++position) {
            std::vector<int> timesAtPosition(baseOrder.size(), 0);
            for (std::size_t repetition = 0; repetition < baseOrder.size(); ++repetition) {
                const auto rotated = bench::rotatedOrder(baseOrder, repetition);
                for (std::size_t variantIndex = 0; variantIndex < baseOrder.size(); ++variantIndex) {
                    if (rotated[position].version == baseOrder[variantIndex].version &&
                        rotated[position].threads == baseOrder[variantIndex].threads) {
                        ++timesAtPosition[variantIndex];
                    }
                }
            }
            for (int times : timesAtPosition) CHECK(times == 1);
        }
        CHECK(bench::rotatedOrder({}, 3).empty());
    }

    // Protocolo completo: calentamiento sin registrar, reps filas por variante, orden rotado.
    {
        FakeApp app;
        MemorySink sink;
        bench::BenchContext context;
        std::ostringstream log;
        const auto outcome = bench::runBench(makeOptions(3, {1, 2}), {&app}, context, sink, log);
        const std::size_t variantCount = 3;  // cpu1, cpuN@1, cpuN@2
        const std::size_t sizeCount = 2;
        CHECK(outcome.incorrect == 0 && outcome.errors == 0);
        CHECK(sink.rows.size() == sizeCount * 3 * variantCount);
        CHECK(outcome.rows == sink.rows.size());
        // Misma semilla en todas las preparaciones.
        CHECK(app.prepared.size() == sizeCount);
        for (auto& preparation : app.prepared) CHECK(preparation.second == 777);
        // Llamadas totales = calentamiento (variantCount) + reps * variantCount, por tamano.
        CHECK(app.calls.size() == sizeCount * (variantCount + 3 * variantCount));
        // El primer bloque de un tamano es el calentamiento en orden base.
        CHECK(app.calls[0].version == Version::Cpu1);
        // En la primera repeticion registrada el orden es el base; en la 2.a empieza por la siguiente.
        std::vector<bench::Row> rowsOfSizeTen;
        for (auto& row : sink.rows) {
            if (row.size == 10) rowsOfSizeTen.push_back(row);
        }
        CHECK(rowsOfSizeTen.size() == 9);
        CHECK(rowsOfSizeTen[0].rep == 0 && rowsOfSizeTen[0].version == "cpu1");
        CHECK(rowsOfSizeTen[3].rep == 1 && rowsOfSizeTen[3].version == "cpuN" && rowsOfSizeTen[3].threads == 1);
        CHECK(rowsOfSizeTen[6].rep == 2 && rowsOfSizeTen[6].version == "cpuN" && rowsOfSizeTen[6].threads == 2);
        for (auto& row : sink.rows) {
            CHECK(row.seed == 777 && row.correct && row.application == "fake");
        }
    }

    // Una fila incorrecta se escribe igual (correct=no) y se cuenta.
    {
        FakeApp app;
        const Version wrongVersion = Version::CpuN;
        app.wrongVersion = &wrongVersion;
        MemorySink sink;
        bench::BenchContext context;
        std::ostringstream log;
        const auto outcome = bench::runBench(makeOptions(2, {1}), {&app}, context, sink, log);
        CHECK(outcome.incorrect == 2 * 2);  // 2 tamanos x 2 reps, una variante cpuN
        std::size_t incorrectRows = 0;
        for (auto& row : sink.rows) {
            if (!row.correct) ++incorrectRows;
        }
        CHECK(incorrectRows == outcome.incorrect);
        CHECK(outcome.exitCode() != 0);
    }

    // Un error de ejecucion no escribe fila, se informa y el exit code es != 0.
    {
        FakeApp app;
        const Version failingVersion = Version::Cpu1;
        app.failVersion = &failingVersion;
        MemorySink sink;
        bench::BenchContext context;
        std::ostringstream log;
        const auto outcome = bench::runBench(makeOptions(2, {1}), {&app}, context, sink, log);
        CHECK(outcome.errors > 0);
        CHECK(outcome.exitCode() != 0);
        CHECK(log.str().find("fallo simulado") != std::string::npos);
        for (auto& row : sink.rows) CHECK(row.version != "cpu1");
    }

    // Una excepcion de la aplicacion (p. ej. el pool no pudo crear hilos) no sale del runner:
    // se cuenta como corrida con error y el programa puede terminar con un mensaje, no con abort.
    {
        FakeApp app;
        const Version throwingVersion = Version::CpuN;
        app.throwVersion = &throwingVersion;
        MemorySink sink;
        bench::BenchContext context;
        std::ostringstream log;
        const auto outcome = bench::runBench(makeOptions(2, {1}), {&app}, context, sink, log);
        CHECK(outcome.errors == 2 * 2);
        CHECK(outcome.exitCode() != 0);
        CHECK(log.str().find("pool roto") != std::string::npos);
        for (auto& row : sink.rows) CHECK(row.version != "cpuN");
    }

    // Todo bien -> exit code 0.
    {
        bench::BenchOutcome cleanOutcome;
        CHECK(cleanOutcome.exitCode() == 0);
    }

    // Se niega a medir un build que no es Release o que tiene sanitizers.
    {
        bench::BuildInfo buildInfo;
        buildInfo.buildType = "Release";
        buildInfo.sanitizer = "none";
        CHECK(bench::benchRefusalReason(buildInfo).empty());
        for (const char* buildType : {"Debug", "RelWithDebInfo", "MinSizeRel", "unknown", ""}) {
            buildInfo.buildType = buildType;
            const std::string reason = bench::benchRefusalReason(buildInfo);
            CHECK(!reason.empty());
            CHECK(reason.find("Never time a Debug or sanitizer build") != std::string::npos);
        }
        buildInfo.buildType = "Release";
        for (const char* sanitizer : {"thread", "address,undefined"}) {
            buildInfo.sanitizer = sanitizer;
            CHECK(!bench::benchRefusalReason(buildInfo).empty());
        }
        // Un Release compilado sin optimizar (p. ej. -O0) tampoco se mide.
        buildInfo.sanitizer = "none";
        buildInfo.optimized = false;
        CHECK(!bench::benchRefusalReason(buildInfo).empty());
        buildInfo.optimized = true;
        // Ni uno con simbolos de depuracion de dispositivo (-G / --device-debug).
        for (const char* cudaFlags : {"-G", "-O3 -G -lineinfo", "--device-debug"}) {
            buildInfo.cudaFlags = cudaFlags;
            CHECK(!bench::benchRefusalReason(buildInfo).empty());
        }
        for (const char* cudaFlags : {"", "-O3 -lineinfo archs=75;89", "-Gx"}) {
            buildInfo.cudaFlags = cudaFlags;
            CHECK(bench::benchRefusalReason(buildInfo).empty());
        }
    }

    std::cout << "test_bench_runner OK\n";
    return 0;
}
