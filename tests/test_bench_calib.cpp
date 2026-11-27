#include "bench/BenchApp.h"
#include "bench/BenchRegistry.h"

#include <cstdlib>
#include <iostream>
#include <string>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

// Las aplicaciones de calibracion, en el build sin CUDA: solo versiones de CPU, todas correctas.
int main() {
    auto apps = bench::makeApps({});
    CHECK(apps.size() >= 2);
    bool hasCopyApp = false;
    bool hasLaunchApp = false;
    for (auto& registeredApp : apps) {
        if (registeredApp->name() == "calib_copy") hasCopyApp = true;
        if (registeredApp->name() == "calib_launch") hasLaunchApp = true;
    }
    CHECK(hasCopyApp && hasLaunchApp);
    CHECK(bench::makeApps({"calib_copy"}).size() == 1);
    CHECK(bench::makeApps({"no_existe"}).empty());

    bench::BenchContext context;
    for (auto& app : apps) {
        const auto cpuVersions = app->versions(false);
        CHECK(!cpuVersions.empty());
        for (bench::Version version : cpuVersions) {
            CHECK(version == bench::Version::Cpu1 || version == bench::Version::CpuN);
        }
        // Al menos 5 tamanos en un rango de dos ordenes de magnitud.
        const auto sizes = app->defaultSizes(context.gpu);
        CHECK(sizes.size() >= 5);
        CHECK(sizes.back() >= 100 * sizes.front());

        const std::size_t size = app->name() == "calib_copy" ? 4096 : 10;
        CHECK(app->prepare(size, 5, context).empty());
        for (bench::Variant variant : {bench::Variant{bench::Version::Cpu1, 1}, bench::Variant{bench::Version::CpuN, 3}}) {
            const auto runResult = app->run(variant, context);
            CHECK(runResult.ok);
            CHECK(runResult.correct);
            CHECK(runResult.timing.ms >= 0.0);
        }
        // Tamano fuera de rango -> error claro, sin abortar.
        CHECK(!app->prepare(0, 5, context).empty());
        CHECK(!app->prepare(app->maxSize() + 1, 5, context).empty());
    }
    std::cout << "test_bench_calib OK\n";
    return 0;
}
