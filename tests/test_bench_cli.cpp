#include "bench/BenchCli.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

using Args = std::vector<std::string>;

static bench::CliResult parse(Args arguments, unsigned logicalThreads = 8) {
    arguments.insert(arguments.begin(), "config/x.json");
    return bench::parseBenchArgs(arguments, logicalThreads);
}

int main() {
    // Caso minimo valido.
    {
        auto parsed = parse({"--headless", "--bench", "out.csv"});
        CHECK(parsed.error.empty());
        CHECK(parsed.mode == bench::Mode::Bench);
        CHECK(parsed.bench.csvPath == "out.csv");
        CHECK(parsed.bench.reps == 10);
        CHECK(!parsed.bench.force);
        CHECK(parsed.bench.apps.empty());
        // barrido por defecto recortado a hardware_concurrency = 8
        CHECK((parsed.bench.threads == std::vector<unsigned>{1, 2, 4, 6, 8}));
    }
    {
        auto parsed = parse({"--headless", "--bench", "out.csv"}, 3);
        CHECK((parsed.bench.threads == std::vector<unsigned>{1, 2}));
        auto singleThread = parse({"--headless", "--bench", "out.csv"}, 1);
        CHECK((singleThread.bench.threads == std::vector<unsigned>{1}));
        auto manyThreads = parse({"--headless", "--bench", "out.csv"}, 32);
        CHECK((manyThreads.bench.threads == std::vector<unsigned>{1, 2, 4, 6, 8, 12, 16}));
    }
    // Todo especificado.
    {
        auto parsed = parse({"--headless", "--bench", "o.csv", "--apps", "calib_copy,calib_launch", "--reps", "2",
                        "--threads", "1,3", "--seed", "99", "--sizes", "4096,8192", "--force"});
        CHECK(parsed.error.empty());
        CHECK((parsed.bench.apps == std::vector<std::string>{"calib_copy", "calib_launch"}));
        CHECK(parsed.bench.reps == 2);
        CHECK((parsed.bench.threads == std::vector<unsigned>{1, 3}));
        CHECK(parsed.bench.seed == 99);
        CHECK((parsed.bench.sizes == std::vector<std::size_t>{4096, 8192}));
        CHECK(parsed.bench.force);
    }
    // bench-once.
    {
        auto parsed = parse({"--headless", "--bench-once", "--app", "calib_copy", "--size", "4096", "--version",
                        "gpu_kernel"});
        CHECK(parsed.error.empty());
        CHECK(parsed.mode == bench::Mode::Once);
        CHECK(parsed.once.app == "calib_copy" && parsed.once.size == 4096);
        CHECK(parsed.once.version.has_value());
        auto withoutVersion = parse({"--headless", "--bench-once", "--app", "calib_copy", "--size", "4096"});
        CHECK(withoutVersion.error.empty() && !withoutVersion.once.version.has_value());
    }
    // Rechazos.
    const std::vector<Args> rejectedCommandLines = {
        {"--bench", "o.csv"},                       // falta --headless
        {"--headless", "--bench"},                  // falta la ruta
        {"--headless"},                             // ni --bench ni --bench-once
        {"--headless", "--bench", "o.csv", "--bench-once", "--app", "a", "--size", "1"},
        {"--headless", "--bench", "o.csv", "--frobnicate"},  // flag desconocido
        {"--headless", "--bench", "o.csv", "--reps", "0"},
        {"--headless", "--bench", "o.csv", "--reps", "-3"},
        {"--headless", "--bench", "o.csv", "--reps", "abc"},
        {"--headless", "--bench", "o.csv", "--reps", "12x"},
        {"--headless", "--bench", "o.csv", "--reps", "999999999999999999999"},
        {"--headless", "--bench", "o.csv", "--reps", "100001"},
        {"--headless", "--bench", "o.csv", "--reps"},
        {"--headless", "--bench", "o.csv", "--threads", "0"},
        {"--headless", "--bench", "o.csv", "--threads", "1,,2"},
        {"--headless", "--bench", "o.csv", "--threads", "1,2,"},
        {"--headless", "--bench", "o.csv", "--threads", "5000"},
        {"--headless", "--bench", "o.csv", "--seed", "4294967296"},
        {"--headless", "--bench", "o.csv", "--seed", "-1"},
        {"--headless", "--bench", "o.csv", "--apps", ""},
        {"--headless", "--bench", "o.csv", "--apps", "a,,b"},
        {"--headless", "--bench", "o.csv", "--sizes", "0"},
        {"--headless", "--bench", "o.csv", "--threads", "1,1"},      // duplicados sesgan las medianas
        {"--headless", "--bench", "o.csv", "--threads", "1,2,1"},
        {"--headless", "--bench", "o.csv", "--apps", "calib_copy,calib_copy"},
        {"--headless", "--bench", "o.csv", "--sizes", "100,100"},
        {"--headless", "--bench", "o.csv", "--app", "calib_copy"},  // flag de bench-once
        {"--headless", "--bench-once", "--size", "4096"},           // falta --app
        {"--headless", "--bench-once", "--app", "x"},               // falta --size
        {"--headless", "--bench-once", "--app", "x", "--size", "4096", "--version", "turbo"},
        {"--headless", "--bench-once", "--app", "x", "--size", "4096", "--reps", "2"},
        {"--headless", "--bench-once", "--app", "x", "--size", "99999999999999999999999"},
        {"--headless", "--bench", "o.csv", "--log", "x.log"},  // flag de la simulacion
    };
    for (const Args& rejectedArguments : rejectedCommandLines) {
        const auto parsed = parse(rejectedArguments);
        if (parsed.error.empty()) {
            std::cerr << "se esperaba rechazo de:";
            for (const auto& argument : rejectedArguments) std::cerr << ' ' << argument;
            std::cerr << '\n';
            CHECK(!parsed.error.empty());
        }
    }

    // isBenchInvocation distingue la CLI de bench de la de la simulacion.
    CHECK(bench::isBenchInvocation({"c.json", "--headless", "--bench", "o.csv"}));
    CHECK(bench::isBenchInvocation({"c.json", "--bench-once"}));
    CHECK(!bench::isBenchInvocation({"c.json", "--headless"}));

    std::cout << "test_bench_cli OK\n";
    return 0;
}
