#include "bench/BenchMain.h"

#include <cerrno>
#include <exception>
#include <filesystem>
#include <memory>
#include <system_error>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <thread>

#include "Config.h"
#include "accel/GpuInfo.h"
#include "bench/BenchCli.h"
#include "bench/BenchRegistry.h"
#include "bench/BenchRunner.h"
#include "bench/BuildInfo.h"
#include "bench/EnvInfo.h"

namespace bench {

namespace {

std::string joinNames(const std::vector<std::string>& names) {
    std::string joined;
    for (const std::string& appName : names) {
        joined += (joined.empty() ? "" : ", ") + appName;
    }
    return joined;
}

// Escribe `text` en `path` con las mismas reglas que el CSV (exclusivo sin --force, sin seguir enlaces).
std::string writeFile(const std::string& path, const std::string& text, bool force) {
    std::string errorMessage;
    std::FILE* stream = openOutputFile(path, force, errorMessage);
    if (stream == nullptr) {
        return errorMessage;
    }
    const bool written = std::fputs(text.c_str(), stream) >= 0;
    const bool closed = std::fclose(stream) == 0;
    return (written && closed) ? "" : "no se pudo escribir '" + path + "'";
}

// argv de la corrida sin rutas absolutas personales (ver sanitizePath).
std::vector<std::string> sanitizedCommand(const std::vector<std::string>& args) {
    std::error_code cwdError;
    const std::string workingDirectory = std::filesystem::current_path(cwdError).string();
    std::vector<std::string> command{"delivery_sim"};
    for (const std::string& argument : args) {
        command.push_back(sanitizePath(argument, cwdError ? std::string() : workingDirectory));
    }
    return command;
}

int runBenchCsv(const BenchOptions& options, BenchContext& context, std::vector<std::unique_ptr<BenchApp>>& apps,
                const std::vector<std::string>& args) {
    EnvInputs env = collectEnv();
    env.reps = options.reps;
    env.seed = options.seed;
    env.threads = options.threads;
    for (const auto& app : apps) {
        env.apps.push_back(app->name());
    }
    env.command = sanitizedCommand(args);
    env.config = env.command.size() > 1 ? env.command[1] : "";

    // El .env.json primero: si ya existe y no hay --force se corta antes de crear el CSV.
    const std::string envPath = options.csvPath + ".env.json";
    std::string errorMessage = writeFile(envPath, envJson(env), options.force);
    if (!errorMessage.empty()) {
        std::cerr << "Error: " << errorMessage << '\n';
        return 2;
    }
    CsvFile csv;
    errorMessage = csv.open(options.csvPath, options.force);
    if (!errorMessage.empty()) {
        std::cerr << "Error: " << errorMessage << '\n';
        if (!options.force) {
            std::remove(envPath.c_str());  // lo acabamos de crear nosotros (apertura exclusiva)
        }
        return 2;
    }

    std::vector<BenchApp*> rawApps;
    for (auto& app : apps) {
        rawApps.push_back(app.get());
    }
    const BenchOutcome outcome = runBench(options, rawApps, context, csv, std::cerr);
    std::cout << formatSummary(outcome.summary);
    std::cerr << "Filas escritas: " << outcome.rows << ", con correct=no: " << outcome.incorrect
              << ", corridas con error: " << outcome.errors << '\n';
    if (!csv.good()) {
        std::cerr << "Error: fallo la escritura del CSV\n";
        return 1;
    }
    return outcome.exitCode();
}

}  // namespace

namespace {

int runBenchModeImpl(const std::vector<std::string>& args) {
    const CliResult cli = parseBenchArgs(args, std::thread::hardware_concurrency());
    if (!cli.error.empty()) {
        std::cerr << "Error: " << cli.error << '\n' << benchUsage();
        return 2;
    }

    const BuildInfo build = currentBuildInfo();
    const std::string refusal = benchRefusalReason(build);
    if (cli.mode == Mode::Bench && !refusal.empty()) {
        std::cerr << "Error: " << refusal << '\n';
        return 2;
    }
    if (cli.mode == Mode::Once && !refusal.empty()) {
        std::cerr << "Aviso (los tiempos de esta corrida no valen): " << refusal << '\n';
    }

    BenchContext context;
    try {
        context.config = loadConfig(args[0]);
    } catch (const ConfigError& caught) {
        std::cerr << "Error: " << caught.what() << '\n';
        return 1;
    }
    context.gpu = accel::gpuInfo();
    std::cerr << accel::describe(context.gpu) << '\n';
    if (!context.gpu.available) {
        std::cerr << "Aviso: sin GPU, se omiten las versiones gpu_kernel y gpu_e2e; se miden solo las de CPU.\n";
    }

    const std::vector<std::string> known = BenchRegistry::instance().names();
    if (cli.mode == Mode::Bench) {
        auto apps = makeApps(cli.bench.apps);
        for (const std::string& wanted : cli.bench.apps) {
            if (BenchRegistry::instance().create(wanted) == nullptr) {
                std::cerr << "Error: aplicacion desconocida '" << wanted << "'. Disponibles: " << joinNames(known)
                          << '\n';
                return 2;
            }
        }
        for (const std::size_t size : cli.bench.sizes) {
            for (const auto& app : apps) {
                if (size > app->maxSize()) {
                    std::cerr << "Error: --sizes " << size << " supera el maximo de " << app->name() << " ("
                              << app->maxSize() << ")\n";
                    return 2;
                }
            }
        }
        return runBenchCsv(cli.bench, context, apps, args);
    }

    // --bench-once
    std::unique_ptr<BenchApp> app = BenchRegistry::instance().create(cli.once.app);
    if (!app) {
        std::cerr << "Error: aplicacion desconocida '" << cli.once.app << "'. Disponibles: " << joinNames(known)
                  << '\n';
        return 2;
    }
    if (cli.once.size > app->maxSize()) {
        std::cerr << "Error: --size supera el maximo de " << app->name() << " (" << app->maxSize() << ")\n";
        return 2;
    }
    std::vector<Variant> variants;
    if (cli.once.version) {
        const Version requested = *cli.once.version;
        if (isGpu(requested) && !context.gpu.available) {
            std::cerr << "Error: la version " << versionName(requested) << " necesita GPU y no hay\n";
            return 2;
        }
        bool supported = false;
        for (Version available : app->versions(context.gpu.available)) {
            supported = supported || available == requested;
        }
        if (!supported) {
            std::cerr << "Error: " << app->name() << " no tiene la version " << versionName(requested) << '\n';
            return 2;
        }
        variants.push_back(Variant{requested, requested == Version::CpuN ? cli.once.threads : 1});
    } else {
        for (Version version : app->versions(context.gpu.available)) {
            variants.push_back(Variant{version, version == Version::CpuN ? cli.once.threads : 1});
        }
    }
    std::cout << csvHeader() << '\n';
    return runOnce(*app, cli.once.size, cli.once.seed, variants, context, std::cout, std::cerr).exitCode();
}

}  // namespace

int runBenchMode(const std::vector<std::string>& args) {
    // Cualquier excepcion (hilos que no se pueden crear, sin memoria...) es un error claro, no un abort.
    try {
        return runBenchModeImpl(args);
    } catch (const std::exception& caught) {
        std::cerr << "Error: " << caught.what() << '\n';
        return 1;
    }
}

}  // namespace bench
