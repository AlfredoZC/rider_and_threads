#include "bench/BenchCli.h"

#include <algorithm>
#include <charconv>
#include <limits>

namespace bench {

namespace {

// Entero sin signo, decimal, sin espacios ni basura; en [minValue, maxValue].
bool parseUnsigned(const std::string& text, unsigned long long minValue, unsigned long long maxValue,
                   unsigned long long& output) {
    if (text.empty() || text.front() < '0' || text.front() > '9') {
        return false;
    }
    unsigned long long parsedValue = 0;
    const char* textEnd = text.data() + text.size();
    const auto conversion = std::from_chars(text.data(), textEnd, parsedValue);
    if (conversion.ec != std::errc() || conversion.ptr != textEnd) {
        return false;
    }
    if (parsedValue < minValue || parsedValue > maxValue) {
        return false;
    }
    output = parsedValue;
    return true;
}

std::vector<std::string> splitList(const std::string& text) {
    std::vector<std::string> items;
    std::size_t itemStart = 0;
    for (;;) {
        const std::size_t commaPosition = text.find(',', itemStart);
        items.push_back(
            text.substr(itemStart, commaPosition == std::string::npos ? std::string::npos : commaPosition - itemStart));
        if (commaPosition == std::string::npos) {
            break;
        }
        itemStart = commaPosition + 1;
    }
    return items;
}

template <typename T>
bool hasDuplicates(std::vector<T> values) {
    std::sort(values.begin(), values.end());
    return std::adjacent_find(values.begin(), values.end()) != values.end();
}

bool validAppName(const std::string& name) {
    if (name.empty() || name.size() > 64) {
        return false;
    }
    return std::all_of(name.begin(), name.end(), [](char character) {
        return (character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') || character == '_';
    });
}

template <typename T>
bool parseNumberList(const std::string& text, unsigned long long minValue, unsigned long long maxValue,
                     std::vector<T>& values, std::string& error, const char* flag) {
    const std::vector<std::string> items = splitList(text);
    if (items.size() > kMaxListItems) {
        error = std::string(flag) + ": demasiados valores (maximo " + std::to_string(kMaxListItems) + ")";
        return false;
    }
    values.clear();
    for (const std::string& item : items) {
        unsigned long long parsedValue = 0;
        if (!parseUnsigned(item, minValue, maxValue, parsedValue)) {
            error = std::string(flag) + ": valor invalido '" + item + "' (entero entre " + std::to_string(minValue) +
                    " y " + std::to_string(maxValue) + ")";
            return false;
        }
        values.push_back(static_cast<T>(parsedValue));
    }
    if (hasDuplicates(values)) {
        error = std::string(flag) + ": valores repetidos (duplicarian filas y sesgarian las medianas)";
        return false;
    }
    return true;
}

}  // namespace

bool isBenchInvocation(const std::vector<std::string>& args) {
    for (std::size_t argumentIndex = 1; argumentIndex < args.size(); ++argumentIndex) {
        if (args[argumentIndex] == "--bench" || args[argumentIndex] == "--bench-once") {
            return true;
        }
    }
    return false;
}

std::string benchUsage() {
    return "Uso:\n"
           "  delivery_sim <config> --headless --bench <results.csv> [--apps a,b] [--reps N] [--threads 1,2,4]\n"
           "                        [--sizes s1,s2] [--seed S] [--force]\n"
           "  delivery_sim <config> --headless --bench-once --app X --size N [--version V] [--threads N] [--seed S]\n"
           "  V: cpu1 | cpuN | gpu_kernel | gpu_e2e\n";
}

CliResult parseBenchArgs(const std::vector<std::string>& args, unsigned hardwareConcurrency) {
    CliResult result;
    bool headless = false;
    bool benchFlag = false;
    bool onceFlag = false;
    bool haveApp = false;
    bool haveSize = false;
    bool haveThreads = false;
    // Flags exclusivos de cada modo, para rechazar mezclas.
    std::vector<std::string> benchOnlyFlags;
    std::vector<std::string> onceOnlyFlags;
    std::vector<unsigned> threadList;

    auto fail = [&](const std::string& message) {
        result.error = message;
        return result;
    };

    for (std::size_t argumentIndex = 1; argumentIndex < args.size(); ++argumentIndex) {
        const std::string& flag = args[argumentIndex];
        auto readValue = [&](std::string& destination) {
            if (argumentIndex + 1 >= args.size()) {
                return false;
            }
            destination = args[++argumentIndex];
            return true;
        };
        std::string argumentText;
        unsigned long long numericValue = 0;
        if (flag == "--headless") {
            headless = true;
        } else if (flag == "--bench") {
            if (!readValue(result.bench.csvPath) || result.bench.csvPath.empty()) {
                return fail("--bench necesita la ruta del CSV");
            }
            benchFlag = true;
        } else if (flag == "--bench-once") {
            onceFlag = true;
        } else if (flag == "--force") {
            result.bench.force = true;
            benchOnlyFlags.push_back(flag);
        } else if (flag == "--apps") {
            if (!readValue(argumentText)) {
                return fail("--apps necesita una lista");
            }
            result.bench.apps = splitList(argumentText);
            if (result.bench.apps.size() > kMaxListItems) {
                return fail("--apps: demasiados valores");
            }
            if (hasDuplicates(result.bench.apps)) {
                return fail("--apps: nombres repetidos (duplicarian filas y sesgarian las medianas)");
            }
            for (const std::string& appName : result.bench.apps) {
                if (!validAppName(appName)) {
                    return fail("--apps: nombre invalido '" + appName + "'");
                }
            }
            benchOnlyFlags.push_back(flag);
        } else if (flag == "--reps") {
            if (!readValue(argumentText) || !parseUnsigned(argumentText, 1, kMaxReps, numericValue)) {
                return fail("--reps: entero entre 1 y " + std::to_string(kMaxReps));
            }
            result.bench.reps = static_cast<unsigned>(numericValue);
            benchOnlyFlags.push_back(flag);
        } else if (flag == "--sizes") {
            if (!readValue(argumentText)) {
                return fail("--sizes necesita una lista");
            }
            std::string listError;
            if (!parseNumberList(argumentText, 1, kMaxSizeArgument, result.bench.sizes, listError, "--sizes")) {
                return fail(listError);
            }
            benchOnlyFlags.push_back(flag);
        } else if (flag == "--threads") {
            if (!readValue(argumentText)) {
                return fail("--threads necesita una lista");
            }
            std::string listError;
            if (!parseNumberList(argumentText, 1, kMaxThreads, threadList, listError, "--threads")) {
                return fail(listError);
            }
            haveThreads = true;
        } else if (flag == "--seed") {
            if (!readValue(argumentText) ||
                !parseUnsigned(argumentText, 0, std::numeric_limits<std::uint32_t>::max(), numericValue)) {
                return fail("--seed: entero entre 0 y 4294967295");
            }
            result.bench.seed = static_cast<std::uint32_t>(numericValue);
            result.once.seed = static_cast<std::uint32_t>(numericValue);
        } else if (flag == "--app") {
            if (!readValue(result.once.app) || !validAppName(result.once.app)) {
                return fail("--app: nombre invalido");
            }
            haveApp = true;
            onceOnlyFlags.push_back(flag);
        } else if (flag == "--size") {
            if (!readValue(argumentText) || !parseUnsigned(argumentText, 1, kMaxSizeArgument, numericValue)) {
                return fail("--size: entero entre 1 y " + std::to_string(kMaxSizeArgument));
            }
            result.once.size = static_cast<std::size_t>(numericValue);
            haveSize = true;
            onceOnlyFlags.push_back(flag);
        } else if (flag == "--version") {
            Version parsedVersion = Version::Cpu1;
            if (!readValue(argumentText) || !parseVersion(argumentText, parsedVersion)) {
                return fail("--version: use cpu1, cpuN, gpu_kernel o gpu_e2e");
            }
            result.once.version = parsedVersion;
            onceOnlyFlags.push_back(flag);
        } else {
            return fail("argumento desconocido '" + flag + "'");
        }
    }

    if (!headless) {
        return fail("el modo bench requiere --headless");
    }
    if (benchFlag == onceFlag) {
        return fail(benchFlag ? "--bench y --bench-once no se combinan" : "falta --bench <csv> o --bench-once");
    }

    const unsigned logicalThreads = hardwareConcurrency == 0 ? 1 : hardwareConcurrency;
    if (benchFlag) {
        if (!onceOnlyFlags.empty()) {
            return fail(onceOnlyFlags.front() + " solo vale con --bench-once");
        }
        result.mode = Mode::Bench;
        if (haveThreads) {
            result.bench.threads = threadList;
        } else {
            for (unsigned candidateThreads : {1u, 2u, 4u, 6u, 8u, 12u, 16u}) {
                if (candidateThreads <= logicalThreads) {
                    result.bench.threads.push_back(candidateThreads);
                }
            }
        }
    } else {
        if (!benchOnlyFlags.empty()) {
            return fail(benchOnlyFlags.front() + " solo vale con --bench");
        }
        if (!haveApp || !haveSize) {
            return fail("--bench-once requiere --app y --size");
        }
        if (haveThreads && threadList.size() != 1) {
            return fail("--threads con --bench-once admite un solo valor");
        }
        result.mode = Mode::Once;
        result.once.threads = haveThreads ? threadList.front() : logicalThreads;
    }
    return result;
}

}  // namespace bench
