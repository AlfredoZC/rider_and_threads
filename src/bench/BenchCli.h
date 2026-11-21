#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "bench/BenchRunner.h"

namespace bench {

enum class Mode { None, Bench, Once };

// --bench-once: una sola corrida (para ncu / compute-sanitizer).
struct OnceOptions {
    std::string app;
    std::size_t size = 0;
    std::optional<Version> version;  // sin valor: todas las versiones de la aplicacion, una vez cada una
    unsigned threads = 1;            // hilos de cpuN
    std::uint32_t seed = 12345;
};

struct CliResult {
    Mode mode = Mode::None;
    std::string error;  // vacio si los argumentos son validos
    BenchOptions bench;
    OnceOptions once;
};

// Limites de los argumentos numericos (todo valor fuera de ellos se rechaza con un mensaje).
inline constexpr unsigned kMaxReps = 100000;
inline constexpr unsigned kMaxThreads = 1024;
inline constexpr std::size_t kMaxListItems = 32;
inline constexpr std::size_t kMaxSizeArgument = std::size_t{1} << 40;

// true si algun argumento (despues del config) es --bench o --bench-once: main enruta al modo bench.
bool isBenchInvocation(const std::vector<std::string>& args);

// args[0] es la ruta del config. Rechaza flags desconocidos, valores faltantes, numeros mal
// formados (std::from_chars, sin signos ni basura al final) o fuera de limite.
// hardwareConcurrency recorta el barrido de hilos por defecto (1,2,4,6,8,12,16).
CliResult parseBenchArgs(const std::vector<std::string>& args, unsigned hardwareConcurrency);

std::string benchUsage();

}  // namespace bench
