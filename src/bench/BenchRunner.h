#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "bench/BenchApp.h"
#include "bench/BenchCsv.h"

namespace bench {

struct BenchOptions {
    std::string csvPath;
    bool force = false;
    std::vector<std::string> apps;  // vacio = todas las registradas
    unsigned reps = 10;
    std::vector<unsigned> threads;  // barrido de cpuN
    std::uint32_t seed = 12345;
    std::vector<std::size_t> sizes;  // vacio = los de cada aplicacion
};

// Una linea del resumen: estadisticos de las repeticiones de una (app, tamano, version, hilos).
struct SummaryLine {
    std::string application;
    std::size_t size = 0;
    std::string version;
    unsigned threads = 1;
    std::size_t samples = 0;
    double medianMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    std::string note;  // metrica derivada de la aplicacion (GB/s, us por lanzamiento...)
};

struct BenchOutcome {
    std::size_t rows = 0;       // filas escritas
    std::size_t incorrect = 0;  // filas con correct=no
    std::size_t errors = 0;     // corridas que no se pudieron ejecutar
    std::vector<SummaryLine> summary;

    // 0 solo si todo se ejecuto y todo dio correct=yes.
    int exitCode() const { return (incorrect == 0 && errors == 0) ? 0 : 1; }
};

// Versiones a medir: cpu1, una cpuN por cada valor del barrido y las GPU si hay GPU y la app las tiene.
std::vector<Variant> variantsFor(const BenchApp& app, bool gpuAvailable, const std::vector<unsigned>& threadSweep);

// El orden de las versiones en la repeticion `repetition`: el base rotado a la izquierda `repetition` posiciones.
// Asi cada version pasa por todas las posiciones y el calor acumulado no favorece siempre a la misma.
std::vector<Variant> rotatedOrder(const std::vector<Variant>& baseOrder, std::size_t repetition);

// El protocolo del enunciado (2.2) por (app, tamano): prepara con la semilla, una corrida de
// calentamiento de cada version sin registrar, y `reps` repeticiones con el orden rotado. Cada
// corrida se compara con la referencia y deja una fila. Avisos y errores van a `log`.
BenchOutcome runBench(const BenchOptions& options, const std::vector<BenchApp*>& apps, BenchContext& context,
                      RowSink& sink, std::ostream& log);

// Tabla de texto con la mediana, el minimo y el maximo de cada linea.
std::string formatSummary(const std::vector<SummaryLine>& lines);

// Una sola corrida de cada version pedida, sin calentamiento ni CSV (para ncu / compute-sanitizer).
// Escribe una linea por corrida en `output` y errores en `log`.
BenchOutcome runOnce(BenchApp& app, std::size_t size, std::uint32_t seed, const std::vector<Variant>& variants,
                     BenchContext& context, std::ostream& output, std::ostream& log);

}  // namespace bench
