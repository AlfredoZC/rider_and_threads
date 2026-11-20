#pragma once

#include <string>
#include <vector>

namespace bench {

// Punto de entrada del modo bench (--bench / --bench-once). `args[0]` es el config. Devuelve el
// codigo de salida: 0 todo bien; 1 corridas con error o con correct=no; 2 argumentos invalidos o
// build que no se puede medir.
int runBenchMode(const std::vector<std::string>& args);

}  // namespace bench
