#pragma once

#include <string>

namespace bench {

// Datos del build, fijados en tiempo de configuracion por CMake (BuildInfoGenerated.h). Si el
// archivo no existe (p. ej. tests/run.sh compila a mano) los campos quedan en "unknown".
struct BuildInfo {
    std::string buildType = "unknown";  // CMAKE_BUILD_TYPE
    std::string sanitizer = "none";     // none | thread | address,undefined
    std::string compiler;               // __VERSION__
    std::string cxxFlags;               // CMAKE_CXX_FLAGS + los del tipo de build
    std::string cudaFlags;              // vacio sin CUDA
    std::string gitCommit = "unknown";
    bool cuda = false;
    bool optimized = true;              // el compilador definio __OPTIMIZE__ (-O1 o mas)
};

BuildInfo currentBuildInfo();

// "" si se puede medir; si no, el motivo (empieza por "Never time a Debug or sanitizer build").
// Solo se mide Release sin sanitizers: el resto da tiempos que no significan nada.
std::string benchRefusalReason(const BuildInfo& info);

}  // namespace bench
