#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>
#include <vector>

#include "accel/GpuInfo.h"
#include "bench/BuildInfo.h"

namespace bench {

// Todo lo que describe la maquina y el build de una medicion (seccion 2.2: "Hardware, stated exactly").
struct EnvInputs {
    std::string cpuModel = "unknown";
    unsigned hardwareConcurrency = 0;  // hilos logicos
    unsigned physicalCores = 0;        // nucleos fisicos (0 = no se pudo determinar)
    std::string os = "unknown";
    std::size_t memTotalKb = 0;
    BuildInfo build;
    accel::GpuInfo gpu;
    std::string nvidiaDriver = "unknown";  // version real del driver NVIDIA (p. ej. 576.52)
    std::vector<std::string> command;      // argv de esta corrida, sin rutas absolutas
    std::string config;                    // ruta del config, sin ruta absoluta
    unsigned reps = 0;
    std::uint32_t seed = 0;
    std::vector<unsigned> threads;
    std::vector<std::string> apps;
};

// Una ruta absoluta se acorta para no dejar rutas personales en el JSON: relativa a `workingDirectory` si esta
// dentro; si no, solo su ultimo componente. Lo que no empieza con "/" se devuelve igual.
std::string sanitizePath(const std::string& argument, const std::string& workingDirectory);

// Nucleos fisicos segun el formato de /proc/cpuinfo: pares (physical id, core id) distintos. 0 si no hay.
unsigned countPhysicalCores(std::istream& cpuinfo);

// Lee el modelo de CPU (/proc/cpuinfo), la memoria (/proc/meminfo), el SO (uname) y los hilos
// logicos de esta maquina; el build sale de BuildInfo y la GPU de accel::gpuInfo().
EnvInputs collectEnv();

// El JSON que acompana al CSV (`<csv>.env.json`), con sangria de 2 espacios y salto de linea final.
// Solo contiene datos de hardware y build: ninguna ruta ni usuario.
std::string envJson(const EnvInputs& in);

}  // namespace bench
