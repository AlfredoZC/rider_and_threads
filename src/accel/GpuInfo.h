#pragma once

#include <cstddef>
#include <string>

namespace accel {

// Resultado de consultar el backend GPU. Nunca lanza ni aborta: si no hay GPU,
// driver o soporte CUDA, `available` queda en false y `message` explica el motivo.
struct GpuInfo {
    bool available = false;
    std::string name;                 // vacio si no hay dispositivo
    int computeMajor = 0;             // compute capability, p. ej. 8.9
    int computeMinor = 0;
    std::size_t totalMemoryBytes = 0;
    std::string message;              // motivo cuando no esta disponible
};

// Detecta el backend. Hay dos implementaciones, elegidas por CMake:
//   accel/cuda/GpuInfo.cu      con -DUSE_CUDA=ON
//   accel/cpu/GpuInfoCpu.cpp   por defecto ("compilado sin CUDA")
GpuInfo gpuInfo();

// Linea de una sola fila para el arranque (va a stderr, nunca a stdout).
std::string describe(const GpuInfo& info);

}  // namespace accel
