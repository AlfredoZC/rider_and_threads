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
    int multiProcessorCount = 0;      // SMs
    int clockRateKHz = 0;             // reloj maximo de los SMs
    int memoryClockRateKHz = 0;
    int memoryBusWidthBits = 0;
    int runtimeVersion = 0;           // CUDA runtime, p. ej. 12090 = 12.9
    int driverVersion = 0;            // maximo CUDA que soporta el driver
    std::string message;              // motivo cuando no esta disponible
};

// Ancho de banda pico teorico de la memoria en GB/s (1e9 B/s): 2 (DDR) x reloj x bus / 8.
// 0 si no hay datos.
inline double peakBandwidthGBs(const GpuInfo& info) {
    if (info.memoryClockRateKHz <= 0 || info.memoryBusWidthBits <= 0) {
        return 0.0;
    }
    return 2.0 * info.memoryClockRateKHz * 1e3 * (info.memoryBusWidthBits / 8.0) / 1e9;
}

// Detecta el backend. Hay dos implementaciones, elegidas por CMake:
//   accel/cuda/GpuInfo.cu      con -DUSE_CUDA=ON
//   accel/cpu/GpuInfoCpu.cpp   por defecto ("compilado sin CUDA")
GpuInfo gpuInfo();

// Linea de una sola fila para el arranque (va a stderr, nunca a stdout).
std::string describe(const GpuInfo& info);

}  // namespace accel
