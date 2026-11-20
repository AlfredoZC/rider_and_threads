#include <cuda_runtime.h>

#include "accel/GpuInfo.h"

namespace accel {

GpuInfo gpuInfo() {
    GpuInfo info;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // Sin driver o sin GPU: se informa y se sigue en CPU, sin abortar.
        info.message = std::string("CUDA: ") + cudaGetErrorString(err);
        cudaGetLastError();  // limpia el error pegajoso
        return info;
    }
    if (count == 0) {
        info.message = "CUDA: ningun dispositivo";
        return info;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        info.message = std::string("CUDA: ") + cudaGetErrorString(err);
        cudaGetLastError();
        return info;
    }

    info.available = true;
    info.name = prop.name;
    info.computeMajor = prop.major;
    info.computeMinor = prop.minor;
    info.totalMemoryBytes = prop.totalGlobalMem;
    info.multiProcessorCount = prop.multiProcessorCount;
    // Los campos clockRate/memoryClockRate de cudaDeviceProp estan obsoletos: se piden como atributos.
    cudaDeviceGetAttribute(&info.clockRateKHz, cudaDevAttrClockRate, 0);
    cudaDeviceGetAttribute(&info.memoryClockRateKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&info.memoryBusWidthBits, cudaDevAttrGlobalMemoryBusWidth, 0);
    cudaRuntimeGetVersion(&info.runtimeVersion);
    cudaDriverGetVersion(&info.driverVersion);
    cudaGetLastError();  // un atributo no disponible no debe dejar un error pegajoso
    return info;
}

}  // namespace accel
