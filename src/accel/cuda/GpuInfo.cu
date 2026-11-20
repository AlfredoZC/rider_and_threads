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
    return info;
}

}  // namespace accel
