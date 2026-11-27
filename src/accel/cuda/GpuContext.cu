#include <cuda_runtime.h>

#include <string>

#include "accel/GpuContext.h"
#include "accel/GpuError.h"

namespace accel {

namespace {

GpuInfo failed(const char* step, cudaError_t err) {
    GpuInfo info;
    info.message = std::string("CUDA: ") + step + ": " + cudaGetErrorString(err);
    cudaGetLastError();  // limpia el error pegajoso para no contaminar llamadas posteriores
    return info;
}

}  // namespace

GpuInfo initGpuContext() {
    // BlockingSync ANTES de crear el contexto: cudaStreamSynchronize/cudaEventSynchronize
    // duermen el hilo (sin girar) hasta que el kernel termina.
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (err != cudaSuccess) {
        return failed("cudaSetDeviceFlags", err);
    }
    int count = 0;
    err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return failed("cudaGetDeviceCount", err);
    }
    if (count == 0) {
        GpuInfo info;
        info.message = "CUDA: ningun dispositivo";
        return info;
    }
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return failed("cudaSetDevice", err);
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return failed("cudaGetDeviceProperties", err);
    }
    if (prop.computeMode == cudaComputeModeProhibited) {
        GpuInfo info;
        info.message = std::string("CUDA: dispositivo 0 en modo prohibido (") + prop.name + ")";
        return info;
    }
    if (prop.major * 10 + prop.minor < 75) {
        GpuInfo info;
        info.message = std::string("CUDA: compute capability ") + std::to_string(prop.major) +
                       "." + std::to_string(prop.minor) + " < 7.5 (" + prop.name + ")";
        return info;
    }
    err = cudaFree(nullptr);  // fuerza la creacion del contexto: el primer kernel no la paga
    if (err != cudaSuccess) {
        return failed("cudaFree(0) (crear contexto)", err);
    }

    GpuInfo info;
    info.available = true;
    info.name = prop.name;
    info.computeMajor = prop.major;
    info.computeMinor = prop.minor;
    info.totalMemoryBytes = prop.totalGlobalMem;
    return info;
}

void throwIfGpuError(const char* what) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw GpuFatalError(std::string("CUDA: ") + what + ": " + cudaGetErrorString(err));
    }
}

}  // namespace accel
