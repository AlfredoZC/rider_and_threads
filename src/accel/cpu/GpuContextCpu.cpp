#include "accel/GpuContext.h"

namespace accel {

GpuInfo initGpuContext() {
    GpuInfo info;
    info.message = "compilado sin CUDA";
    return info;
}

void throwIfGpuError(const char*) {}

}  // namespace accel
