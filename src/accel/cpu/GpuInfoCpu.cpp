#include "accel/GpuInfo.h"

namespace accel {

GpuInfo gpuInfo() {
    GpuInfo info;
    info.message = "compilado sin CUDA";
    return info;
}

}  // namespace accel
