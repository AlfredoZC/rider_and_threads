#include "accel/GpuInfo.h"

namespace accel {

std::string describe(const GpuInfo& info) {
    if (!info.available) {
        return "Backend: CPU (GPU no disponible: " + info.message + ")";
    }
    const std::size_t mib = info.totalMemoryBytes / (1024 * 1024);
    return "Backend: GPU CUDA \"" + info.name + "\", cc " + std::to_string(info.computeMajor) +
           "." + std::to_string(info.computeMinor) + ", " + std::to_string(mib) + " MiB";
}

}  // namespace accel
