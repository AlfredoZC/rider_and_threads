#include "accel/Calib.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

DeviceBuffer::~DeviceBuffer() = default;
std::string DeviceBuffer::allocate(std::size_t) { return kNoCuda; }
void DeviceBuffer::release() {}
std::string DeviceBuffer::upload(const void*, std::size_t, double*) { return kNoCuda; }
std::string DeviceBuffer::download(void*, std::size_t, double*) { return kNoCuda; }
std::string DeviceBuffer::copyTo(DeviceBuffer&, double*) { return kNoCuda; }
std::string DeviceBuffer::clear() { return kNoCuda; }

std::string deviceSynchronize() { return kNoCuda; }
std::string gpuMemory(std::size_t*, std::size_t*) { return kNoCuda; }
std::string launchCounterKernel(DeviceBuffer&, std::size_t, bool, double*) { return kNoCuda; }

}  // namespace accel
