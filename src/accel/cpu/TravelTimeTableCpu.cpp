#include "accel/TravelTimeTable.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

ApspDevice::~ApspDevice() = default;
std::string ApspDevice::allocate(std::size_t) { return kNoCuda; }
void ApspDevice::release() {}
std::string ApspDevice::upload(const std::int32_t*, double*) { return kNoCuda; }
std::string ApspDevice::runTiled(int, double*) { return kNoCuda; }
std::string ApspDevice::download(std::int32_t*, double*) { return kNoCuda; }

}  // namespace accel
