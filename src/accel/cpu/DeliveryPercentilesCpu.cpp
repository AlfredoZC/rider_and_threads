#include "accel/DeliveryPercentiles.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

DeviceRadixSort::~DeviceRadixSort() = default;
std::string DeviceRadixSort::allocate(std::size_t) { return kNoCuda; }
void DeviceRadixSort::release() {}
std::string DeviceRadixSort::upload(const std::uint32_t*, std::size_t, double*) { return kNoCuda; }
std::string DeviceRadixSort::run(double*) { return kNoCuda; }
std::string DeviceRadixSort::download(std::uint32_t*, double*) { return kNoCuda; }

}  // namespace accel
