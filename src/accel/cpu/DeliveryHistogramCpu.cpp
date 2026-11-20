#include "accel/DeliveryHistogram.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

DeviceHistogram::~DeviceHistogram() = default;
std::string DeviceHistogram::allocate(std::size_t) { return kNoCuda; }
void DeviceHistogram::release() {}
std::string DeviceHistogram::upload(const std::int32_t*, std::size_t, double*) { return kNoCuda; }
std::string DeviceHistogram::run(HistogramKernel, double*) { return kNoCuda; }
std::string DeviceHistogram::download(unsigned*, double*) { return kNoCuda; }

}  // namespace accel
