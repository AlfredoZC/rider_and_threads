#include "accel/DeliveryTimeline.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

DeviceTimeline::~DeviceTimeline() = default;
std::string DeviceTimeline::allocate(std::size_t, const TimelinePlan&) { return kNoCuda; }
void DeviceTimeline::release() {}
std::string DeviceTimeline::upload(const DeliveryRecord*, std::size_t, const TimelinePlan&, double*) { return kNoCuda; }
std::string DeviceTimeline::run(double*) { return kNoCuda; }
std::string DeviceTimeline::download(DeliveryRecord*, int*, double*) { return kNoCuda; }

}  // namespace accel
