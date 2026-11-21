#include "accel/BestCourier.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

BestCourierDevice::~BestCourierDevice() = default;
std::string BestCourierDevice::uploadEta(const std::int32_t*, std::size_t, std::size_t, double*) { return kNoCuda; }
std::string BestCourierDevice::uploadCourierIds(const std::int32_t*, std::size_t, double*) { return kNoCuda; }
std::string BestCourierDevice::run(const void*, std::size_t, std::size_t, bool, double*) { return kNoCuda; }
std::string BestCourierDevice::download(std::vector<unsigned long long>&, double*) { return kNoCuda; }
void BestCourierDevice::release() {}

}  // namespace accel
