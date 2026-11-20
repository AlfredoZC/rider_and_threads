#include "accel/OpenOrders.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

DeviceOpenOrders::~DeviceOpenOrders() = default;
std::string DeviceOpenOrders::allocate(std::size_t) { return kNoCuda; }
void DeviceOpenOrders::release() {}
std::string DeviceOpenOrders::upload(const std::int32_t*, std::size_t, double*) { return kNoCuda; }
std::string DeviceOpenOrders::run(double*) { return kNoCuda; }
std::string DeviceOpenOrders::download(std::int32_t*, unsigned long long*, double*) { return kNoCuda; }

}  // namespace accel
