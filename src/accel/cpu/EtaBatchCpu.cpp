#include "accel/EtaBatch.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

EtaDevice::~EtaDevice() = default;
std::string EtaDevice::uploadTable(const std::int32_t*, std::size_t, double*) { return kNoCuda; }
std::string EtaDevice::uploadCouriersSoA(const CourierColumns&, double*) { return kNoCuda; }
std::string EtaDevice::uploadCouriersAoS(const std::vector<CourierRecord>&, double*) { return kNoCuda; }
std::string EtaDevice::uploadOrders(const OrderColumns&, double*) { return kNoCuda; }
std::string EtaDevice::runEta(bool, double*) { return kNoCuda; }
std::string EtaDevice::download(std::vector<std::int32_t>&, double*) { return kNoCuda; }
void EtaDevice::release() {}
std::string EtaDevice::ensure(void**, std::size_t&, std::size_t, std::size_t, const char*) { return kNoCuda; }

}  // namespace accel
