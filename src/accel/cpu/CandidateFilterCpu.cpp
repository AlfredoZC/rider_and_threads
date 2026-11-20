#include "accel/CandidateFilter.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

CandidateFilterDevice::~CandidateFilterDevice() = default;
std::string CandidateFilterDevice::uploadTable(const std::int32_t*, std::size_t, double*) { return kNoCuda; }
std::string CandidateFilterDevice::reserveFleet(std::size_t) { return kNoCuda; }
std::string CandidateFilterDevice::uploadFleet(const FleetColumns&, double*) { return kNoCuda; }
std::string CandidateFilterDevice::runFilter(bool, std::int32_t, std::int32_t, double*) { return kNoCuda; }
std::string CandidateFilterDevice::download(std::vector<std::int32_t>&, double*) { return kNoCuda; }
void CandidateFilterDevice::release() {}
void CandidateFilterDevice::releaseFleet() {}

}  // namespace accel
