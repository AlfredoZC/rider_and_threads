#include "accel/CoverageField.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

CoverageGpu::~CoverageGpu() = default;
std::string CoverageGpu::allocate(int, int) { return kNoCuda; }
void CoverageGpu::release() {}
std::string CoverageGpu::upload(const std::vector<float>&, double*) { return kNoCuda; }
std::string CoverageGpu::run(int, double*) { return kNoCuda; }
std::string CoverageGpu::download(std::vector<float>&, double*) { return kNoCuda; }

}  // namespace accel
