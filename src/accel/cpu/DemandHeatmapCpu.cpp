#include "accel/DemandHeatmap.h"

namespace accel {

namespace {
const char* const kNoCuda = "compilado sin CUDA";
}

DemandHeatGpu::~DemandHeatGpu() = default;
std::string DemandHeatGpu::allocate(int, int) { return kNoCuda; }
void DemandHeatGpu::release() {}
std::string DemandHeatGpu::setFilter(const std::vector<float>&, int) { return kNoCuda; }
std::string DemandHeatGpu::upload(const std::vector<float>&, double*) { return kNoCuda; }
std::string DemandHeatGpu::launch(DemandKernel, double*) { return kNoCuda; }
std::string DemandHeatGpu::download(std::vector<float>&, double*) { return kNoCuda; }

}  // namespace accel
