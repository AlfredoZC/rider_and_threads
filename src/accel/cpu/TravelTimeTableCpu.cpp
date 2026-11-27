// Respaldo sin CUDA de la tabla de tiempos: las versiones de GPU no existen en esta compilacion.
#include "accel/GpuError.h"
#include "accel/TravelTimeTable.h"

namespace accel {

void floydWarshallTiledGpu(const std::vector<std::int32_t>&, std::size_t, std::vector<std::int32_t>&,
                           GpuFloydTiming*) {
    throw GpuUnavailableError("compilado sin CUDA");
}

void floydWarshallNaiveGpu(const std::vector<std::int32_t>&, std::size_t, std::vector<std::int32_t>&,
                           GpuFloydTiming*) {
    throw GpuUnavailableError("compilado sin CUDA");
}

}  // namespace accel
