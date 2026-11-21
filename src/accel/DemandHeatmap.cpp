#include "accel/DemandHeatmap.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <stdexcept>
#include <utility>

#include "accel/GpuContext.h"
#include "accel/GpuError.h"
#include "accel/GpuWorker.h"

namespace accel {

namespace {

// Calcula las filas [firstRow, endRow) de la salida. Es la unica funcion de convolucion de la CPU: cpu1 la llama
// con todas las filas y cpuN con una franja por trozo. Recorre el filtro por filas y salta lo que cae fuera
// de la grilla (aporta 0), el mismo orden de sumas que el kernel de la GPU.
void smoothDemandRows(const std::vector<float>& ordersPerCell, std::vector<float>& demandHeat, int gridWidth,
                      int gridHeight, const std::vector<float>& filterWeights, int filterRadius,
                      std::size_t firstRow, std::size_t endRow) {
    const int filterSize = 2 * filterRadius + 1;
    for (std::size_t cellRow = firstRow; cellRow < endRow; ++cellRow) {
        const int cellY = static_cast<int>(cellRow);
        const int firstOffsetY = std::max(-filterRadius, -cellY);
        const int lastOffsetY = std::min(filterRadius, gridHeight - 1 - cellY);
        for (int cellX = 0; cellX < gridWidth; ++cellX) {
            const int firstOffsetX = std::max(-filterRadius, -cellX);
            const int lastOffsetX = std::min(filterRadius, gridWidth - 1 - cellX);
            float weightedSum = 0.0f;
            for (int offsetY = firstOffsetY; offsetY <= lastOffsetY; ++offsetY) {
                for (int offsetX = firstOffsetX; offsetX <= lastOffsetX; ++offsetX) {
                    const std::size_t weightIndex = static_cast<std::size_t>(offsetY + filterRadius) * filterSize +
                                                    static_cast<std::size_t>(offsetX + filterRadius);
                    const std::size_t neighborIndex = static_cast<std::size_t>(cellY + offsetY) * gridWidth +
                                                      static_cast<std::size_t>(cellX + offsetX);
                    weightedSum += filterWeights[weightIndex] * ordersPerCell[neighborIndex];
                }
            }
            demandHeat[cellRow * gridWidth + static_cast<std::size_t>(cellX)] = weightedSum;
        }
    }
}

}  // namespace

void checkDemandArguments(std::size_t ordersCellCount, int gridWidth, int gridHeight, std::size_t filterWeightCount,
                          int filterRadius) {
    if (gridWidth < 1 || gridHeight < 1 || gridWidth > kMaxGridSide || gridHeight > kMaxGridSide) {
        throw std::invalid_argument("smoothDemand: el lado de la grilla debe estar en 1.." + std::to_string(kMaxGridSide));
    }
    if (filterRadius < 0 || filterRadius > kMaxFilterRadius) {
        throw std::invalid_argument("smoothDemand: el radio del filtro debe estar en 0.." +
                                    std::to_string(kMaxFilterRadius));
    }
    if (ordersCellCount != static_cast<std::size_t>(gridWidth) * static_cast<std::size_t>(gridHeight)) {
        throw std::invalid_argument("smoothDemand: los conteos no coinciden con ancho * alto");
    }
    const std::size_t filterSize = static_cast<std::size_t>(2 * filterRadius + 1);
    if (filterWeightCount != filterSize * filterSize) {
        throw std::invalid_argument("smoothDemand: el filtro debe tener (2 * radio + 1)^2 pesos");
    }
}

std::vector<float> makeGaussianWeights(int filterRadius, double sigmaCells) {
    if (filterRadius < 0 || filterRadius > kMaxFilterRadius) {
        throw std::invalid_argument("makeGaussianWeights: radio fuera de 0.." + std::to_string(kMaxFilterRadius));
    }
    if (!(sigmaCells > 0.0)) {
        throw std::invalid_argument("makeGaussianWeights: sigma debe ser positivo");
    }
    const int filterSize = 2 * filterRadius + 1;
    std::vector<double> rawWeights(static_cast<std::size_t>(filterSize) * filterSize);
    double totalWeight = 0.0;
    for (int offsetY = -filterRadius; offsetY <= filterRadius; ++offsetY) {
        for (int offsetX = -filterRadius; offsetX <= filterRadius; ++offsetX) {
            const double squaredDistance = static_cast<double>(offsetX * offsetX + offsetY * offsetY);
            const double weight = std::exp(-squaredDistance / (2.0 * sigmaCells * sigmaCells));
            rawWeights[static_cast<std::size_t>(offsetY + filterRadius) * filterSize +
                       static_cast<std::size_t>(offsetX + filterRadius)] = weight;
            totalWeight += weight;
        }
    }
    std::vector<float> gaussianWeights(rawWeights.size());
    for (std::size_t i = 0; i < rawWeights.size(); ++i) {
        gaussianWeights[i] = static_cast<float>(rawWeights[i] / totalWeight);  // suman 1
    }
    return gaussianWeights;
}

std::vector<float> smoothDemandCpu1(const std::vector<float>& ordersPerCell, int gridWidth, int gridHeight,
                                    const std::vector<float>& filterWeights, int filterRadius) {
    checkDemandArguments(ordersPerCell.size(), gridWidth, gridHeight, filterWeights.size(), filterRadius);
    std::vector<float> demandHeat(ordersPerCell.size(), 0.0f);
    smoothDemandRows(ordersPerCell, demandHeat, gridWidth, gridHeight, filterWeights, filterRadius, 0,
                     static_cast<std::size_t>(gridHeight));
    return demandHeat;
}

std::vector<float> smoothDemandCpuN(ThreadPool& pool, const std::vector<float>& ordersPerCell, int gridWidth,
                                    int gridHeight, const std::vector<float>& filterWeights, int filterRadius) {
    checkDemandArguments(ordersPerCell.size(), gridWidth, gridHeight, filterWeights.size(), filterRadius);
    std::vector<float> demandHeat(ordersPerCell.size(), 0.0f);
    // Cada franja LEE cualquier fila de la entrada (que nadie escribe) y ESCRIBE solo sus filas de salida.
    pool.parallelFor(static_cast<std::size_t>(gridHeight), [&](std::size_t firstRow, std::size_t endRow) {
        smoothDemandRows(ordersPerCell, demandHeat, gridWidth, gridHeight, filterWeights, filterRadius, firstRow,
                         endRow);
    });
    return demandHeat;
}

std::vector<float> smoothDemandGpu(const std::vector<float>& ordersPerCell, int gridWidth, int gridHeight,
                                   const std::vector<float>& filterWeights, int filterRadius, DemandKernel kernel) {
    checkDemandArguments(ordersPerCell.size(), gridWidth, gridHeight, filterWeights.size(), filterRadius);
    DemandHeatGpu heatGpu;
    std::vector<float> demandHeat(ordersPerCell.size(), 0.0f);
    std::string errorMessage = heatGpu.allocate(gridWidth, gridHeight);
    if (errorMessage.empty()) errorMessage = heatGpu.setFilter(filterWeights, filterRadius);
    if (errorMessage.empty()) errorMessage = heatGpu.upload(ordersPerCell, nullptr);
    if (errorMessage.empty()) errorMessage = heatGpu.launch(kernel, nullptr);
    if (errorMessage.empty()) errorMessage = heatGpu.download(demandHeat, nullptr);
    if (!errorMessage.empty()) {
        throwIfGpuError("smoothDemandGpu");  // error pegajoso: el worker queda no saludable
        throw std::runtime_error("smoothDemandGpu: " + errorMessage);
    }
    return demandHeat;
}

DemandSmoothing smoothDemand(GpuWorker* gpu, ThreadPool& cpuPool, const std::vector<float>& ordersPerCell,
                             int gridWidth, int gridHeight, const std::vector<float>& filterWeights,
                             int filterRadius, std::int64_t gpuTimeoutMs) {
    checkDemandArguments(ordersPerCell.size(), gridWidth, gridHeight, filterWeights.size(), filterRadius);
    DemandSmoothing smoothing;
    if (gpu != nullptr) {
        try {
            const auto deadline = deadlineAfterMs(gpuTimeoutMs);
            // Por VALOR: la tarea puede seguir en cola o corriendo cuando esta funcion ya volvio.
            std::future<std::vector<float>> gpuResult = gpu->submit(
                [ordersPerCell, gridWidth, gridHeight, filterWeights, filterRadius] {
                    return smoothDemandGpu(ordersPerCell, gridWidth, gridHeight, filterWeights, filterRadius);
                },
                deadline);
            if (waitUntilDeadline(gpuResult, deadline)) {
                smoothing.demandHeat = gpuResult.get();  // ya esta lista: no bloquea
                smoothing.usedGpu = true;
                return smoothing;
            }
        } catch (const std::exception&) {
            // GpuUnavailableError, GpuTaskExpiredError, GpuFatalError o fallo de la tarea: respaldo CPU.
        }
    }
    smoothing.demandHeat = smoothDemandCpuN(cpuPool, ordersPerCell, gridWidth, gridHeight, filterWeights, filterRadius);
    smoothing.usedGpu = false;
    return smoothing;
}

double maxRelativeDifference(const std::vector<float>& reference, const std::vector<float>& candidate) {
    if (reference.size() != candidate.size()) {
        throw std::invalid_argument("maxRelativeDifference: tamanos distintos");
    }
    double worstDifference = 0.0;
    for (std::size_t cell = 0; cell < reference.size(); ++cell) {
        const double difference = std::fabs(static_cast<double>(candidate[cell]) - static_cast<double>(reference[cell]));
        const double relative = difference / (1.0 + std::fabs(static_cast<double>(reference[cell])));
        // NaN (un resultado corrupto) cuenta como diferencia infinita.
        worstDifference = std::isnan(relative) ? INFINITY : std::max(worstDifference, relative);
        if (std::isinf(worstDifference)) break;
    }
    return worstDifference;
}

std::size_t hottestCellIndex(const std::vector<float>& demandHeat) {
    std::size_t hottest = 0;
    for (std::size_t cell = 1; cell < demandHeat.size(); ++cell) {
        if (demandHeat[cell] > demandHeat[hottest]) hottest = cell;
    }
    return hottest;
}

}  // namespace accel
