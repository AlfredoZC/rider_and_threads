#include "accel/CoverageField.h"

#include <algorithm>
#include <future>
#include <stdexcept>
#include <utility>

#include "accel/DemandHeatmap.h"
#include "accel/GpuContext.h"
#include "accel/GpuError.h"
#include "accel/GpuWorker.h"

namespace accel {

namespace {

// UN barrido de Jacobi sobre las filas [firstRow, endRow): LEE solo coverageBefore y ESCRIBE solo coverageAfter.
// Es la unica funcion de barrido de la CPU: cpu1 la llama con todas las filas y cpuN con una franja por trozo.
// El orden de las condiciones importa: primero el borde, despues las fuentes, por eso el programa cuenta los
// couriers con margen 1 (ninguno cae en el borde, donde se borraria).
void coverageSweepRows(const std::vector<float>& coverageBefore, std::vector<float>& coverageAfter,
                       const std::vector<float>& couriersPerCell, int gridWidth, int gridHeight, std::size_t firstRow,
                       std::size_t endRow) {
    const std::size_t width = static_cast<std::size_t>(gridWidth);
    for (std::size_t cellY = firstRow; cellY < endRow; ++cellY) {
        for (std::size_t cellX = 0; cellX < width; ++cellX) {
            const std::size_t cell = cellY * width + cellX;
            const bool onBorder = cellX == 0 || cellY == 0 || cellX == width - 1 || cellY == static_cast<std::size_t>(gridHeight) - 1;
            if (onBorder) {
                coverageAfter[cell] = 0.0f;  // borde fijo en 0
            } else if (couriersPerCell[cell] > 0.0f) {
                coverageAfter[cell] = couriersPerCell[cell];  // fuente fija
            } else {  // libre: promedio de las 4 vecinas del barrido anterior
                coverageAfter[cell] = 0.25f * (coverageBefore[cell - 1] + coverageBefore[cell + 1] +
                                               coverageBefore[cell - width] + coverageBefore[cell + width]);
            }
        }
    }
}

}  // namespace

void checkCoverageArguments(std::size_t couriersCellCount, int gridWidth, int gridHeight, int sweepCount) {
    if (gridWidth < 1 || gridHeight < 1 || gridWidth > kMaxGridSide || gridHeight > kMaxGridSide) {
        throw std::invalid_argument("coverageField: el lado de la grilla debe estar en 1.." + std::to_string(kMaxGridSide));
    }
    if (couriersCellCount != static_cast<std::size_t>(gridWidth) * static_cast<std::size_t>(gridHeight)) {
        throw std::invalid_argument("coverageField: los couriers no coinciden con ancho * alto");
    }
    if (sweepCount < 0 || sweepCount > kMaxSweepCount) {
        throw std::invalid_argument("coverageField: los barridos deben estar en 0.." + std::to_string(kMaxSweepCount));
    }
}

std::vector<float> coverageFieldCpu1(const std::vector<float>& couriersPerCell, int gridWidth, int gridHeight,
                                     int sweepCount) {
    checkCoverageArguments(couriersPerCell.size(), gridWidth, gridHeight, sweepCount);
    std::vector<float> coverageBefore = couriersPerCell;  // barrido 0: fuentes y ceros
    std::vector<float> coverageAfter(couriersPerCell.size(), 0.0f);
    for (int sweep = 0; sweep < sweepCount; ++sweep) {
        coverageSweepRows(coverageBefore, coverageAfter, couriersPerCell, gridWidth, gridHeight, 0,
                          static_cast<std::size_t>(gridHeight));
        std::swap(coverageBefore, coverageAfter);  // O(1): cambia punteros, no copia
    }
    return coverageBefore;  // tras el ultimo swap, el resultado mas nuevo
}

std::vector<float> coverageFieldCpuN(ThreadPool& pool, const std::vector<float>& couriersPerCell, int gridWidth,
                                     int gridHeight, int sweepCount) {
    checkCoverageArguments(couriersPerCell.size(), gridWidth, gridHeight, sweepCount);
    std::vector<float> coverageBefore = couriersPerCell;
    std::vector<float> coverageAfter(couriersPerCell.size(), 0.0f);
    for (int sweep = 0; sweep < sweepCount; ++sweep) {
        // Durante el barrido coverageBefore solo se LEE y cada celda de coverageAfter la escribe UNA franja.
        // parallelFor no vuelve hasta que terminaron todas: es la barrera entre barridos.
        pool.parallelFor(static_cast<std::size_t>(gridHeight), [&](std::size_t firstRow, std::size_t endRow) {
            coverageSweepRows(coverageBefore, coverageAfter, couriersPerCell, gridWidth, gridHeight, firstRow, endRow);
        });
        std::swap(coverageBefore, coverageAfter);  // ya nadie lee ni escribe
    }
    return coverageBefore;
}

std::vector<float> coverageFieldGpu(const std::vector<float>& couriersPerCell, int gridWidth, int gridHeight,
                                    int sweepCount) {
    checkCoverageArguments(couriersPerCell.size(), gridWidth, gridHeight, sweepCount);
    CoverageGpu coverageGpu;
    std::vector<float> coverage(couriersPerCell.size(), 0.0f);
    std::string errorMessage = coverageGpu.allocate(gridWidth, gridHeight);
    if (errorMessage.empty()) errorMessage = coverageGpu.upload(couriersPerCell, nullptr);
    if (errorMessage.empty()) errorMessage = coverageGpu.run(sweepCount, nullptr);
    if (errorMessage.empty()) errorMessage = coverageGpu.download(coverage, nullptr);
    if (!errorMessage.empty()) {
        throwIfGpuError("coverageFieldGpu");  // error pegajoso: el worker queda no saludable
        throw std::runtime_error("coverageFieldGpu: " + errorMessage);
    }
    return coverage;
}

CoverageResult computeCoverage(GpuWorker* gpu, ThreadPool& cpuPool, const std::vector<float>& couriersPerCell,
                               int gridWidth, int gridHeight, int sweepCount, std::int64_t gpuTimeoutMs) {
    checkCoverageArguments(couriersPerCell.size(), gridWidth, gridHeight, sweepCount);
    CoverageResult result;
    if (gpu != nullptr) {
        try {
            const auto deadline = deadlineAfterMs(gpuTimeoutMs);
            // Por VALOR: la tarea puede seguir en cola o corriendo cuando esta funcion ya volvio.
            std::future<std::vector<float>> gpuResult = gpu->submit(
                [couriersPerCell, gridWidth, gridHeight, sweepCount] {
                    return coverageFieldGpu(couriersPerCell, gridWidth, gridHeight, sweepCount);
                },
                deadline);
            if (waitUntilDeadline(gpuResult, deadline)) {
                result.coverage = gpuResult.get();  // ya esta lista: no bloquea
                result.usedGpu = true;
                return result;
            }
        } catch (const std::exception&) {
            // GpuUnavailableError, GpuTaskExpiredError, GpuFatalError o fallo de la tarea: respaldo CPU.
        }
    }
    result.coverage = coverageFieldCpuN(cpuPool, couriersPerCell, gridWidth, gridHeight, sweepCount);
    result.usedGpu = false;
    return result;
}

CoverageZone leastCoveredZone(const std::vector<float>& coverage, int gridWidth, int gridHeight, int zoneSide) {
    if (zoneSide < 1) {
        throw std::invalid_argument("leastCoveredZone: el lado de la zona debe ser >= 1");
    }
    checkCoverageArguments(coverage.size(), gridWidth, gridHeight, 0);
    const std::size_t width = static_cast<std::size_t>(gridWidth);
    const std::size_t height = static_cast<std::size_t>(gridHeight);
    const std::size_t side = static_cast<std::size_t>(zoneSide);
    const std::size_t zonesAcross = (width + side - 1) / side;
    const std::size_t zonesDown = (height + side - 1) / side;
    CoverageZone worst;
    bool first = true;
    for (std::size_t zoneY = 0; zoneY < zonesDown; ++zoneY) {
        for (std::size_t zoneX = 0; zoneX < zonesAcross; ++zoneX) {
            const std::size_t firstX = zoneX * side;
            const std::size_t firstY = zoneY * side;
            const std::size_t endX = std::min(width, firstX + side);
            const std::size_t endY = std::min(height, firstY + side);
            double coverageSum = 0.0;
            for (std::size_t cellY = firstY; cellY < endY; ++cellY)
                for (std::size_t cellX = firstX; cellX < endX; ++cellX) coverageSum += coverage[cellY * width + cellX];
            const float meanCoverage = static_cast<float>(coverageSum / static_cast<double>((endX - firstX) * (endY - firstY)));
            if (first || meanCoverage < worst.meanCoverage) {
                worst.zoneX = static_cast<int>(zoneX);
                worst.zoneY = static_cast<int>(zoneY);
                worst.meanCoverage = meanCoverage;
            }
            first = false;
        }
    }
    return worst;
}

}  // namespace accel
