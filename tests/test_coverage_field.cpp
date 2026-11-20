// Aplicacion 06 (campo de cobertura de la flota): stencil de 5 puntos con barridos de Jacobi y doble buffer.
// Sin CUDA: prueba cpu1, cpuN y el respaldo de computeCoverage. La equivalencia GPU == CPU la prueba el
// bench (columna correct) con el build CUDA.
#include "accel/CoverageField.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "CoverageLayer.h"
#include "DemandHeat.h"
#include "accel/DemandHeatmap.h"
#include "accel/GpuWorker.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

// splitmix64: fuentes deterministas segun la semilla (solo en celdas interiores, como el programa real).
std::vector<float> syntheticCouriers(int gridWidth, int gridHeight, std::uint64_t seed) {
    std::vector<float> couriers(static_cast<std::size_t>(gridWidth) * gridHeight, 0.0f);
    std::uint64_t state = seed;
    for (int cellY = 1; cellY < gridHeight - 1; ++cellY) {
        for (int cellX = 1; cellX < gridWidth - 1; ++cellX) {
            state += 0x9E3779B97F4A7C15ull;
            std::uint64_t mixed = state;
            mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
            mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
            mixed ^= mixed >> 31;
            if (mixed % 16 == 0) couriers[static_cast<std::size_t>(cellY) * gridWidth + cellX] = 1.0f + (mixed >> 8) % 3;
        }
    }
    return couriers;
}

std::vector<float> centerSource5x5() {
    std::vector<float> couriers(25, 0.0f);
    couriers[2 * 5 + 2] = 4.0f;  // fuente con 4 couriers en el centro
    return couriers;
}

bool throwsInvalidArgument(const std::vector<float>& couriers, int gridWidth, int gridHeight, int sweepCount) {
    try {
        accel::coverageFieldCpu1(couriers, gridWidth, gridHeight, sweepCount);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    // Ejemplo 5x5 del material: borde en 0, una fuente con 4 couriers en el centro; barridos 0, 1, 2 y 3.
    {
        const std::vector<float> couriers = centerSource5x5();
        CHECK(accel::coverageFieldCpu1(couriers, 5, 5, 0) == couriers);  // barrido 0 = la grilla de couriers
        const std::vector<float> sweep1{0, 0, 0,    0, 0,  //
                                        0, 0, 1,    0, 0,  //
                                        0, 1, 4,    1, 0,  //
                                        0, 0, 1,    0, 0,  //
                                        0, 0, 0,    0, 0};
        const std::vector<float> sweep2{0, 0,    0, 0,    0,  //
                                        0, 0.5f, 1, 0.5f, 0,  //
                                        0, 1,    4, 1,    0,  //
                                        0, 0.5f, 1, 0.5f, 0,  //
                                        0, 0,    0, 0,    0};
        const std::vector<float> sweep3{0, 0,    0,     0,    0,  //
                                        0, 0.5f, 1.25f, 0.5f, 0,  //
                                        0, 1.25f, 4,    1.25f, 0,  //
                                        0, 0.5f, 1.25f, 0.5f, 0,  //
                                        0, 0,    0,     0,    0};
        CHECK(accel::coverageFieldCpu1(couriers, 5, 5, 1) == sweep1);
        CHECK(accel::coverageFieldCpu1(couriers, 5, 5, 2) == sweep2);
        CHECK(accel::coverageFieldCpu1(couriers, 5, 5, 3) == sweep3);
        accel::ThreadPool pool(4);
        CHECK(accel::coverageFieldCpuN(pool, couriers, 5, 5, 1) == sweep1);
        CHECK(accel::coverageFieldCpuN(pool, couriers, 5, 5, 2) == sweep2);
        CHECK(accel::coverageFieldCpuN(pool, couriers, 5, 5, 3) == sweep3);
        // Equilibrio: vecinas directas 4/3 y diagonales 2/3.
        const std::vector<float> converged = accel::coverageFieldCpu1(couriers, 5, 5, 60);
        CHECK(std::fabs(converged[1 * 5 + 2] - 4.0f / 3.0f) < 1e-4f);
        CHECK(std::fabs(converged[1 * 5 + 1] - 2.0f / 3.0f) < 1e-4f);
        CHECK(converged[2 * 5 + 2] == 4.0f);
    }

    // Bordes: lo que cae en el borde se borra a 0 desde el primer barrido; las fuentes interiores quedan fijas.
    {
        std::vector<float> couriers(25, 0.0f);
        couriers[0] = 3.0f;          // esquina
        couriers[2] = 2.0f;          // borde de arriba
        couriers[2 * 5 + 4] = 5.0f;  // borde derecho
        couriers[1 * 5 + 1] = 2.0f;  // interior: fuente fija
        for (int sweepCount : {1, 2, 10}) {
            const std::vector<float> field = accel::coverageFieldCpu1(couriers, 5, 5, sweepCount);
            for (int cellY = 0; cellY < 5; ++cellY)
                for (int cellX = 0; cellX < 5; ++cellX) {
                    const bool onBorder = cellX == 0 || cellY == 0 || cellX == 4 || cellY == 4;
                    if (onBorder) CHECK(field[static_cast<std::size_t>(cellY) * 5 + cellX] == 0.0f);
                }
            CHECK(field[1 * 5 + 1] == 2.0f);
        }
        // Grillas sin interior (1x1, 1xN, 2x2): todo es borde y queda en 0.
        CHECK(accel::coverageFieldCpu1({7.0f}, 1, 1, 5) == std::vector<float>{0.0f});
        CHECK(accel::coverageFieldCpu1({1, 2, 3}, 3, 1, 4) == std::vector<float>(3, 0.0f));
        CHECK(accel::coverageFieldCpu1({1, 2, 3, 4}, 1, 4, 1) == std::vector<float>(4, 0.0f));
        CHECK(accel::coverageFieldCpu1({1, 1, 1, 1}, 2, 2, 3) == std::vector<float>(4, 0.0f));
        // 3x3: un solo interior, una fuente: queda fija; sin fuente: promedio de 4 ceros.
        std::vector<float> threeByThree(9, 0.0f);
        CHECK(accel::coverageFieldCpu1(threeByThree, 3, 3, 3) == threeByThree);
        threeByThree[4] = 2.0f;
        CHECK(accel::coverageFieldCpu1(threeByThree, 3, 3, 3) == threeByThree);
        // Barrido 0 con cualquier valor: se devuelve tal cual (nada se calcula).
        CHECK(accel::coverageFieldCpu1(couriers, 5, 5, 0) == couriers);
    }

    // Un solo courier: el campo es no negativo, simetrico y nunca supera a la fuente.
    {
        const int side = 11;
        std::vector<float> couriers(static_cast<std::size_t>(side) * side, 0.0f);
        couriers[5 * side + 5] = 1.0f;
        const std::vector<float> field = accel::coverageFieldCpu1(couriers, side, side, 50);
        for (int cellY = 0; cellY < side; ++cellY)
            for (int cellX = 0; cellX < side; ++cellX) {
                const float value = field[static_cast<std::size_t>(cellY) * side + cellX];
                CHECK(value >= 0.0f && value <= 1.0f);
                CHECK(std::fabs(value - field[static_cast<std::size_t>(cellX) * side + cellY]) < 1e-6f);  // transpuesta
                CHECK(std::fabs(value - field[static_cast<std::size_t>(side - 1 - cellY) * side + (side - 1 - cellX)]) < 1e-6f);  // giro de 180 (el orden de sumas no es simetrico)
            }
        CHECK(field[5 * side + 5] == 1.0f);
        CHECK(field[5 * side + 4] > field[5 * side + 3] && field[5 * side + 3] > field[5 * side + 1]);  // baja con la distancia
    }

    // cpu1 == cpuN con tolerancia (y en la practica exacto): tamanos que no son multiplo de 16, varios barridos.
    {
        const int sizes[][2] = {{128, 128}, {37, 23}, {17, 3}, {3, 19}, {5, 5}, {100, 100}, {2, 9}};
        for (const auto& size : sizes) {
            const int gridWidth = size[0];
            const int gridHeight = size[1];
            const std::vector<float> couriers = syntheticCouriers(gridWidth, gridHeight, 42);
            for (int sweepCount : {1, 7, 200}) {
                const std::vector<float> reference = accel::coverageFieldCpu1(couriers, gridWidth, gridHeight, sweepCount);
                for (unsigned threadCount : {1u, 2u, 3u, 8u, 16u}) {
                    accel::ThreadPool pool(threadCount);
                    const std::vector<float> parallel =
                        accel::coverageFieldCpuN(pool, couriers, gridWidth, gridHeight, sweepCount);
                    CHECK(accel::maxRelativeDifference(reference, parallel) <= accel::kCoverageTolerance);
                    CHECK(parallel == reference);
                }
            }
        }
    }

    // Argumentos invalidos.
    {
        const std::vector<float> cells(4, 1.0f);
        CHECK(!throwsInvalidArgument(cells, 2, 2, 1));
        CHECK(throwsInvalidArgument(cells, 0, 4, 1));
        CHECK(throwsInvalidArgument(cells, 2, 0, 1));
        CHECK(throwsInvalidArgument(cells, -2, -2, 1));
        CHECK(throwsInvalidArgument(cells, 2, 3, 1));      // 4 celdas para 2 x 3
        CHECK(throwsInvalidArgument(cells, 2, 2, -1));     // barridos negativos
        CHECK(throwsInvalidArgument(cells, 2, 2, accel::kMaxSweepCount + 1));
        CHECK(throwsInvalidArgument(cells, accel::kMaxGridSide + 1, 1, 1));
        CHECK(throwsInvalidArgument(std::vector<float>(), 1, 1, 1));
    }

    // computeCoverage: sin GPU, con una GPU que no existe en este build (respaldo) y con un worker roto.
    {
        const std::vector<float> couriers = syntheticCouriers(64, 64, 7);
        const std::vector<float> reference = accel::coverageFieldCpu1(couriers, 64, 64, 50);
        accel::ThreadPool pool(3);

        const accel::CoverageResult withoutGpu = accel::computeCoverage(nullptr, pool, couriers, 64, 64, 50, 200);
        CHECK(!withoutGpu.usedGpu);
        CHECK(withoutGpu.coverage == reference);

        accel::GpuWorker healthyWorker([] {});  // en el build sin CUDA la tarea falla: respaldo CPU
        const accel::CoverageResult viaWorker = accel::computeCoverage(&healthyWorker, pool, couriers, 64, 64, 50, 200);
        CHECK(accel::maxRelativeDifference(reference, viaWorker.coverage) <= accel::kCoverageTolerance);

        accel::GpuWorker brokenWorker([] { throw std::runtime_error("sin GPU"); });
        CHECK(!brokenWorker.healthy());
        const accel::CoverageResult viaBroken = accel::computeCoverage(&brokenWorker, pool, couriers, 64, 64, 50, 200);
        CHECK(!viaBroken.usedGpu);
        CHECK(viaBroken.coverage == reference);

        accel::GpuWorker stoppedWorker([] {});
        stoppedWorker.stop();
        CHECK(!accel::computeCoverage(&stoppedWorker, pool, couriers, 64, 64, 50, 200).usedGpu);

        const accel::CoverageResult zeroDeadline = accel::computeCoverage(&healthyWorker, pool, couriers, 64, 64, 50, 0);
        CHECK(accel::maxRelativeDifference(reference, zeroDeadline.coverage) <= accel::kCoverageTolerance);

        bool threw = false;
        try {
            accel::computeCoverage(&healthyWorker, pool, couriers, 63, 64, 50, 200);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }

    // Zona menos cubierta.
    {
        // 4 x 4, zonas de 2 x 2: la (1, 0) es la peor.
        const std::vector<float> field{5, 5, 1, 1,  //
                                       5, 5, 1, 0,  //
                                       4, 4, 3, 3,  //
                                       4, 4, 3, 3};
        const accel::CoverageZone worst = accel::leastCoveredZone(field, 4, 4, 2);
        CHECK(worst.zoneX == 1 && worst.zoneY == 0 && worst.meanCoverage == 0.75f);
        // Empate: gana la primera por filas.
        const accel::CoverageZone tie = accel::leastCoveredZone(std::vector<float>(16, 2.0f), 4, 4, 2);
        CHECK(tie.zoneX == 0 && tie.zoneY == 0 && tie.meanCoverage == 2.0f);
        // Lado que no divide la grilla: la zona del borde es mas chica y su media usa sus celdas reales.
        std::vector<float> uneven(25, 3.0f);
        uneven[4 * 5 + 4] = 0.0f;  // la zona (2, 2) tiene una sola celda: media 0
        const accel::CoverageZone corner = accel::leastCoveredZone(uneven, 5, 5, 2);
        CHECK(corner.zoneX == 2 && corner.zoneY == 2 && corner.meanCoverage == 0.0f);
        // Zona mas grande que la grilla: una sola zona con todo.
        CHECK(accel::leastCoveredZone(std::vector<float>(6, 1.0f), 3, 2, 16).meanCoverage == 1.0f);
        bool threw = false;
        try {
            accel::leastCoveredZone(field, 4, 4, 0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
        threw = false;
        try {
            accel::leastCoveredZone(field, 4, 3, 2);  // el tamano no coincide
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }

    // Conteo con margen 1: ningun courier cae en el borde fijo en 0.
    {
        Bounds bounds;
        bounds.north = -17.0;
        bounds.south = -18.0;
        bounds.west = -63.0;
        bounds.east = -62.0;
        const std::vector<std::pair<double, double>> points{{-17.0, -63.0}, {-18.0, -62.0}, {-17.55, -62.55}, {-10.0, -70.0}};
        const std::vector<float> counts = countPointsPerCell(bounds, 10, 10, points, 1);
        CHECK(counts[1 * 10 + 1] == 2.0f);  // noroeste y fuera: ambos a la primera celda interior
        CHECK(counts[8 * 10 + 8] == 1.0f);  // sureste
        CHECK(counts[5 * 10 + 4] == 1.0f);
        for (int cellY = 0; cellY < 10; ++cellY)
            for (int cellX = 0; cellX < 10; ++cellX)
                if (cellX == 0 || cellY == 0 || cellX == 9 || cellY == 9) CHECK(counts[static_cast<std::size_t>(cellY) * 10 + cellX] == 0.0f);
        // Con margen 0 (demanda) el comportamiento anterior no cambia.
        CHECK(countPointsPerCell(bounds, 10, 10, points)[0] == 2.0f);
        // Grilla demasiado chica para margen 1: se recorta sin salirse.
        CHECK(countPointsPerCell(bounds, 2, 2, points, 1).size() == 4);
        const auto position = interpolateLatLon({-17.0, -63.0}, {-18.0, -62.0}, 0.5);
        CHECK(std::fabs(position.first - -17.5) < 1e-12 && std::fabs(position.second - -62.5) < 1e-12);
    }

    std::cout << "test_coverage_field OK\n";
    return 0;
}
