// Aplicacion 05 (mapa de calor de la demanda): convolucion 2D con filtro gaussiano.
// Sin CUDA: prueba cpu1, cpuN y el respaldo de smoothDemand. La equivalencia GPU == CPU la prueba el
// bench (columna correct) con el build CUDA.
#include "accel/DemandHeatmap.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "DemandHeat.h"
#include "EventLog.h"
#include "OrderBook.h"
#include "accel/GpuWorker.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

// Filtro binomial 3x3 del material: 1 2 1 / 2 4 2 / 1 2 1, dividido por 16 (pesos exactos en binario).
std::vector<float> binomialWeights() {
    const std::vector<float> rawWeights{1, 2, 1, 2, 4, 2, 1, 2, 1};
    std::vector<float> weights;
    for (float rawWeight : rawWeights) weights.push_back(rawWeight / 16.0f);
    return weights;
}

// splitmix64: conteos deterministas segun la semilla.
std::vector<float> syntheticCounts(std::size_t cellCount, std::uint64_t seed) {
    std::vector<float> counts(cellCount);
    std::uint64_t state = seed;
    for (float& count : counts) {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t mixed = state;
        mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
        mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
        mixed ^= mixed >> 31;
        count = static_cast<float>(mixed % 4);
    }
    return counts;
}

bool throwsInvalidArgument(const std::vector<float>& orders, int gridWidth, int gridHeight,
                           const std::vector<float>& weights, int radius) {
    try {
        accel::smoothDemandCpu1(orders, gridWidth, gridHeight, weights, radius);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    // Ejemplo 6x6 del material con el filtro binomial 3x3.
    {
        const std::vector<float> ordersPerCell{0, 0, 0, 0, 0, 2,  //
                                               0, 1, 2, 1, 0, 1,  //
                                               0, 2, 6, 3, 0, 0,  //
                                               0, 1, 3, 2, 0, 0,  //
                                               0, 0, 0, 0, 0, 0,  //
                                               0, 0, 0, 0, 0, 0};
        const std::vector<float> expectedHeat{0.0625f, 0.2500f, 0.3750f, 0.2500f, 0.3750f, 0.6250f,  //
                                              0.2500f, 1.1250f, 1.8125f, 1.2500f, 0.5625f, 0.5000f,  //
                                              0.3750f, 1.8125f, 3.0625f, 2.1875f, 0.6250f, 0.1250f,  //
                                              0.2500f, 1.2500f, 2.1875f, 1.6250f, 0.4375f, 0.0000f,  //
                                              0.0625f, 0.3125f, 0.5625f, 0.4375f, 0.1250f, 0.0000f,  //
                                              0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f};
        const std::vector<float> demandHeat = accel::smoothDemandCpu1(ordersPerCell, 6, 6, binomialWeights(), 1);
        CHECK(demandHeat.size() == 36);
        CHECK(demandHeat[2 * 6 + 2] == 3.0625f);  // celda interior (x=2, y=2): 49 / 16
        CHECK(demandHeat[0 * 6 + 5] == 0.625f);   // esquina (x=5, y=0): cinco vecinas fuera del mapa cuentan 0
        for (std::size_t cell = 0; cell < demandHeat.size(); ++cell) CHECK(demandHeat[cell] == expectedHeat[cell]);
        float totalHeat = 0.0f;
        for (float cellHeat : demandHeat) totalHeat += cellHeat;
        CHECK(totalHeat == 22.875f);  // de 24 pedidos, 1,125 se fueron por el borde
        CHECK(accel::hottestCellIndex(demandHeat) == 2 * 6 + 2);
        // cpuN da exactamente lo mismo.
        accel::ThreadPool pool(4);
        const std::vector<float> demandHeatParallel =
            accel::smoothDemandCpuN(pool, ordersPerCell, 6, 6, binomialWeights(), 1);
        for (std::size_t cell = 0; cell < demandHeat.size(); ++cell) CHECK(demandHeatParallel[cell] == demandHeat[cell]);
    }

    // Borde = 0: una grilla de unos con filtro normalizado vale 1 adentro y menos en el borde.
    {
        const std::vector<float> weights = accel::makeGaussianWeights(3, 1.5);
        double weightSum = 0.0;
        for (float weight : weights) weightSum += weight;
        CHECK(std::fabs(weightSum - 1.0) < 1e-6);
        CHECK(weights.size() == 49);
        CHECK(weights[0] == weights[48] && weights[3] == weights[45] && weights[24] > weights[23]);  // simetrico
        const int side = 9;
        const std::vector<float> allOnes(static_cast<std::size_t>(side) * side, 1.0f);
        const std::vector<float> demandHeat = accel::smoothDemandCpu1(allOnes, side, side, weights, 3);
        CHECK(std::fabs(demandHeat[4 * side + 4] - 1.0f) < 1e-5f);  // centro: el filtro entero cabe
        // Esquina (0, 0): solo suman los desplazamientos con ambos ejes >= 0.
        double cornerExpected = 0.0;
        for (int offsetY = 0; offsetY <= 3; ++offsetY)
            for (int offsetX = 0; offsetX <= 3; ++offsetX) cornerExpected += weights[(offsetY + 3) * 7 + (offsetX + 3)];
        CHECK(std::fabs(demandHeat[0] - cornerExpected) < 1e-5);
        CHECK(demandHeat[0] < 0.6f);
        // Un solo pedido en la esquina: lo que cae fuera del mapa no aparece, la suma baja del total.
        std::vector<float> singleOrder(static_cast<std::size_t>(side) * side, 0.0f);
        singleOrder[0] = 1.0f;
        const std::vector<float> spread = accel::smoothDemandCpu1(singleOrder, side, side, weights, 3);
        double spreadTotal = 0.0;
        for (float cellHeat : spread) spreadTotal += cellHeat;
        CHECK(spreadTotal < 0.6 && spreadTotal > 0.0);
        CHECK(spread[0] == weights[3 * 7 + 3]);  // la celda del pedido pesa lo que el centro del filtro
    }

    // Grilla de 1 x 1 (y de una sola fila): todo el filtro excepto el centro cae fuera.
    {
        const std::vector<float> oneCell{5.0f};
        CHECK(accel::smoothDemandCpu1(oneCell, 1, 1, binomialWeights(), 1)[0] == 5.0f * 4.0f / 16.0f);
        CHECK(accel::smoothDemandCpu1(oneCell, 1, 1, std::vector<float>{1.0f}, 0)[0] == 5.0f);
        const std::vector<float> gaussian = accel::makeGaussianWeights(3, 1.5);
        CHECK(accel::smoothDemandCpu1(oneCell, 1, 1, gaussian, 3)[0] == 5.0f * gaussian[24]);
        accel::ThreadPool pool(8);  // mas hilos que filas: los trozos vacios se omiten
        CHECK(accel::smoothDemandCpuN(pool, oneCell, 1, 1, binomialWeights(), 1)[0] == 5.0f * 4.0f / 16.0f);
        // Fila de 3 celdas (1, 0, 2): solo cuenta la fila central del filtro (2 4 2) / 16.
        const std::vector<float> oneRow{1.0f, 0.0f, 2.0f};
        const std::vector<float> rowHeat = accel::smoothDemandCpu1(oneRow, 3, 1, binomialWeights(), 1);
        CHECK(rowHeat[0] == (4.0f * 1.0f + 2.0f * 0.0f) / 16.0f);
        CHECK(rowHeat[1] == (2.0f * 1.0f + 4.0f * 0.0f + 2.0f * 2.0f) / 16.0f);
        CHECK(rowHeat[2] == (2.0f * 0.0f + 4.0f * 2.0f) / 16.0f);
    }

    // cpu1 == cpuN con tolerancia (y en la practica exacto: mismo codigo), tamanos que no son multiplo de 16.
    {
        const std::vector<float> gaussian = accel::makeGaussianWeights(3, 1.5);
        const int sizes[][2] = {{128, 128}, {37, 23}, {17, 1}, {1, 19}, {5, 2}, {100, 100}};
        for (const auto& size : sizes) {
            const int gridWidth = size[0];
            const int gridHeight = size[1];
            const std::vector<float> counts = syntheticCounts(static_cast<std::size_t>(gridWidth) * gridHeight, 42);
            const std::vector<float> reference = accel::smoothDemandCpu1(counts, gridWidth, gridHeight, gaussian, 3);
            for (unsigned threadCount : {1u, 2u, 3u, 8u, 16u}) {
                accel::ThreadPool pool(threadCount);
                const std::vector<float> parallel =
                    accel::smoothDemandCpuN(pool, counts, gridWidth, gridHeight, gaussian, 3);
                CHECK(accel::maxRelativeDifference(reference, parallel) <= accel::kDemandTolerance);
                CHECK(parallel == reference);
            }
        }
    }

    // Argumentos invalidos: se rechazan, no se leen ni escriben fuera de los vectores.
    {
        const std::vector<float> cells(4, 1.0f);
        const std::vector<float> weights = binomialWeights();
        CHECK(!throwsInvalidArgument(cells, 2, 2, weights, 1));
        CHECK(throwsInvalidArgument(cells, 0, 4, weights, 1));
        CHECK(throwsInvalidArgument(cells, 2, 0, weights, 1));
        CHECK(throwsInvalidArgument(cells, -2, -2, weights, 1));
        CHECK(throwsInvalidArgument(cells, 2, 3, weights, 1));  // 4 celdas para 2 x 3
        CHECK(throwsInvalidArgument(cells, 2, 2, weights, 2));  // 9 pesos para radio 2 (25)
        CHECK(throwsInvalidArgument(cells, 2, 2, weights, -1));
        CHECK(throwsInvalidArgument(cells, 2, 2, std::vector<float>(81, 0.1f), 4));  // radio > maximo
        CHECK(throwsInvalidArgument(cells, accel::kMaxGridSide + 1, 1, weights, 1));
        CHECK(throwsInvalidArgument(std::vector<float>(), 1, 1, weights, 1));
        bool threwForSigma = false;
        try {
            accel::makeGaussianWeights(3, 0.0);
        } catch (const std::invalid_argument&) {
            threwForSigma = true;
        }
        CHECK(threwForSigma);
        CHECK(accel::makeGaussianWeights(0, 1.5) == std::vector<float>{1.0f});
    }

    // maxRelativeDifference y hottestCellIndex.
    {
        CHECK(accel::maxRelativeDifference({1.0f, 3.0f}, {1.0f, 3.0f}) == 0.0);
        CHECK(std::fabs(accel::maxRelativeDifference({0.0f}, {0.5f}) - 0.5) < 1e-12);   // celda vacia: absoluta
        CHECK(std::fabs(accel::maxRelativeDifference({3.0f}, {4.0f}) - 0.25) < 1e-12);  // |d| / (1 + |ref|)
        bool threwForSize = false;
        try {
            accel::maxRelativeDifference({1.0f}, {1.0f, 2.0f});
        } catch (const std::invalid_argument&) {
            threwForSize = true;
        }
        CHECK(threwForSize);
        CHECK(accel::hottestCellIndex({}) == 0);
        CHECK(accel::hottestCellIndex({0.0f, 0.0f, 0.0f}) == 0);
        CHECK(accel::hottestCellIndex({1.0f, 5.0f, 5.0f, 2.0f}) == 1);  // empate: el menor indice
    }

    // smoothDemand: sin GPU, con una GPU que no existe en este build (respaldo) y con un worker roto.
    {
        const std::vector<float> gaussian = accel::makeGaussianWeights(3, 1.5);
        const std::vector<float> counts = syntheticCounts(64 * 64, 7);
        const std::vector<float> reference = accel::smoothDemandCpu1(counts, 64, 64, gaussian, 3);
        accel::ThreadPool pool(3);

        const accel::DemandSmoothing withoutGpu = accel::smoothDemand(nullptr, pool, counts, 64, 64, gaussian, 3, 200);
        CHECK(!withoutGpu.usedGpu);
        CHECK(withoutGpu.demandHeat == reference);

        accel::GpuWorker healthyWorker([] {});  // en el build sin CUDA la tarea falla: respaldo CPU
        const accel::DemandSmoothing viaWorker =
            accel::smoothDemand(&healthyWorker, pool, counts, 64, 64, gaussian, 3, 200);
        CHECK(accel::maxRelativeDifference(reference, viaWorker.demandHeat) <= accel::kDemandTolerance);

        accel::GpuWorker brokenWorker([] { throw std::runtime_error("sin GPU"); });
        CHECK(!brokenWorker.healthy());
        const accel::DemandSmoothing viaBrokenWorker =
            accel::smoothDemand(&brokenWorker, pool, counts, 64, 64, gaussian, 3, 200);
        CHECK(!viaBrokenWorker.usedGpu);
        CHECK(viaBrokenWorker.demandHeat == reference);

        accel::GpuWorker stoppedWorker([] {});
        stoppedWorker.stop();
        CHECK(!accel::smoothDemand(&stoppedWorker, pool, counts, 64, 64, gaussian, 3, 200).usedGpu);

        // Plazo 0: vence antes de empezar; el resultado sigue siendo correcto.
        const accel::DemandSmoothing zeroDeadline =
            accel::smoothDemand(&healthyWorker, pool, counts, 64, 64, gaussian, 3, 0);
        CHECK(accel::maxRelativeDifference(reference, zeroDeadline.demandHeat) <= accel::kDemandTolerance);

        // Argumentos invalidos: se rechazan aunque haya GPU.
        bool threw = false;
        try {
            accel::smoothDemand(&healthyWorker, pool, counts, 63, 64, gaussian, 3, 200);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }

    // Mapa de la grilla: fila 0 = norte, recorte al borde, conteo y linea de estadisticas.
    {
        Bounds bounds;
        bounds.north = -17.0;
        bounds.south = -18.0;
        bounds.west = -63.0;
        bounds.east = -62.0;
        CHECK(demandCellIndex(bounds, 10, 10, -17.0, -63.0) == 0);                  // esquina noroeste
        CHECK(demandCellIndex(bounds, 10, 10, -17.99, -62.01) == 9 * 10 + 9);        // esquina sureste
        CHECK(demandCellIndex(bounds, 10, 10, -17.55, -62.55) == 5 * 10 + 4);
        CHECK(demandCellIndex(bounds, 10, 10, -10.0, -70.0) == 0);                   // fuera: recorta al borde
        CHECK(demandCellIndex(bounds, 10, 10, -30.0, -50.0) == 9 * 10 + 9);
        CHECK(demandCellIndex(bounds, 1, 1, -17.5, -62.5) == 0);
        const std::vector<float> counts =
            countPointsPerCell(bounds, 10, 10, {{-17.55, -62.55}, {-17.56, -62.56}, {-17.0, -63.0}});
        CHECK(counts.size() == 100 && counts[54] == 2.0f && counts[0] == 1.0f);
        const auto center = demandCellCenter(bounds, 10, 10, 4, 5);
        CHECK(std::fabs(center.first - -17.55) < 1e-9 && std::fabs(center.second - -62.55) < 1e-9);
        std::vector<float> heat(100, 0.0f);
        CHECK(hottestZoneLine(bounds, 10, 10, heat) == "[estadisticas] zona mas caliente: sin pedidos recientes");
        heat[54] = 2.5f;
        CHECK(hottestZoneLine(bounds, 10, 10, heat) ==
              "[estadisticas] zona mas caliente: celda (4,5) ~ -17.55000,-62.55000, valor 2.500");
    }

    // OrderBook::dropoffNodesCreatedSince: solo los creados desde el instante pedido (incluidos los rechazados).
    {
        const SimClock clock(60.0);
        EventLog log("tests/bin/demand_book.log", clock);
        OrderBook book(2, clock, log);
        Order first;
        first.id = 1;
        first.restaurantId = "r0";
        first.pickupNode = "n1";
        first.dropoffNode = "n10";
        Order second = first;
        second.id = 2;
        second.dropoffNode = "n20";
        Order rejected = first;
        rejected.id = 3;
        rejected.dropoffNode = "n30";
        book.admit(first);
        book.admit(second);
        book.admit(rejected);  // la cola esta llena: rechazado, pero es demanda
        CHECK(book.dropoffNodesCreatedSince(0).size() == 3);
        CHECK(book.dropoffNodesCreatedSince(INT64_MAX).empty());
    }

    std::cout << "test_demand_heatmap OK\n";
    return 0;
}
