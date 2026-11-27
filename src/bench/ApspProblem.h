#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CityGraph.h"
#include "accel/TravelTimeTable.h"
#include "bench/BenchApp.h"

namespace bench {

// Piezas comunes de las tres aplicaciones de la tabla de tiempos (apsp_floyd, apsp_floyd_naive,
// apsp_dijkstra): el problema (grafo, matriz inicial, referencia) y las corridas de GPU. Sin estado global.

// Subgrafos de la ciudad real del barrido (regiones anidadas) y tope de --size (el de la tabla).
std::vector<std::size_t> apspDefaultSizes();
inline constexpr std::size_t kApspMaxSize = accel::kMaxTableNodes;

// Con que se calcula la referencia (la tabla correcta) en prepare.
//   Dijkstra       Dijkstra desde cada nodo con 1 hilo: otro algoritmo, asi un error de Floyd-Warshall no se tapa solo
//   FloydWarshall  Floyd-Warshall de 1 hilo: referencia independiente para la app que SON los Dijkstra
enum class ApspReference { Dijkstra, FloydWarshall };

struct ApspProblem {
    std::size_t nodeCount = 0;
    bool fromCity = false;  // true: region de la ciudad real; false: grafo sintetico
    std::unique_ptr<CityGraph> graph;
    std::vector<std::int32_t> initialCells;    // matriz inicial de Floyd-Warshall (nodeCount x nodeCount)
    std::vector<std::int32_t> referenceCells;  // la tabla correcta
    std::vector<std::int32_t> resultCells;     // donde escribe cada corrida

    void clear();
};

// Arma el problema de `size` nodos: region de la ciudad (o grafo sintetico sin ciudad), matriz inicial y
// referencia. Devuelve "" si todo bien o un mensaje (tamano fuera de 1..kApspMaxSize, sin memoria, tiempos
// que no caben). `appName` va en los mensajes. Un prepare que falla deja el problema vacio.
std::string prepareApspProblem(const std::string& appName, std::size_t size, std::uint32_t seed,
                               const BenchContext& context, ApspReference reference, ApspProblem& problem);

RunResult failedRun(const std::string& message);

// Marca `correct` comparando problem.resultCells con la referencia (exacto: enteros).
void markCorrectness(const ApspProblem& problem, RunResult& runResult);

// Una funcion de GPU de la tabla (floydWarshallTiledGpu o floydWarshallNaiveGpu).
using GpuFloydFunction = void (*)(const std::vector<std::int32_t>&, std::size_t, std::vector<std::int32_t>&,
                                  accel::GpuFloydTiming*);

// Corre la version GPU pedida (GpuKernel: ms = kernel con cudaEvent; GpuE2e: ms = steady_clock de todo el
// envoltorio, incluidos relleno, cudaMalloc y copias, y h2d/kernel/d2h con cudaEvent) y marca `correct`.
// Nunca lanza: un error CUDA o de memoria vuelve como RunResult con ok = false.
RunResult runGpuFloyd(ApspProblem& problem, GpuFloydFunction gpuFunction, Version version, const char* label);

// Metrica derivada de las medianas: relajaciones n^3 por segundo (Floyd-Warshall) y celdas por segundo.
std::string describeRelaxations(const SummaryStats& stats);
std::string describeCells(const SummaryStats& stats);

}  // namespace bench
