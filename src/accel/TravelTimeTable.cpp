#include "accel/TravelTimeTable.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include "Router.h"
#include "accel/GpuWorker.h"

namespace accel {

namespace {

using Clock = std::chrono::steady_clock;

// Una pasada de Floyd-Warshall: permite pasar por `viaNode` para las filas [firstRow, endRow). Cada fila
// solo escribe en si misma y solo lee la fila `viaNode`, que en esta pasada nadie escribe (D[via][via] = 0,
// asi que la celda de esa fila nunca mejora): por eso las franjas de filas se pueden repartir entre hilos.
void relaxRowsThroughVia(std::int32_t* cells, std::size_t nodeCount, std::size_t viaNode, std::size_t firstRow,
                         std::size_t endRow) {
    const std::int32_t* viaRow = cells + cellIndex(viaNode, 0, nodeCount);
    for (std::size_t fromNode = firstRow; fromNode < endRow; ++fromNode) {
        std::int32_t* fromRow = cells + cellIndex(fromNode, 0, nodeCount);
        const std::int32_t timeToVia = fromRow[viaNode];
        if (timeToVia >= kNoPathMs) {
            continue;  // desde esta fila no se llega al intermedio: pasar por el no mejora nada
        }
        for (std::size_t toNode = 0; toNode < nodeCount; ++toNode) {
            const std::int32_t timeThroughVia = timeToVia + viaRow[toNode];
            if (timeThroughVia < fromRow[toNode]) {
                fromRow[toNode] = timeThroughVia;  // solo se escribe si mejora
            }
        }
    }
}

void checkSquareMatrix(const std::vector<std::int32_t>& cells, std::size_t nodeCount, const char* caller) {
    if (cells.size() != tableCellCount(nodeCount)) {
        throw TravelTimeTableError(std::string(caller) + ": la matriz tiene " + std::to_string(cells.size()) +
                                   " celdas y se esperaban " + std::to_string(tableCellCount(nodeCount)));
    }
}

// Escribe en `row` (nodeCount celdas) los tiempos desde `sourceNode` que dejo el Dijkstra del Router.
void storeDijkstraRow(Router& router, std::size_t sourceNode, std::int32_t* row, std::size_t nodeCount) {
    const std::vector<std::int64_t>& travelTimes = router.travelTimesFrom(static_cast<NodeIndex>(sourceNode));
    for (std::size_t toNode = 0; toNode < nodeCount; ++toNode) {
        const std::int64_t travelTimeMs = travelTimes[toNode];
        if (travelTimeMs == Router::kInfinityMs) {
            row[toNode] = kNoPathMs;
        } else if (travelTimeMs >= kNoPathMs) {
            throw TravelTimeTableError("el tiempo de " + std::to_string(sourceNode) + " a " + std::to_string(toNode) +
                                       " (" + std::to_string(travelTimeMs) + " ms) no cabe en una celda de la tabla");
        } else {
            row[toNode] = static_cast<std::int32_t>(travelTimeMs);
        }
    }
}

double elapsedMs(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

std::string describeBuild(std::size_t nodeCount, std::size_t memoryBytes, const std::string& who, double ms) {
    std::ostringstream text;
    text << "Tabla de tiempos: " << nodeCount << " nodos, " << std::fixed << std::setprecision(1)
         << static_cast<double>(memoryBytes) / (1024.0 * 1024.0) << " MiB, calculada por " << who << " en " << ms
         << " ms";
    return text.str();
}

}  // namespace

std::size_t tableCellCount(std::size_t nodeCount) {
    if (nodeCount != 0 && nodeCount > std::numeric_limits<std::size_t>::max() / nodeCount) {
        throw TravelTimeTableError("tabla de tiempos: " + std::to_string(nodeCount) + " nodos desbordan n*n");
    }
    if (nodeCount > kMaxTableNodes) {
        throw TravelTimeTableError("tabla de tiempos: " + std::to_string(nodeCount) + " nodos superan el tope de " +
                                   std::to_string(kMaxTableNodes));
    }
    return nodeCount * nodeCount;
}

std::size_t paddedNodeCount(std::size_t nodeCount) {
    return (nodeCount + kTileSize - 1) / kTileSize * kTileSize;
}

std::vector<DirectedStreet> streetsOf(const CityGraph& graph) {
    std::vector<DirectedStreet> streets;
    streets.reserve(graph.edgeCount());
    for (std::size_t fromNode = 0; fromNode < graph.nodeCount(); ++fromNode) {
        for (const Edge& edge : graph.edgesFrom(static_cast<NodeIndex>(fromNode))) {
            streets.push_back(DirectedStreet{fromNode, edge.to, edge.travelMs});
        }
    }
    return streets;
}

std::vector<std::int32_t> buildInitialCells(std::size_t nodeCount, const std::vector<DirectedStreet>& streets) {
    std::vector<std::int32_t> cells(tableCellCount(nodeCount), kNoPathMs);
    for (std::size_t node = 0; node < nodeCount; ++node) {
        cells[cellIndex(node, node, nodeCount)] = 0;
    }
    std::vector<std::int64_t> slowestStreetFromNode(nodeCount, 0);
    for (const DirectedStreet& street : streets) {
        if (street.fromNode >= nodeCount || street.toNode >= nodeCount) {
            throw TravelTimeTableError("calle con un nodo fuera de rango (" + std::to_string(street.fromNode) + " -> " +
                                       std::to_string(street.toNode) + ", hay " + std::to_string(nodeCount) + " nodos)");
        }
        if (street.travelTimeMs < 0 || street.travelTimeMs >= kNoPathMs) {
            throw TravelTimeTableError("tiempo de calle fuera de rango: " + std::to_string(street.travelTimeMs) +
                                       " ms (debe estar entre 0 y " + std::to_string(kNoPathMs - 1) + ")");
        }
        std::int32_t& cell = cells[cellIndex(street.fromNode, street.toNode, nodeCount)];
        cell = std::min(cell, static_cast<std::int32_t>(street.travelTimeMs));
        slowestStreetFromNode[street.fromNode] = std::max(slowestStreetFromNode[street.fromNode], street.travelTimeMs);
    }
    // Un camino simple usa a lo sumo una calle de salida por nodo: si la suma de las mas lentas cabe, ningun
    // tiempo finito llega a kNoPathMs y no se puede confundir con "no hay camino".
    std::int64_t longestPossiblePathMs = 0;
    for (const std::int64_t slowestMs : slowestStreetFromNode) {
        longestPossiblePathMs += slowestMs;
    }
    if (longestPossiblePathMs >= kNoPathMs) {
        throw TravelTimeTableError("un camino podria sumar " + std::to_string(longestPossiblePathMs) +
                                   " ms y no cabe en una celda de la tabla (limite " + std::to_string(kNoPathMs) + ")");
    }
    return cells;
}

void floydWarshallSingleThread(std::vector<std::int32_t>& cells, std::size_t nodeCount) {
    checkSquareMatrix(cells, nodeCount, "floydWarshallSingleThread");
    for (std::size_t viaNode = 0; viaNode < nodeCount; ++viaNode) {
        relaxRowsThroughVia(cells.data(), nodeCount, viaNode, 0, nodeCount);
    }
}

void floydWarshallMultiThread(std::vector<std::int32_t>& cells, std::size_t nodeCount, ThreadPool& pool) {
    checkSquareMatrix(cells, nodeCount, "floydWarshallMultiThread");
    std::int32_t* cellData = cells.data();
    for (std::size_t viaNode = 0; viaNode < nodeCount; ++viaNode) {
        pool.parallelFor(nodeCount, [cellData, nodeCount, viaNode](std::size_t firstRow, std::size_t endRow) {
            relaxRowsThroughVia(cellData, nodeCount, viaNode, firstRow, endRow);
        });
        // parallelFor ya volvio: todas las franjas terminaron esta pasada antes de empezar la siguiente.
    }
}

void dijkstraFromEveryNode(const CityGraph& graph, std::vector<std::int32_t>& cells) {
    const std::size_t nodeCount = graph.nodeCount();
    cells.resize(tableCellCount(nodeCount));
    Router router(graph);
    for (std::size_t sourceNode = 0; sourceNode < nodeCount; ++sourceNode) {
        storeDijkstraRow(router, sourceNode, cells.data() + cellIndex(sourceNode, 0, nodeCount), nodeCount);
    }
}

void dijkstraFromEveryNodeParallel(const CityGraph& graph, std::vector<std::int32_t>& cells, ThreadPool& pool) {
    const std::size_t nodeCount = graph.nodeCount();
    cells.resize(tableCellCount(nodeCount));
    std::int32_t* cellData = cells.data();
    const CityGraph* graphPointer = &graph;
    pool.parallelFor(nodeCount, [graphPointer, cellData, nodeCount](std::size_t firstSource, std::size_t endSource) {
        Router router(*graphPointer);  // un Router por franja (por hilo): tiene buffers internos
        for (std::size_t sourceNode = firstSource; sourceNode < endSource; ++sourceNode) {
            storeDijkstraRow(router, sourceNode, cellData + cellIndex(sourceNode, 0, nodeCount), nodeCount);
        }
    });
}

std::vector<std::int32_t> padCells(const std::vector<std::int32_t>& cells, std::size_t nodeCount) {
    checkSquareMatrix(cells, nodeCount, "padCells");
    const std::size_t paddedCount = paddedNodeCount(nodeCount);
    std::vector<std::int32_t> padded(tableCellCount(paddedCount), kNoPathMs);
    for (std::size_t node = 0; node < paddedCount; ++node) {
        padded[cellIndex(node, node, paddedCount)] = 0;
    }
    for (std::size_t fromNode = 0; fromNode < nodeCount; ++fromNode) {
        std::copy(cells.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, 0, nodeCount)),
                  cells.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, nodeCount, nodeCount)),
                  padded.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, 0, paddedCount)));
    }
    return padded;
}

std::vector<std::int32_t> unpadCells(const std::vector<std::int32_t>& paddedCells, std::size_t nodeCount) {
    const std::size_t paddedCount = paddedNodeCount(nodeCount);
    checkSquareMatrix(paddedCells, paddedCount, "unpadCells");
    std::vector<std::int32_t> cells(tableCellCount(nodeCount));
    for (std::size_t fromNode = 0; fromNode < nodeCount; ++fromNode) {
        std::copy(paddedCells.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, 0, paddedCount)),
                  paddedCells.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, nodeCount, paddedCount)),
                  cells.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, 0, nodeCount)));
    }
    return cells;
}

TravelTimeTable::TravelTimeTable(std::size_t nodeCount, std::vector<std::int32_t> cells)
    : nodeCount_(nodeCount), cells_(std::move(cells)) {
    checkSquareMatrix(cells_, nodeCount_, "TravelTimeTable");
}

TravelTimeBuild buildTravelTimeTable(const CityGraph& graph, GpuWorker* gpu, unsigned cpuThreads,
                                     std::size_t maxNodes, std::int64_t gpuTimeoutMs) {
    TravelTimeBuild build;
    const std::size_t nodeCount = graph.nodeCount();
    const std::size_t nodeLimit = std::min(maxNodes, kMaxTableNodes);
    if (nodeCount == 0) {
        build.line = "Tabla de tiempos: omitida (el grafo no tiene nodos)";
        return build;
    }
    if (nodeCount > nodeLimit) {
        build.line = "Tabla de tiempos: omitida, " + std::to_string(nodeCount) + " nodos superan el tope de " +
                     std::to_string(nodeLimit) + " (la tabla tiene n^2 celdas); las estimaciones usan el Router";
        return build;
    }

    const Clock::time_point startTime = Clock::now();
    std::string gpuNote;  // por que no se uso la GPU ("" = no se pidio o se uso)
    try {
        if (gpu != nullptr && gpu->healthy()) {
            try {
                std::vector<std::int32_t> initialCells = buildInitialCells(nodeCount, streetsOf(graph));
                // Un solo plazo absoluto en tiempo real: vale para salir de la cola y para esperar el resultado.
                const GpuWorker::Clock::time_point deadline = deadlineAfterMs(gpuTimeoutMs);
                // La tarea captura por VALOR (la matriz y n): si vence el plazo y se descarta, puede seguir viva
                // en la cola sin referirse a nada de esta funcion.
                std::future<std::vector<std::int32_t>> pending = gpu->submit(
                    [initialCells = std::move(initialCells), nodeCount]() {
                        std::vector<std::int32_t> resultCells;
                        floydWarshallTiledGpu(initialCells, nodeCount, resultCells, nullptr);
                        return resultCells;
                    },
                    deadline);
                if (!waitUntilDeadline(pending, deadline)) {  // bloqueado hasta el plazo, sin giro ni sleep
                    throw GpuTaskExpiredError("la GPU no entrego la tabla en " + std::to_string(gpuTimeoutMs) +
                                              " ms (plazo de arranque)");
                }
                std::vector<std::int32_t> cells = pending.get();
                build.table = std::make_shared<const TravelTimeTable>(nodeCount, std::move(cells));
                build.line = describeBuild(nodeCount, build.table->memoryBytes(),
                                           "la GPU (Floyd-Warshall con tiles de 32x32)", elapsedMs(startTime, Clock::now()));
                return build;
            } catch (const std::exception& caught) {
                gpuNote = caught.what();
            }
        } else if (gpu != nullptr) {
            gpuNote = "el hilo de la GPU no esta sano: " + gpu->unhealthyReason();
        }

        const unsigned threads = std::max(1u, cpuThreads);
        ThreadPool pool(threads);
        std::vector<std::int32_t> cells;
        dijkstraFromEveryNodeParallel(graph, cells, pool);
        build.table = std::make_shared<const TravelTimeTable>(nodeCount, std::move(cells));
        build.line = describeBuild(nodeCount, build.table->memoryBytes(),
                                   "la CPU (Dijkstra desde cada nodo, " + std::to_string(threads) + " hilos)",
                                   elapsedMs(startTime, Clock::now()));
        if (!gpuNote.empty()) {
            build.line += "; respaldo CPU, la GPU no se uso (plazo o error): " + gpuNote;
        }
    } catch (const std::exception& caught) {
        build.table.reset();
        build.line = std::string("Tabla de tiempos: omitida (") + caught.what() + "); las estimaciones usan el Router";
    }
    return build;
}

}  // namespace accel
