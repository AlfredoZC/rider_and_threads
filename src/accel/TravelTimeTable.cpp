#include "accel/TravelTimeTable.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

#include "accel/GpuContext.h"
#include "accel/GpuError.h"
#include "accel/GpuWorker.h"

namespace accel {

namespace {

// Celda (fromNode, toNode) de una tabla de nodeCount columnas; el producto se calcula en size_t.
std::size_t cellIndex(std::size_t fromNode, std::size_t toNode, std::size_t nodeCount) {
    return fromNode * nodeCount + toNode;
}

// Tamano de una tabla cuadrada, o std::length_error si no cabe en size_t.
std::size_t squareCellCount(std::size_t nodeCount) {
    if (nodeCount != 0 && nodeCount > std::numeric_limits<std::size_t>::max() / nodeCount) {
        throw std::length_error("la tabla de tiempos no cabe en memoria");
    }
    return nodeCount * nodeCount;
}

// Permite pasar por viaNode para las filas [firstRow, endRow). Solo escribe si mejora.
void relaxRowsThroughVia(std::vector<std::int32_t>& cells, std::size_t nodeCount, std::size_t viaNode,
                         std::size_t firstRow, std::size_t endRow) {
    for (std::size_t fromNode = firstRow; fromNode < endRow; ++fromNode) {
        const std::int32_t timeToVia = cells[cellIndex(fromNode, viaNode, nodeCount)];
        if (timeToVia >= kUnreachableMs) {
            continue;  // desde fromNode no se llega a viaNode: pasar por ahi no mejora nada
        }
        for (std::size_t toNode = 0; toNode < nodeCount; ++toNode) {
            const std::int32_t timeThroughVia = timeToVia + cells[cellIndex(viaNode, toNode, nodeCount)];
            std::int32_t& bestTime = cells[cellIndex(fromNode, toNode, nodeCount)];
            if (timeThroughVia < bestTime) {
                bestTime = timeThroughVia;
            }
        }
    }
}

// Dijkstra desde sourceNode; escribe la fila de la tabla (saturada a kUnreachableMs).
void dijkstraRow(const CityGraph& graph, NodeIndex sourceNode, std::vector<std::int64_t>& bestTimes,
                 std::int32_t* rowOut) {
    const std::size_t nodeCount = graph.nodeCount();
    std::fill(bestTimes.begin(), bestTimes.end(), std::int64_t{kUnreachableMs});
    using Entry = std::pair<std::int64_t, NodeIndex>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
    bestTimes[sourceNode] = 0;
    queue.emplace(0, sourceNode);
    while (!queue.empty()) {
        const Entry top = queue.top();
        queue.pop();
        if (top.first > bestTimes[top.second]) {
            continue;
        }
        for (const Edge& edge : graph.edgesFrom(top.second)) {
            const std::int64_t candidate = top.first + saturateEdgeMs(edge.travelMs);
            if (candidate < bestTimes[edge.to]) {  // bestTimes arranca en kUnreachableMs: no se pasa de ahi
                bestTimes[edge.to] = candidate;
                queue.emplace(candidate, edge.to);
            }
        }
    }
    for (std::size_t targetNode = 0; targetNode < nodeCount; ++targetNode) {
        rowOut[targetNode] = static_cast<std::int32_t>(bestTimes[targetNode]);
    }
}

void dijkstraRows(const CityGraph& graph, std::size_t firstRow, std::size_t endRow, std::vector<std::int32_t>& cells) {
    std::vector<std::int64_t> bestTimes(graph.nodeCount());
    for (std::size_t sourceNode = firstRow; sourceNode < endRow; ++sourceNode) {
        dijkstraRow(graph, static_cast<NodeIndex>(sourceNode), bestTimes,
                    cells.data() + cellIndex(sourceNode, 0, graph.nodeCount()));
    }
}

}  // namespace

TravelTimeTable::TravelTimeTable(std::size_t nodeCount, std::vector<std::int32_t> cells)
    : nodeCount_(nodeCount), cells_(std::move(cells)) {
    if (cells_.size() != squareCellCount(nodeCount_)) {
        throw std::invalid_argument("TravelTimeTable: el numero de celdas no es nodeCount^2");
    }
}

std::int64_t TravelTimeTable::travelTimeMs(NodeIndex from, NodeIndex to) const {
    if (from >= nodeCount_ || to >= nodeCount_) {
        return kNoRoute;
    }
    const std::int32_t cell = cells_[cellIndex(from, to, nodeCount_)];
    return cell >= kUnreachableMs ? kNoRoute : cell;
}

void TravelTimeTableSlot::publish(std::shared_ptr<const TravelTimeTable> table) {
    std::lock_guard<std::mutex> lock(mutex_);
    table_ = std::move(table);
}

std::shared_ptr<const TravelTimeTable> TravelTimeTableSlot::get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return table_;
}

std::int32_t saturateEdgeMs(std::int64_t travelMs) {
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(travelMs, 0, kUnreachableMs - 1));
}

std::vector<std::int32_t> buildInitialCells(std::size_t nodeCount, const std::vector<WeightedEdge>& edges) {
    std::vector<std::int32_t> cells(squareCellCount(nodeCount), kUnreachableMs);
    for (std::size_t node = 0; node < nodeCount; ++node) {
        cells[cellIndex(node, node, nodeCount)] = 0;
    }
    for (const WeightedEdge& edge : edges) {
        if (edge.fromNode >= nodeCount || edge.toNode >= nodeCount) {
            continue;
        }
        std::int32_t& cell = cells[cellIndex(edge.fromNode, edge.toNode, nodeCount)];
        cell = std::min(cell, saturateEdgeMs(edge.travelMs));  // calles paralelas: gana la mas rapida
    }
    return cells;
}

std::vector<std::int32_t> buildInitialCells(const CityGraph& graph) {
    std::vector<WeightedEdge> edges;
    edges.reserve(graph.edgeCount());
    for (std::size_t fromNode = 0; fromNode < graph.nodeCount(); ++fromNode) {
        for (const Edge& edge : graph.edgesFrom(static_cast<NodeIndex>(fromNode))) {
            edges.push_back({fromNode, edge.to, edge.travelMs});
        }
    }
    return buildInitialCells(graph.nodeCount(), edges);
}

void floydWarshallSingleThread(std::vector<std::int32_t>& cells, std::size_t nodeCount) {
    for (std::size_t viaNode = 0; viaNode < nodeCount; ++viaNode) {
        relaxRowsThroughVia(cells, nodeCount, viaNode, 0, nodeCount);
    }
}

void floydWarshallMultiThread(std::vector<std::int32_t>& cells, std::size_t nodeCount, ThreadPool& pool) {
    for (std::size_t viaNode = 0; viaNode < nodeCount; ++viaNode) {
        pool.parallelFor(nodeCount, [&cells, nodeCount, viaNode](std::size_t firstRow, std::size_t endRow) {
            relaxRowsThroughVia(cells, nodeCount, viaNode, firstRow, endRow);
        });
        // parallelFor ya volvio: todas las filas terminaron este pivote antes de empezar el siguiente.
    }
}

std::vector<std::int32_t> dijkstraAllPairsSingleThread(const CityGraph& graph) {
    std::vector<std::int32_t> cells(squareCellCount(graph.nodeCount()), kUnreachableMs);
    dijkstraRows(graph, 0, graph.nodeCount(), cells);
    return cells;
}

std::vector<std::int32_t> dijkstraAllPairsMultiThread(const CityGraph& graph, ThreadPool& pool) {
    std::vector<std::int32_t> cells(squareCellCount(graph.nodeCount()), kUnreachableMs);
    // Cada hilo escribe filas distintas: no comparten ninguna celda.
    pool.parallelFor(graph.nodeCount(), [&graph, &cells](std::size_t firstRow, std::size_t endRow) {
        dijkstraRows(graph, firstRow, endRow, cells);
    });
    return cells;
}

std::shared_ptr<const TravelTimeTable> buildTravelTimeTableCpu(const CityGraph& graph, ThreadPool& pool) {
    return std::make_shared<const TravelTimeTable>(graph.nodeCount(), dijkstraAllPairsMultiThread(graph, pool));
}

std::size_t paddedNodeCount(std::size_t nodeCount, std::size_t tileSize) {
    if (tileSize == 0) {
        throw std::invalid_argument("paddedNodeCount: tileSize 0");
    }
    const std::size_t remainder = nodeCount % tileSize;
    if (remainder == 0) {
        return nodeCount;
    }
    if (nodeCount > std::numeric_limits<std::size_t>::max() - (tileSize - remainder)) {
        throw std::length_error("paddedNodeCount: no cabe en size_t");
    }
    return nodeCount + (tileSize - remainder);
}

std::vector<std::int32_t> padCells(const std::vector<std::int32_t>& cells, std::size_t nodeCount,
                                   std::size_t paddedCount) {
    if (paddedCount < nodeCount || cells.size() != squareCellCount(nodeCount)) {
        throw std::invalid_argument("padCells: tamanos incoherentes");
    }
    std::vector<std::int32_t> padded(squareCellCount(paddedCount), kUnreachableMs);
    for (std::size_t fromNode = 0; fromNode < paddedCount; ++fromNode) {
        if (fromNode < nodeCount) {
            std::copy_n(cells.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, 0, nodeCount)), nodeCount,
                        padded.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, 0, paddedCount)));
        }
        padded[cellIndex(fromNode, fromNode, paddedCount)] = 0;
    }
    return padded;
}

std::vector<std::int32_t> unpadCells(const std::vector<std::int32_t>& paddedCells, std::size_t paddedCount,
                                     std::size_t nodeCount) {
    if (paddedCount < nodeCount || paddedCells.size() != squareCellCount(paddedCount)) {
        throw std::invalid_argument("unpadCells: tamanos incoherentes");
    }
    std::vector<std::int32_t> cells(squareCellCount(nodeCount));
    for (std::size_t fromNode = 0; fromNode < nodeCount; ++fromNode) {
        std::copy_n(paddedCells.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, 0, paddedCount)), nodeCount,
                    cells.begin() + static_cast<std::ptrdiff_t>(cellIndex(fromNode, 0, nodeCount)));
    }
    return cells;
}

std::string floydWarshallGpu(const std::vector<std::int32_t>& cells, std::size_t nodeCount,
                             std::vector<std::int32_t>& result, ApspGpuTiming* timing) {
    ApspGpuTiming measured;
    if (nodeCount == 0) {
        result.clear();
        if (timing != nullptr) *timing = measured;
        return "";
    }
    std::size_t paddedCount = 0;
    std::vector<std::int32_t> padded;
    try {
        paddedCount = paddedNodeCount(nodeCount, static_cast<std::size_t>(kApspTileSize));
        padded = padCells(cells, nodeCount, paddedCount);
    } catch (const std::exception& error) {
        return std::string("floydWarshallGpu: ") + error.what();
    }
    ApspDevice device;
    std::string errorMessage = device.allocate(paddedCount);
    if (errorMessage.empty()) errorMessage = device.upload(padded.data(), &measured.h2dMs);
    if (errorMessage.empty()) errorMessage = device.runTiled(kApspTileSize, &measured.kernelMs);
    if (errorMessage.empty()) errorMessage = device.download(padded.data(), &measured.d2hMs);
    if (!errorMessage.empty()) {
        return errorMessage;
    }
    result = unpadCells(padded, paddedCount, nodeCount);
    if (timing != nullptr) *timing = measured;
    return "";
}

std::shared_ptr<const TravelTimeTable> buildTravelTimeTableGpuTask(const std::vector<std::int32_t>& initialCells,
                                                                    std::size_t nodeCount) {
    std::vector<std::int32_t> result;
    const std::string errorMessage = floydWarshallGpu(initialCells, nodeCount, result, nullptr);
    if (!errorMessage.empty()) {
        throwIfGpuError("tabla de tiempos (Floyd-Warshall)");  // pegajoso -> GpuFatalError
        throw std::runtime_error(errorMessage);
    }
    return std::make_shared<const TravelTimeTable>(nodeCount, std::move(result));
}


void computeAndPublishTravelTimeTable(const CityGraph& graph, GpuWorker* gpu, TravelTimeTableSlot& slot) {
    try {
        const auto startTime = std::chrono::steady_clock::now();
        std::shared_ptr<const TravelTimeTable> table;
        const char* backendName = "cpu";
        if (gpu != nullptr && gpu->healthy()) {
            try {
                // La tarea captura la matriz inicial por shared_ptr: puede seguir en cola despues de salir de aqui.
                const auto initialCells = std::make_shared<const std::vector<std::int32_t>>(buildInitialCells(graph));
                const std::size_t nodeCount = graph.nodeCount();
                std::future<std::shared_ptr<const TravelTimeTable>> future = gpu->submit(
                    [initialCells, nodeCount] { return buildTravelTimeTableGpuTask(*initialCells, nodeCount); });
                table = future.get();
                backendName = "gpu";
            } catch (const std::exception& error) {
                std::cerr << "[gpu] tabla de tiempos: la GPU fallo (" << error.what() << "), se usa la CPU" << std::endl;
            }
        }
        if (!table) {
            ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
            table = buildTravelTimeTableCpu(graph, pool);
        }
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
        slot.publish(table);
        std::cerr << "[gpu] tabla de tiempos lista: " << graph.nodeCount() << " nodos, backend " << backendName
                  << ", " << static_cast<long long>(elapsedMs + 0.5) << " ms" << std::endl;
    } catch (const std::exception& error) {
        // Sin tabla la simulacion sigue con Dijkstra, como siempre.
        std::cerr << "[gpu] tabla de tiempos no disponible: " << error.what() << std::endl;
    }
}

}  // namespace accel
