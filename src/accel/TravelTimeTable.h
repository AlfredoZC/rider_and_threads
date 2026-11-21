#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "CityGraph.h"
#include "accel/ThreadPool.h"

namespace accel {

// Aplicacion 03: tabla de tiempos de viaje entre TODOS los pares de intersecciones.
//
// La tabla se calcula una sola vez al arrancar (las calles no cambian durante la simulacion) y
// despues cada estimacion de un viaje es leer una celda. Cada celda es un int32 en milisegundos;
// una calle de un solo sentido solo cuenta en su sentido y entre dos calles paralelas gana la mas
// rapida. 0x3f3f3f3f ms (mas de 12 dias) significa "no hay camino": dos de ellas sumadas todavia
// caben en un int32, asi que ninguna suma de Floyd-Warshall desborda.
inline constexpr std::int32_t kUnreachableMs = 0x3f3f3f3f;

// Tabla inmutable: despues de construirla no cambia, asi que cualquier hilo puede leerla sin candado.
class TravelTimeTable {
public:
    // Lo que devuelve travelTimeMs cuando no hay camino (o el nodo no existe).
    static constexpr std::int64_t kNoRoute = -1;

    // `cells` tiene nodeCount * nodeCount valores, por filas: (desde, hasta) esta en desde * nodeCount + hasta.
    // Lanza std::invalid_argument si el tamano no coincide.
    TravelTimeTable(std::size_t nodeCount, std::vector<std::int32_t> cells);

    std::size_t nodeCount() const { return nodeCount_; }
    const std::vector<std::int32_t>& cells() const { return cells_; }

    // Tiempo minimo de `from` a `to` en ms, o kNoRoute si no hay camino o algun nodo esta fuera de rango.
    std::int64_t travelTimeMs(NodeIndex from, NodeIndex to) const;

private:
    std::size_t nodeCount_;
    std::vector<std::int32_t> cells_;
};

// Punto de publicacion de la tabla. Quien la calcula llama publish() una vez; los hilos de la
// simulacion llaman get() cuando la necesitan: mientras vale nullptr se sigue usando Dijkstra.
// Un mutex protege el puntero y el shared_ptr mantiene viva la tabla mientras alguien la lee.
class TravelTimeTableSlot {
public:
    void publish(std::shared_ptr<const TravelTimeTable> table);
    std::shared_ptr<const TravelTimeTable> get() const;

private:
    mutable std::mutex mutex_;
    std::shared_ptr<const TravelTimeTable> table_;
};

// ---- Entrada: la matriz inicial --------------------------------------------------------------

// Una calle en UN sentido, con su tiempo en ms sin saturar.
struct WeightedEdge {
    std::size_t fromNode = 0;
    std::size_t toNode = 0;
    std::int64_t travelMs = 0;
};

// Peso de una calle en la tabla: negativo -> 0, y como maximo kUnreachableMs - 1.
std::int32_t saturateEdgeMs(std::int64_t travelMs);

// 0 en la diagonal, el tiempo de cada calle en su sentido (la mas rapida si hay varias), kUnreachableMs en
// el resto. Las aristas con un nodo fuera de rango se ignoran.
std::vector<std::int32_t> buildInitialCells(std::size_t nodeCount, const std::vector<WeightedEdge>& edges);
std::vector<std::int32_t> buildInitialCells(const CityGraph& graph);

// ---- CPU -------------------------------------------------------------------------------------

// Floyd-Warshall en el lugar, 1 hilo: una pasada por cada nodo intermedio, en orden.
void floydWarshallSingleThread(std::vector<std::int32_t>& cells, std::size_t nodeCount);
// Igual, pero dentro de cada pivote las filas se reparten entre los hilos del pool.
void floydWarshallMultiThread(std::vector<std::int32_t>& cells, std::size_t nodeCount, ThreadPool& pool);

// Dijkstra desde cada nodo (la mejor CPU honesta para este problema): mismo resultado que
// Floyd-Warshall, pero en tiempo O(V * E log V) y sin tabla intermedia.
std::vector<std::int32_t> dijkstraAllPairsSingleThread(const CityGraph& graph);
std::vector<std::int32_t> dijkstraAllPairsMultiThread(const CityGraph& graph, ThreadPool& pool);

// Tabla calculada en CPU con Dijkstra desde cada nodo repartido entre los hilos del pool.
std::shared_ptr<const TravelTimeTable> buildTravelTimeTableCpu(const CityGraph& graph, ThreadPool& pool);

class GpuWorker;

// Calcula la tabla al arrancar la simulacion y la publica en `slot`: en la GPU (tarea para el
// GpuWorker) si `gpu` existe y esta sano; si no, o si la tarea falla o vence, con Dijkstra desde cada
// nodo en CPU. Bloquea al hilo que llama (sin sondeo) hasta tenerla; main la corre en un hilo propio.
// Escribe en stderr una linea "[gpu] tabla de tiempos lista: N nodos, backend X, T ms".
void computeAndPublishTravelTimeTable(const CityGraph& graph, GpuWorker* gpu, TravelTimeTableSlot& slot);

// ---- GPU -------------------------------------------------------------------------------------

// Tamano del tile del kernel por bloques; la tabla en la GPU se rellena hasta un multiplo.
inline constexpr int kApspTileSize = 32;

// Tiempos medidos con cudaEvent, en ms.
struct ApspGpuTiming {
    double h2dMs = 0.0;
    double kernelMs = 0.0;
    double d2hMs = 0.0;
};

// Rellena la tabla (nodeCount x nodeCount) hasta paddedCount x paddedCount: los nodos de relleno no
// tienen calles y tienen 0 en su diagonal, asi que no crean caminos. Requiere paddedCount >= nodeCount.
std::vector<std::int32_t> padCells(const std::vector<std::int32_t>& cells, std::size_t nodeCount,
                                   std::size_t paddedCount);
// Quita el relleno (inversa de padCells).
std::vector<std::int32_t> unpadCells(const std::vector<std::int32_t>& paddedCells, std::size_t paddedCount,
                                     std::size_t nodeCount);
// Menor multiplo de tileSize que es >= nodeCount; lanza std::length_error si el calculo no cabe.
std::size_t paddedNodeCount(std::size_t nodeCount, std::size_t tileSize);

// Memoria de la tabla en la GPU. Este encabezado no incluye CUDA: lo implementan
// accel/cuda/TravelTimeTable.cu (con -DUSE_CUDA=ON) y accel/cpu/TravelTimeTableCpu.cpp (sin CUDA,
// donde todo devuelve "compilado sin CUDA"). Cada funcion devuelve "" si salio bien o un mensaje;
// ninguna lanza ni aborta. Un error de ejecucion deja el error CUDA pendiente (sin limpiarlo) para que
// throwIfGpuError lo vea; solo el fallo de cudaMalloc se limpia (es recuperable).
class ApspDevice {
public:
    ApspDevice() = default;
    ~ApspDevice();
    ApspDevice(const ApspDevice&) = delete;
    ApspDevice& operator=(const ApspDevice&) = delete;

    // Reserva paddedCount^2 enteros y los dos cudaEvent (libera antes lo que hubiera).
    std::string allocate(std::size_t paddedCount);
    void release();

    // Copia host -> GPU de paddedCount^2 enteros; `elapsedMs` (opcional) recibe el tiempo con cudaEvent.
    std::string upload(const std::int32_t* host, double* elapsedMs);
    // Las tres fases de Floyd-Warshall por tiles, por cada tile pivote, con tiles de tileSize (16 o 32).
    // paddedCount debe ser multiplo de tileSize. `elapsedMs` (opcional): tiempo con cudaEvent.
    std::string runTiled(int tileSize, double* elapsedMs);
    // Copia GPU -> host (espera a los kernels pendientes).
    std::string download(std::int32_t* host, double* elapsedMs);

private:
    void* devicePointer_ = nullptr;
    std::size_t paddedCount_ = 0;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

// Floyd-Warshall completo en la GPU: rellena, copia, lanza los kernels, trae el resultado y quita el
// relleno. `result` queda del tamano nodeCount^2. Devuelve "" o el mensaje de error; `timing` opcional.
std::string floydWarshallGpu(const std::vector<std::int32_t>& cells, std::size_t nodeCount,
                             std::vector<std::int32_t>& result, ApspGpuTiming* timing);

// Cuerpo de la tarea que se envia al GpuWorker (se ejecuta en su hilo): construye la tabla en la GPU.
// Lanza GpuFatalError si queda un error CUDA pegajoso (el worker pasa a "no saludable") y
// std::runtime_error con el mensaje en cualquier otro fallo; el llamador usa entonces la CPU.
// Captura el grafo por shared_ptr / valor: ver las reglas de GpuWorker::submit.
std::shared_ptr<const TravelTimeTable> buildTravelTimeTableGpuTask(const std::vector<std::int32_t>& initialCells,
                                                                    std::size_t nodeCount);

}  // namespace accel
