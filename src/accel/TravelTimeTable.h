#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "CityGraph.h"
#include "accel/ThreadPool.h"

namespace accel {

class GpuWorker;

// Tabla de tiempos minimos entre TODOS los pares de intersecciones
// (n x n celdas de int32, por filas: la celda (desde, hasta) esta en desde * n + hasta). Se calcula una
// vez al arrancar y cada estimacion de entrega lee dos celdas en vez de correr dos Dijkstra.
//
// Este encabezado no incluye nada de CUDA. Lo implementan:
//   accel/TravelTimeTable.cpp          la tabla, Floyd-Warshall de 1 y N hilos, Dijkstra desde cada nodo,
//                                      el relleno a multiplos de 32 y el armado al arrancar
//   accel/cuda/TravelTimeTable.cu      los kernels (con tiles y sin tiles) y sus envoltorios (-DUSE_CUDA=ON)
//   accel/cpu/TravelTimeTableCpu.cpp   respaldo sin CUDA: las funciones de GPU lanzan GpuUnavailableError

// "No hay camino" dentro de la tabla: 1.000.000.000 ms (11,5 dias, mas que cualquier viaje real). Dos
// "no hay camino" sumados dan 2.000.000.000, que todavia cabe en un int32 (max 2.147.483.647): la suma
// D[desde][via] + D[via][hasta] nunca desborda. Hacia afuera (TravelTimeTable::travelTimeMs) es -1.
constexpr std::int32_t kNoPathMs = 1'000'000'000;

// Lado del tile de memoria compartida de los kernels (32 x 32 hilos = 1.024 hilos por bloque).
constexpr std::size_t kTileSize = 32;

// Tope de intersecciones con tabla: 8.192^2 celdas x 4 B = 256 MiB, en RAM y otro tanto en la GPU. La ciudad
// tiene 3.741 (56 MB), asi que sobra margen. Por encima de esto el programa sigue con el Router, como antes.
constexpr std::size_t kMaxTableNodes = 8192;

// Plazo REAL (steady_clock) de arranque para que la GPU entregue la tabla: 60 s, muy por encima de los ~0,2 s
// medidos con 3.741 nodos. Si vence, la CPU la arma y la tarea de la GPU se descarta.
constexpr std::int64_t kStartupGpuTimeoutMs = 60000;

// Error de tamano o de datos que la tabla no puede representar (n demasiado grande, n*n desbordado, un
// tiempo de calle fuera de rango, un camino que podria alcanzar el "infinito"...). Mensaje claro, sin abortar.
class TravelTimeTableError : public std::runtime_error {
public:
    explicit TravelTimeTableError(const std::string& what) : std::runtime_error(what) {}
};

// Una calle ya dirigida: las de doble sentido se cargan como dos.
struct DirectedStreet {
    std::size_t fromNode = 0;
    std::size_t toNode = 0;
    std::int64_t travelTimeMs = 0;
};

// n * n verificado (size_t sin desborde y n <= kMaxTableNodes). Lanza TravelTimeTableError.
std::size_t tableCellCount(std::size_t nodeCount);

// n redondeado hacia arriba al multiplo de kTileSize (0 queda en 0). Requiere n <= kMaxTableNodes.
std::size_t paddedNodeCount(std::size_t nodeCount);

// Posicion de la celda (desde, hasta) en una matriz de `nodeCount` columnas.
inline std::size_t cellIndex(std::size_t fromNode, std::size_t toNode, std::size_t nodeCount) {
    return fromNode * nodeCount + toNode;
}

// Las calles del grafo, una por arco dirigido (el orden de las listas de adyacencia).
std::vector<DirectedStreet> streetsOf(const CityGraph& graph);

// Matriz inicial: 0 en la diagonal, el tiempo de cada calle en SU sentido (si hay varias entre los mismos
// nodos gana la mas rapida) y kNoPathMs en el resto. Lanza TravelTimeTableError si un nodo esta fuera de
// rango, un tiempo es negativo o >= kNoPathMs, o si algun camino simple podria sumar >= kNoPathMs (se
// acota con la suma, por nodo, de su calle de salida mas lenta): asi ningun tiempo finito se confunde con "no hay camino".
std::vector<std::int32_t> buildInitialCells(std::size_t nodeCount, const std::vector<DirectedStreet>& streets);

// Floyd-Warshall en el lugar sobre una matriz de nodeCount x nodeCount (1 hilo): una pasada por cada nodo
// intermedio, en orden. La celda solo se escribe si mejora (la fila y la columna del intermedio no se tocan).
void floydWarshallSingleThread(std::vector<std::int32_t>& cells, std::size_t nodeCount);

// Igual con N hilos: el bucle de nodos intermedios sigue en orden y, dentro de cada uno, las filas se
// reparten en franjas con `pool.parallelFor` (el fin de parallelFor es la barrera entre pasadas).
void floydWarshallMultiThread(std::vector<std::int32_t>& cells, std::size_t nodeCount, ThreadPool& pool);

// Dijkstra desde cada nodo (reutiliza Router::travelTimesFrom): llena `cells` (se redimensiona a n*n) con la
// tabla de tiempos. 1 hilo. Lanza TravelTimeTableError si un tiempo finito no cabe (>= kNoPathMs).
void dijkstraFromEveryNode(const CityGraph& graph, std::vector<std::int32_t>& cells);

// Igual con N hilos: las filas de origen se reparten en franjas con `pool.parallelFor` y cada franja usa
// SU PROPIO Router (el Router tiene buffers internos y no se comparte entre hilos). Cada hilo escribe solo sus filas.
void dijkstraFromEveryNodeParallel(const CityGraph& graph, std::vector<std::int32_t>& cells, ThreadPool& pool);

// Relleno: copia la matriz de n x n en una de paddedNodeCount(n) con kNoPathMs en los nodos de relleno y 0 en
// su diagonal (no crean caminos y mantienen D[k][k] = 0). unpadCells hace el camino inverso.
std::vector<std::int32_t> padCells(const std::vector<std::int32_t>& cells, std::size_t nodeCount);
std::vector<std::int32_t> unpadCells(const std::vector<std::int32_t>& paddedCells, std::size_t nodeCount);

// La tabla terminada. Inmutable: se arma una vez y despues solo se lee (muchos hilos a la vez, sin candados).
class TravelTimeTable {
public:
    TravelTimeTable() = default;
    // `cells` debe tener nodeCount * nodeCount valores (kNoPathMs = sin camino). Lanza TravelTimeTableError si no.
    TravelTimeTable(std::size_t nodeCount, std::vector<std::int32_t> cells);

    std::size_t nodeCount() const { return nodeCount_; }
    const std::vector<std::int32_t>& cells() const { return cells_; }
    std::size_t memoryBytes() const { return cells_.size() * sizeof(std::int32_t); }

    // Tiempo minimo en ms de `fromNode` a `toNode`; -1 si no hay camino o si algun indice esta fuera de
    // rango (igual que un Router que no encuentra ruta).
    std::int64_t travelTimeMs(std::size_t fromNode, std::size_t toNode) const {
        if (fromNode >= nodeCount_ || toNode >= nodeCount_) {
            return -1;
        }
        const std::int32_t cell = cells_[cellIndex(fromNode, toNode, nodeCount_)];
        return cell >= kNoPathMs ? -1 : cell;
    }

private:
    std::size_t nodeCount_ = 0;
    std::vector<std::int32_t> cells_;
};

// ---- GPU (accel/cuda/TravelTimeTable.cu; sin CUDA lanzan GpuUnavailableError) -----------------------------
// Tiempos medidos con cudaEvent, en ms.
struct GpuFloydTiming {
    double hostToDeviceMs = 0.0;
    double kernelMs = 0.0;  // todas las pasadas del algoritmo
    double deviceToHostMs = 0.0;
};

// Floyd-Warshall en la GPU con tiles de 32 x 32 en memoria compartida, en 3 fases por tile pivote.
// `initialCells` = matriz inicial de n x n (buildInitialCells); `resultCells` recibe la tabla de n x n.
// Rellena a multiplo de 32 en el host, copia una vez, lanza las fases, trae la tabla y quita el relleno.
// Errores: sin memoria o configuracion invalida -> std::runtime_error (la GPU sigue sana); un error CUDA
// pegajoso -> GpuFatalError (quien llama desde el GpuWorker deja que viaje en el future).
void floydWarshallTiledGpu(const std::vector<std::int32_t>& initialCells, std::size_t nodeCount,
                           std::vector<std::int32_t>& resultCells, GpuFloydTiming* timing);

// Floyd-Warshall en la GPU SIN tiles: un lanzamiento por cada nodo intermedio, un hilo por celda, todo en
// memoria global. Sirve para medir cuanto aporta el tiling. Mismos errores que la version con tiles.
void floydWarshallNaiveGpu(const std::vector<std::int32_t>& initialCells, std::size_t nodeCount,
                           std::vector<std::int32_t>& resultCells, GpuFloydTiming* timing);

// ---- armado al arrancar -----------------------------------------------------------------------------------
struct TravelTimeBuild {
    std::shared_ptr<const TravelTimeTable> table;  // nullptr = sin tabla: el programa sigue con el Router
    std::string line;                              // una linea para stderr: quien la calculo, cuanto tardo
};

// Calcula la tabla UNA vez, antes de lanzar los hilos de la simulacion. Con `gpu` sano manda Floyd-Warshall con
// tiles al GpuWorker (submit + esperar el future hasta un plazo REAL de `gpuTimeoutMs`; la tarea captura por valor).
// Si no hay GPU, falla, no tiene memoria o no entrega a tiempo, la CPU la arma con Dijkstra desde cada nodo en
// `cpuThreads` hilos (la mejor CPU) y la tarea de la GPU se descarta. Si el grafo tiene mas de `maxNodes` nodos o
// los tiempos no caben, devuelve table == nullptr con el motivo en `line`. Nunca lanza.
TravelTimeBuild buildTravelTimeTable(const CityGraph& graph, GpuWorker* gpu, unsigned cpuThreads,
                                     std::size_t maxNodes = kMaxTableNodes,
                                     std::int64_t gpuTimeoutMs = kStartupGpuTimeoutMs);

}  // namespace accel
