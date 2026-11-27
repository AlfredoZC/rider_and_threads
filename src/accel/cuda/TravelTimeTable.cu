// Floyd-Warshall en la GPU: kernels y envoltorios, sin hilos ni estado de la aplicacion.
//
// Floyd-Warshall es una "multiplicacion de matrices" (min, +): D[i][j] = min(D[i][j], D[i][k] + D[k][j]) para
// cada nodo intermedio k, en orden. Hay dos versiones:
//   naive  un lanzamiento por cada k, un hilo por celda, todo en memoria global (3 lecturas globales por relajacion)
//   tiled  tiles de 32 x 32 en memoria compartida, en 3 fases por tile pivote (ver los kernels abajo): las lecturas
//          globales bajan 32 veces y los lanzamientos pasan de n a 3 * n / 32
//
// Cada llamada CUDA y cada lanzamiento se revisan. Un error recuperable (sin memoria, argumento o configuracion
// invalidos) lanza std::runtime_error y limpia el error; cualquier otro se trata como pegajoso y lanza GpuFatalError.
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "accel/GpuError.h"
#include "accel/TravelTimeTable.h"

namespace accel {

namespace {

constexpr int kTile = 32;
static_assert(kTile == static_cast<int>(kTileSize), "el kernel y TravelTimeTable.h deben usar el mismo tile");
static_assert(kTile * kTile <= 1024, "un bloque CUDA tiene como maximo 1024 hilos");

constexpr int kNaiveBlockSide = 16;  // 16 x 16 = 256 hilos: el kernel ingenuo no usa memoria compartida

bool isRecoverable(cudaError_t status) {
    return status == cudaErrorMemoryAllocation || status == cudaErrorInvalidValue ||
           status == cudaErrorInvalidConfiguration;
}

// Convierte un error de CUDA en excepcion. `operation` dice que se estaba haciendo.
void check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
    if (isRecoverable(status)) {
        cudaGetLastError();  // se borra: si no, la proxima tarea lo veria como suyo
        throw std::runtime_error(message);
    }
    throw GpuFatalError("CUDA: " + message);
}

// ---- Naive: un hilo por celda (fromNode, toNode) de la pasada del nodo intermedio viaNode -----------------------
__global__ void floydWarshallNaiveStep(std::int32_t* travelTimeMs, int nodeCount, int viaNode) {
    const int toNode = blockIdx.x * blockDim.x + threadIdx.x;    // x = columna: hilos vecinos, columnas vecinas (coalescido)
    const int fromNode = blockIdx.y * blockDim.y + threadIdx.y;  // y = fila
    if (fromNode >= nodeCount || toNode >= nodeCount) {
        return;  // hilos sobrantes del ultimo bloque
    }
    const std::size_t nodes = static_cast<std::size_t>(nodeCount);
    const std::int32_t timeThroughVia = travelTimeMs[static_cast<std::size_t>(fromNode) * nodes + viaNode]  // D[i][k]: global
                                        + travelTimeMs[static_cast<std::size_t>(viaNode) * nodes + toNode];  // D[k][j]: global
    const std::size_t cell = static_cast<std::size_t>(fromNode) * nodes + toNode;
    if (timeThroughVia < travelTimeMs[cell]) {  // D[i][j]: global; solo se escribe si mejora (fila y columna k no se tocan)
        travelTimeMs[cell] = timeThroughVia;
    }
}

// ---- Tiled. nodeCount aqui es el numero de nodos YA RELLENO (multiplo de kTile) -------------------------------------
// Fase 1: el tile pivote (pivotTile, pivotTile), solo, como un Floyd-Warshall chico. Un bloque de 32 x 32 hilos,
// un hilo por celda; el tile vive en memoria compartida (4 KB) y se escribe una sola vez al final.
__global__ void floydWarshallPhase1(std::int32_t* travelTimeMs, int nodeCount, int pivotTile) {
    __shared__ std::int32_t pivotBlock[kTile][kTile];
    const int row = threadIdx.y;
    const int column = threadIdx.x;
    const int pivotBase = pivotTile * kTile;
    const std::size_t cell = static_cast<std::size_t>(pivotBase + row) * nodeCount + (pivotBase + column);

    pivotBlock[row][column] = travelTimeMs[cell];  // cada hilo carga su celda
    __syncthreads();                               // el tile completo antes de usarlo

    for (int via = 0; via < kTile; ++via) {  // los nodos intermedios de este tile, en orden
        const std::int32_t timeThroughVia = pivotBlock[row][via] + pivotBlock[via][column];
        if (timeThroughVia < pivotBlock[row][column]) {
            pivotBlock[row][column] = timeThroughVia;  // nunca escribe la fila ni la columna `via`
        }
        __syncthreads();  // la vuelta via + 1 lee lo que escribio la vuelta via
    }
    travelTimeMs[cell] = pivotBlock[row][column];
}

// Fase 2: los tiles de la FILA del pivote (blockIdx.y == 0) y de la COLUMNA del pivote (blockIdx.y == 1);
// blockIdx.x recorre los tiles. Usan el tile pivote ya terminado (8 KB de memoria compartida: pivote + el propio).
__global__ void floydWarshallPhase2(std::int32_t* travelTimeMs, int nodeCount, int pivotTile) {
    const int otherTile = blockIdx.x;
    if (otherTile == pivotTile) {
        return;  // el pivote lo hizo la fase 1 (sale el bloque entero, antes de cualquier __syncthreads)
    }
    const bool isPivotRow = (blockIdx.y == 0);
    __shared__ std::int32_t pivotBlock[kTile][kTile];  // tile pivote, ya terminado
    __shared__ std::int32_t ownBlock[kTile][kTile];    // el tile que este bloque actualiza
    const int row = threadIdx.y;
    const int column = threadIdx.x;
    const int pivotBase = pivotTile * kTile;
    const int otherBase = otherTile * kTile;
    const int fromNode = isPivotRow ? pivotBase + row : otherBase + row;
    const int toNode = isPivotRow ? otherBase + column : pivotBase + column;
    const std::size_t cell = static_cast<std::size_t>(fromNode) * nodeCount + toNode;

    pivotBlock[row][column] = travelTimeMs[static_cast<std::size_t>(pivotBase + row) * nodeCount + (pivotBase + column)];
    ownBlock[row][column] = travelTimeMs[cell];
    __syncthreads();

    for (int via = 0; via < kTile; ++via) {
        // Fila del pivote:    `i` esta en el pivote -> D[i][k] sale del pivote, D[k][j] de este tile.
        // Columna del pivote: `j` esta en el pivote -> D[i][k] sale de este tile, D[k][j] del pivote.
        const std::int32_t timeThroughVia = isPivotRow ? pivotBlock[row][via] + ownBlock[via][column]
                                                       : ownBlock[row][via] + pivotBlock[via][column];
        if (timeThroughVia < ownBlock[row][column]) {
            ownBlock[row][column] = timeThroughVia;
        }
        __syncthreads();
    }
    travelTimeMs[cell] = ownBlock[row][column];
}

// Fase 3: todos los demas tiles (casi todo el trabajo). Es el matmul con tiles pero con (min, +): el tile
// (tileRow, tileColumn) usa el tile (tileRow, pivote) para D[i][k] y el tile (pivote, tileColumn) para D[k][j],
// los dos ya terminados por la fase 2 y que nadie modifica en esta fase. Cada valor cargado a memoria compartida
// (8 KB por bloque) se usa 32 veces.
__global__ void floydWarshallPhase3(std::int32_t* travelTimeMs, int nodeCount, int pivotTile) {
    const int tileRow = blockIdx.y;
    const int tileColumn = blockIdx.x;
    if (tileRow == pivotTile || tileColumn == pivotTile) {
        return;  // los hicieron las fases 1 y 2 (decision de bloque entero)
    }
    __shared__ std::int32_t pivotColumnTile[kTile][kTile];  // tile (tileRow, pivote):    los D[i][k]
    __shared__ std::int32_t pivotRowTile[kTile][kTile];     // tile (pivote, tileColumn): los D[k][j]
    const int row = threadIdx.y;
    const int column = threadIdx.x;
    const int fromNode = tileRow * kTile + row;
    const int toNode = tileColumn * kTile + column;
    const int pivotBase = pivotTile * kTile;

    // Cada hilo carga UNA celda de cada tile; entre todos cargan los dos tiles completos.
    pivotColumnTile[row][column] = travelTimeMs[static_cast<std::size_t>(fromNode) * nodeCount + (pivotBase + column)];
    pivotRowTile[row][column] = travelTimeMs[static_cast<std::size_t>(pivotBase + row) * nodeCount + toNode];
    __syncthreads();  // nadie lee un tile antes de que este completo

    const std::size_t cell = static_cast<std::size_t>(fromNode) * nodeCount + toNode;
    std::int32_t bestTime = travelTimeMs[cell];  // la celda propia vive en un registro
    for (int via = 0; via < kTile; ++via) {      // 32 relajaciones, todas desde memoria compartida
        bestTime = min(bestTime, pivotColumnTile[row][via] + pivotRowTile[via][column]);
    }
    travelTimeMs[cell] = bestTime;  // una sola escritura a memoria global
}

// ---- RAII ---------------------------------------------------------------------------------------------------------
class DeviceCells {
public:
    explicit DeviceCells(std::size_t bytes) { check(cudaMalloc(&pointer_, bytes), "cudaMalloc de la tabla"); }
    ~DeviceCells() { cudaFree(pointer_); }
    DeviceCells(const DeviceCells&) = delete;
    DeviceCells& operator=(const DeviceCells&) = delete;
    std::int32_t* data() { return static_cast<std::int32_t*>(pointer_); }

private:
    void* pointer_ = nullptr;
};

// Dos cudaEvent para medir un tramo (se crean antes de la region medida).
class DeviceTimer {
public:
    DeviceTimer() {
        check(cudaEventCreate(&start_), "cudaEventCreate");
        const cudaError_t status = cudaEventCreate(&stop_);
        if (status != cudaSuccess) {
            cudaEventDestroy(start_);
            check(status, "cudaEventCreate");
        }
    }
    ~DeviceTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    DeviceTimer(const DeviceTimer&) = delete;
    DeviceTimer& operator=(const DeviceTimer&) = delete;

    void begin() { check(cudaEventRecord(start_), "cudaEventRecord"); }
    // Cierra el tramo, espera a que termine (aqui aparecen los errores de ejecucion de los kernels) y devuelve ms.
    double end() {
        check(cudaEventRecord(stop_), "cudaEventRecord");
        check(cudaEventSynchronize(stop_), "cudaEventSynchronize");
        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, start_, stop_), "cudaEventElapsedTime");
        return elapsed;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

// Lanza todas las pasadas sobre la tabla del dispositivo (`nodeCount` ya incluye el relleno si lo hay).
using LaunchAll = void (*)(std::int32_t* deviceCells, std::size_t nodeCount);

void launchNaive(std::int32_t* deviceCells, std::size_t nodeCount) {
    const unsigned blocksPerSide = static_cast<unsigned>((nodeCount + kNaiveBlockSide - 1) / kNaiveBlockSide);
    const dim3 threadsPerBlock(kNaiveBlockSide, kNaiveBlockSide);
    const dim3 blocks(blocksPerSide, blocksPerSide);
    for (std::size_t viaNode = 0; viaNode < nodeCount; ++viaNode) {  // el fin de cada lanzamiento es la barrera entre k
        floydWarshallNaiveStep<<<blocks, threadsPerBlock>>>(deviceCells, static_cast<int>(nodeCount), static_cast<int>(viaNode));
        check(cudaGetLastError(), "lanzamiento de floydWarshallNaiveStep");
    }
}

void launchTiled(std::int32_t* deviceCells, std::size_t nodeCount) {
    const unsigned tileCount = static_cast<unsigned>(nodeCount / kTile);
    const dim3 threadsPerBlock(kTile, kTile);  // un hilo por celda del tile
    for (unsigned pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
        // Los kernels del mismo stream corren en orden: el fin de un lanzamiento es la barrera entre fases y entre rondas.
        floydWarshallPhase1<<<1, threadsPerBlock>>>(deviceCells, static_cast<int>(nodeCount), static_cast<int>(pivotTile));
        check(cudaGetLastError(), "lanzamiento de floydWarshallPhase1");
        floydWarshallPhase2<<<dim3(tileCount, 2), threadsPerBlock>>>(deviceCells, static_cast<int>(nodeCount),
                                                                    static_cast<int>(pivotTile));
        check(cudaGetLastError(), "lanzamiento de floydWarshallPhase2");
        floydWarshallPhase3<<<dim3(tileCount, tileCount), threadsPerBlock>>>(deviceCells, static_cast<int>(nodeCount),
                                                                            static_cast<int>(pivotTile));
        check(cudaGetLastError(), "lanzamiento de floydWarshallPhase3");
    }
}

// Copia a la GPU, lanza, trae. `hostCells` es la matriz de deviceNodeCount x deviceNodeCount; el resultado la reemplaza.
void runOnDevice(std::vector<std::int32_t>& hostCells, std::size_t deviceNodeCount, LaunchAll launchAll,
                 GpuFloydTiming* timing) {
    const std::size_t cellCount = tableCellCount(deviceNodeCount);  // n*n verificado, sin desborde
    if (hostCells.size() != cellCount) {
        throw std::invalid_argument("runOnDevice: la matriz no tiene n*n celdas");
    }
    const std::size_t bytes = cellCount * sizeof(std::int32_t);

    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    check(cudaMemGetInfo(&freeBytes, &totalBytes), "cudaMemGetInfo");
    if (bytes > freeBytes) {
        throw std::runtime_error("sin memoria en la GPU: la tabla necesita " + std::to_string(bytes) +
                                 " bytes y quedan " + std::to_string(freeBytes) + " libres");
    }

    DeviceCells deviceCells(bytes);
    DeviceTimer timer;
    GpuFloydTiming measured;

    timer.begin();
    check(cudaMemcpy(deviceCells.data(), hostCells.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    measured.hostToDeviceMs = timer.end();

    timer.begin();
    launchAll(deviceCells.data(), deviceNodeCount);
    measured.kernelMs = timer.end();

    timer.begin();
    check(cudaMemcpy(hostCells.data(), deviceCells.data(), bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    measured.deviceToHostMs = timer.end();

    if (timing != nullptr) {
        *timing = measured;
    }
}

}  // namespace

void floydWarshallTiledGpu(const std::vector<std::int32_t>& initialCells, std::size_t nodeCount,
                           std::vector<std::int32_t>& resultCells, GpuFloydTiming* timing) {
    if (timing != nullptr) {
        *timing = GpuFloydTiming{};
    }
    if (nodeCount == 0) {
        resultCells.clear();
        return;
    }
    std::vector<std::int32_t> paddedCells = padCells(initialCells, nodeCount);  // relleno a multiplo de 32 en el host
    runOnDevice(paddedCells, paddedNodeCount(nodeCount), launchTiled, timing);
    resultCells = unpadCells(paddedCells, nodeCount);
}

void floydWarshallNaiveGpu(const std::vector<std::int32_t>& initialCells, std::size_t nodeCount,
                           std::vector<std::int32_t>& resultCells, GpuFloydTiming* timing) {
    if (timing != nullptr) {
        *timing = GpuFloydTiming{};
    }
    if (nodeCount == 0) {
        resultCells.clear();
        return;
    }
    if (initialCells.size() != tableCellCount(nodeCount)) {
        throw std::invalid_argument("floydWarshallNaiveGpu: la matriz inicial no tiene n*n celdas");
    }
    resultCells = initialCells;
    runOnDevice(resultCells, nodeCount, launchNaive, timing);
}

}  // namespace accel
