#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

#include "accel/DemandHeatmap.h"

namespace accel {

namespace {

// Convierte un cudaError_t en mensaje ("" si no hubo error) y limpia el error no pegajoso.
std::string check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return "";
    }
    const std::string message = std::string(operation) + ": " + cudaGetErrorString(status);
    cudaGetLastError();
    return message;
}

constexpr int kBlockSide = 16;  // 16 x 16 = 256 hilos = 8 warps; 100 % de ocupacion en esta GPU
constexpr int kTileWithMaxHalo = kBlockSide + 2 * kMaxFilterRadius;

// Los pesos viven en memoria CONSTANTE: hasta 7 x 7 = 49 floats. Los 32 hilos de un warp leen el mismo peso
// al mismo tiempo (broadcast desde la cache de constantes).
__constant__ float gaussianWeight[kMaxFilterSize * kMaxFilterSize];

// Kernel PRINCIPAL: un hilo = una celda de salida. Es la CPU de 1 hilo sin los dos bucles de afuera.
__global__ void smoothDemandKernel(const float* ordersPerCell, float* demandHeat, int gridWidth, int gridHeight,
                                   int filterRadius) {
    const int cellX = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int cellY = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (cellX >= gridWidth || cellY >= gridHeight) return;  // hilos sobrantes del borde
    const int filterSize = 2 * filterRadius + 1;
    float weightedSum = 0.0f;
    for (int offsetY = -filterRadius; offsetY <= filterRadius; ++offsetY) {
        for (int offsetX = -filterRadius; offsetX <= filterRadius; ++offsetX) {
            const int neighborX = cellX + offsetX;
            const int neighborY = cellY + offsetY;
            if (neighborX >= 0 && neighborX < gridWidth && neighborY >= 0 && neighborY < gridHeight) {
                const int weightIndex = (offsetY + filterRadius) * filterSize + (offsetX + filterRadius);
                weightedSum += gaussianWeight[weightIndex] *
                               ordersPerCell[static_cast<std::size_t>(neighborY) * gridWidth + neighborX];
            }
        }
    }
    demandHeat[static_cast<std::size_t>(cellY) * gridWidth + cellX] = weightedSum;
}

// Variante opcional: cada bloque copia una vez a memoria compartida sus 16 x 16 celdas mas un halo del radio
// del filtro; fuera del mapa se carga 0, asi el bucle del filtro no necesita el `if`.
__global__ void smoothDemandTiledKernel(const float* ordersPerCell, float* demandHeat, int gridWidth,
                                        int gridHeight, int filterRadius) {
    __shared__ float ordersTile[kTileWithMaxHalo][kTileWithMaxHalo];
    const int tileWithHalo = kBlockSide + 2 * filterRadius;
    const int tileStartX = static_cast<int>(blockIdx.x) * kBlockSide - filterRadius;  // puede ser negativo
    const int tileStartY = static_cast<int>(blockIdx.y) * kBlockSide - filterRadius;
    const int threadInBlock = static_cast<int>(threadIdx.y) * kBlockSide + static_cast<int>(threadIdx.x);
    const int tileCellCount = tileWithHalo * tileWithHalo;
    for (int load = threadInBlock; load < tileCellCount; load += kBlockSide * kBlockSide) {
        const int tileY = load / tileWithHalo;
        const int tileX = load % tileWithHalo;
        const int mapX = tileStartX + tileX;
        const int mapY = tileStartY + tileY;
        const bool insideMap = mapX >= 0 && mapX < gridWidth && mapY >= 0 && mapY < gridHeight;
        ordersTile[tileY][tileX] =
            insideMap ? ordersPerCell[static_cast<std::size_t>(mapY) * gridWidth + mapX] : 0.0f;
    }
    __syncthreads();  // el tile completo antes de leerlo; la barrera va ANTES del return

    const int cellX = static_cast<int>(blockIdx.x) * kBlockSide + static_cast<int>(threadIdx.x);
    const int cellY = static_cast<int>(blockIdx.y) * kBlockSide + static_cast<int>(threadIdx.y);
    if (cellX >= gridWidth || cellY >= gridHeight) return;
    const int filterSize = 2 * filterRadius + 1;
    float weightedSum = 0.0f;
    for (int offsetY = -filterRadius; offsetY <= filterRadius; ++offsetY) {
        for (int offsetX = -filterRadius; offsetX <= filterRadius; ++offsetX) {
            const int weightIndex = (offsetY + filterRadius) * filterSize + (offsetX + filterRadius);
            weightedSum += gaussianWeight[weightIndex] *
                           ordersTile[threadIdx.y + filterRadius + offsetY][threadIdx.x + filterRadius + offsetX];
        }
    }
    demandHeat[static_cast<std::size_t>(cellY) * gridWidth + cellX] = weightedSum;
}

}  // namespace

DemandHeatGpu::~DemandHeatGpu() { release(); }

void DemandHeatGpu::release() {
    if (startEvent_ != nullptr) cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
    if (stopEvent_ != nullptr) cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
    if (ordersDevice_ != nullptr) cudaFree(ordersDevice_);
    if (heatDevice_ != nullptr) cudaFree(heatDevice_);
    startEvent_ = nullptr;
    stopEvent_ = nullptr;
    ordersDevice_ = nullptr;
    heatDevice_ = nullptr;
    gridWidth_ = 0;
    gridHeight_ = 0;
    filterRadius_ = -1;
    cellCount_ = 0;
    cudaGetLastError();
}

std::string DemandHeatGpu::allocate(int gridWidth, int gridHeight) {
    release();
    if (gridWidth < 1 || gridHeight < 1 || gridWidth > kMaxGridSide || gridHeight > kMaxGridSide) {
        return "DemandHeatGpu: tamano de grilla fuera de rango";
    }
    const std::size_t cellCount = static_cast<std::size_t>(gridWidth) * static_cast<std::size_t>(gridHeight);
    const std::size_t bytes = cellCount * sizeof(float);
    std::string errorMessage = check(cudaMalloc(&ordersDevice_, bytes), "cudaMalloc de los conteos");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(&heatDevice_, bytes), "cudaMalloc del mapa de calor");
    cudaEvent_t startCreated = nullptr;
    cudaEvent_t stopCreated = nullptr;
    if (errorMessage.empty()) errorMessage = check(cudaEventCreate(&startCreated), "cudaEventCreate");
    if (errorMessage.empty()) {
        startEvent_ = startCreated;
        errorMessage = check(cudaEventCreate(&stopCreated), "cudaEventCreate");
    }
    if (errorMessage.empty()) stopEvent_ = stopCreated;
    if (!errorMessage.empty()) {
        release();
        return errorMessage;
    }
    gridWidth_ = gridWidth;
    gridHeight_ = gridHeight;
    cellCount_ = cellCount;
    return "";
}

std::string DemandHeatGpu::setFilter(const std::vector<float>& filterWeights, int filterRadius) {
    const std::size_t filterSize = static_cast<std::size_t>(2 * filterRadius + 1);
    if (filterRadius < 0 || filterRadius > kMaxFilterRadius || filterWeights.size() != filterSize * filterSize) {
        return "DemandHeatGpu: filtro invalido";
    }
    // El filtro no cambia entre lanzamientos: se copia UNA vez a la memoria constante.
    const std::string errorMessage = check(
        cudaMemcpyToSymbol(gaussianWeight, filterWeights.data(), filterWeights.size() * sizeof(float)),
        "cudaMemcpyToSymbol del filtro");
    if (errorMessage.empty()) filterRadius_ = filterRadius;
    return errorMessage;
}

std::string DemandHeatGpu::upload(const std::vector<float>& ordersPerCell, double* elapsedMs) {
    if (ordersDevice_ == nullptr || ordersPerCell.size() != cellCount_) {
        return "upload: buffer no reservado o tamano distinto";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) {
        errorMessage = check(cudaMemcpy(ordersDevice_, ordersPerCell.data(), cellCount_ * sizeof(float),
                                        cudaMemcpyHostToDevice),
                             "cudaMemcpy H2D");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

std::string DemandHeatGpu::launch(DemandKernel kernel, double* elapsedMs) {
    if (ordersDevice_ == nullptr || filterRadius_ < 0) {
        return "launch: sin buffers o sin filtro";
    }
    const dim3 threadsPerBlock(kBlockSide, kBlockSide);
    const dim3 blockCount(static_cast<unsigned>((gridWidth_ + kBlockSide - 1) / kBlockSide),
                          static_cast<unsigned>((gridHeight_ + kBlockSide - 1) / kBlockSide));
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) {
        const float* orders = static_cast<const float*>(ordersDevice_);
        float* heat = static_cast<float*>(heatDevice_);
        if (kernel == DemandKernel::tiledHalo) {
            smoothDemandTiledKernel<<<blockCount, threadsPerBlock>>>(orders, heat, gridWidth_, gridHeight_,
                                                                      filterRadius_);
        } else {
            smoothDemandKernel<<<blockCount, threadsPerBlock>>>(orders, heat, gridWidth_, gridHeight_,
                                                                filterRadius_);
        }
        errorMessage = check(cudaGetLastError(), "lanzamiento de smoothDemand");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

std::string DemandHeatGpu::download(std::vector<float>& demandHeat, double* elapsedMs) {
    if (heatDevice_ == nullptr) {
        return "download: buffer no reservado";
    }
    demandHeat.assign(cellCount_, 0.0f);
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) {
        // La copia espera a que termine el kernel; si este fallo al ejecutarse, el error aparece aqui.
        errorMessage = check(cudaMemcpy(demandHeat.data(), heatDevice_, cellCount_ * sizeof(float),
                                        cudaMemcpyDeviceToHost),
                             "cudaMemcpy D2H");
    }
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    return errorMessage;
}

}  // namespace accel
