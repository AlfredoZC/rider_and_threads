#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "accel/CoverageField.h"
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

// Un hilo = una celda; LEE coverageBefore (const: no se puede escribir por error) y ESCRIBE coverageAfter. Es el
// cuerpo de coverageSweepRows de la CPU sin los bucles (el grid los reemplaza), con las sumas en el mismo orden.
__global__ void coverageJacobiStep(const float* coverageBefore, float* coverageAfter, const float* couriersPerCell,
                                   int gridWidth, int gridHeight) {
    const int cellX = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int cellY = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (cellX >= gridWidth || cellY >= gridHeight) return;  // hilos sobrantes del borde
    const std::size_t width = static_cast<std::size_t>(gridWidth);
    const std::size_t cell = static_cast<std::size_t>(cellY) * width + static_cast<std::size_t>(cellX);
    const bool onBorder = cellX == 0 || cellY == 0 || cellX == gridWidth - 1 || cellY == gridHeight - 1;
    if (onBorder) {
        coverageAfter[cell] = 0.0f;  // borde fijo en 0
        return;
    }
    if (couriersPerCell[cell] > 0.0f) {
        coverageAfter[cell] = couriersPerCell[cell];  // fuente fija
        return;
    }
    coverageAfter[cell] = 0.25f * (coverageBefore[cell - 1] + coverageBefore[cell + 1] +  // izquierda, derecha
                                   coverageBefore[cell - width] + coverageBefore[cell + width]);  // arriba, abajo
}

}  // namespace

CoverageGpu::~CoverageGpu() { release(); }

void CoverageGpu::release() {
    if (startEvent_ != nullptr) cudaEventDestroy(static_cast<cudaEvent_t>(startEvent_));
    if (stopEvent_ != nullptr) cudaEventDestroy(static_cast<cudaEvent_t>(stopEvent_));
    if (couriersDevice_ != nullptr) cudaFree(couriersDevice_);
    if (bufferA_ != nullptr) cudaFree(bufferA_);
    if (bufferB_ != nullptr) cudaFree(bufferB_);
    startEvent_ = nullptr;
    stopEvent_ = nullptr;
    couriersDevice_ = nullptr;
    bufferA_ = nullptr;
    bufferB_ = nullptr;
    resultInA_ = true;
    ran_ = false;
    gridWidth_ = 0;
    gridHeight_ = 0;
    cellCount_ = 0;
    cudaGetLastError();
}

std::string CoverageGpu::allocate(int gridWidth, int gridHeight) {
    release();
    if (gridWidth < 1 || gridHeight < 1 || gridWidth > kMaxGridSide || gridHeight > kMaxGridSide) {
        return "CoverageGpu: tamano de grilla fuera de rango";
    }
    const std::size_t cellCount = static_cast<std::size_t>(gridWidth) * static_cast<std::size_t>(gridHeight);
    const std::size_t bytes = cellCount * sizeof(float);
    std::string errorMessage = check(cudaMalloc(&couriersDevice_, bytes), "cudaMalloc de los couriers");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(&bufferA_, bytes), "cudaMalloc del buffer A");
    if (errorMessage.empty()) errorMessage = check(cudaMalloc(&bufferB_, bytes), "cudaMalloc del buffer B");
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

std::string CoverageGpu::upload(const std::vector<float>& couriersPerCell, double* elapsedMs) {
    if (couriersDevice_ == nullptr || couriersPerCell.size() != cellCount_) {
        return "upload: buffer no reservado o tamano distinto";
    }
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) {
        errorMessage = check(cudaMemcpy(couriersDevice_, couriersPerCell.data(), cellCount_ * sizeof(float),
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

std::string CoverageGpu::run(int sweepCount, double* elapsedMs) {
    if (couriersDevice_ == nullptr || sweepCount < 0) {
        return "run: sin buffers o barridos negativos";
    }
    ran_ = false;
    const dim3 threadsPerBlock(kBlockSide, kBlockSide);
    const dim3 blockCount(static_cast<unsigned>((gridWidth_ + kBlockSide - 1) / kBlockSide),
                          static_cast<unsigned>((gridHeight_ + kBlockSide - 1) / kBlockSide));
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    // Barrido 0: el estado inicial se arma dentro de la GPU (mas barato que volver a cruzar el PCIe).
    if (errorMessage.empty()) {
        errorMessage = check(cudaMemcpy(bufferA_, couriersDevice_, cellCount_ * sizeof(float), cudaMemcpyDeviceToDevice),
                             "cudaMemcpy D2D del barrido 0");
    }
    float* coverageBefore = static_cast<float*>(bufferA_);
    float* coverageAfter = static_cast<float*>(bufferB_);
    // UN lanzamiento por barrido: el fin del kernel es la barrera global entre barridos (misma cola). Se encolan
    // sin esperar; el intercambio es de PUNTEROS en el host, la GPU no copia nada.
    for (int sweep = 0; errorMessage.empty() && sweep < sweepCount; ++sweep) {
        coverageJacobiStep<<<blockCount, threadsPerBlock>>>(coverageBefore, coverageAfter,
                                                            static_cast<const float*>(couriersDevice_), gridWidth_,
                                                            gridHeight_);
        errorMessage = check(cudaGetLastError(), "lanzamiento de coverageJacobiStep");
        std::swap(coverageBefore, coverageAfter);
    }
    // Tras el ultimo intercambio el resultado mas nuevo esta en coverageBefore.
    resultInA_ = coverageBefore == static_cast<float*>(bufferA_);
    if (errorMessage.empty()) errorMessage = check(cudaEventRecord(stop), "cudaEventRecord");
    if (errorMessage.empty()) errorMessage = check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed = 0.0f;
    if (errorMessage.empty()) errorMessage = check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    if (errorMessage.empty() && elapsedMs != nullptr) *elapsedMs = elapsed;
    ran_ = errorMessage.empty();
    return errorMessage;
}

std::string CoverageGpu::download(std::vector<float>& coverage, double* elapsedMs) {
    if (!ran_) {
        return "download: no hay resultado (run no se ejecuto o fallo)";
    }
    coverage.assign(cellCount_, 0.0f);
    const cudaEvent_t start = static_cast<cudaEvent_t>(startEvent_);
    const cudaEvent_t stop = static_cast<cudaEvent_t>(stopEvent_);
    std::string errorMessage = check(cudaEventRecord(start), "cudaEventRecord");
    if (errorMessage.empty()) {
        // La copia espera a todo lo encolado; si un kernel fallo al ejecutarse, el error aparece aqui.
        errorMessage = check(cudaMemcpy(coverage.data(), resultInA_ ? bufferA_ : bufferB_, cellCount_ * sizeof(float),
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
