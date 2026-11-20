#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accel/ThreadPool.h"

namespace accel {

class GpuWorker;

// Aplicacion 05: mapa de calor de la demanda. Suaviza un conteo de pedidos por celda con un filtro
// gaussiano (convolucion 2D). Cada celda de salida es la suma ponderada de sus vecinas del filtro; lo que
// cae fuera de la grilla cuenta 0. El filtro se pasa como argumento (lado 2 * filterRadius + 1, fila por
// fila empezando en el desplazamiento (-radio, -radio)).
//
//   smoothDemandCpu1   un hilo (referencia)
//   smoothDemandCpuN   las filas de salida en franjas con ThreadPool::parallelFor
//   smoothDemandGpu    un hilo por celda; el filtro vive en memoria constante (kernel PRINCIPAL)
//   DemandKernel::tiledHalo  variante opcional: tile de memoria compartida con halo (mas rapida)
//   smoothDemand       lo que usa el programa: GPU por el GpuWorker con plazo, CPU si falla o vence
//
// Las tres versiones suman en el mismo orden (recorrido del filtro por filas); la GPU fusiona a*b+c en una
// FMA y la CPU no, asi que pueden diferir en el ultimo bit: se comparan con maxRelativeDifference y
// kDemandTolerance (ver su comentario).

// Los pesos del filtro caben en memoria constante hasta 7 x 7 = 49 floats (196 bytes).
constexpr int kMaxFilterRadius = 3;
constexpr int kMaxFilterSize = 2 * kMaxFilterRadius + 1;
// Lado maximo de la grilla: con size_t las cuentas no desbordan y los bloques de 16 x 16 caben en el grid.
constexpr int kMaxGridSide = 1 << 15;
constexpr double kDefaultFilterSigmaCells = 1.5;

// Tolerancia de |candidato - referencia| / (1 + |referencia|). Un float guarda 24 bits (~6e-8 por
// operacion); con 49 sumas de terminos positivos el peor caso teorico ronda 3e-6. Medido: 1,2e-7. Un vecino
// de la esquina del filtro olvidado cambia una celda en ~1e-3 o mas: se detecta. El "+1" evita dividir por
// casi cero en las celdas vacias (alli la comparacion es absoluta).
constexpr double kDemandTolerance = 1e-5;

// Lanza std::invalid_argument si los argumentos no describen un problema valido: lados en 1..kMaxGridSide,
// ordersCellCount == ancho * alto, radio en 0..kMaxFilterRadius y filterWeightCount == (2 * radio + 1)^2.
void checkDemandArguments(std::size_t ordersCellCount, int gridWidth, int gridHeight, std::size_t filterWeightCount,
                          int filterRadius);

// Filtro gaussiano normalizado (suma 1) de lado 2 * filterRadius + 1: peso ~ exp(-d^2 / 2 sigma^2).
std::vector<float> makeGaussianWeights(int filterRadius, double sigmaCells);

std::vector<float> smoothDemandCpu1(const std::vector<float>& ordersPerCell, int gridWidth, int gridHeight,
                                    const std::vector<float>& filterWeights, int filterRadius);

std::vector<float> smoothDemandCpuN(ThreadPool& pool, const std::vector<float>& ordersPerCell, int gridWidth,
                                    int gridHeight, const std::vector<float>& filterWeights, int filterRadius);

enum class DemandKernel { constantFilter, tiledHalo };

// Memoria de la GPU para una grilla y su filtro, con pasos separados para poder medir cada uno. Solo la usa el
// hilo dueño de la GPU (o el bench). Cada metodo devuelve "" si salio bien o un mensaje; ninguno lanza.
// Sin CUDA (accel/cpu/DemandHeatmapCpu.cpp) todos devuelven "compilado sin CUDA".
class DemandHeatGpu {
public:
    DemandHeatGpu() = default;
    ~DemandHeatGpu();
    DemandHeatGpu(const DemandHeatGpu&) = delete;
    DemandHeatGpu& operator=(const DemandHeatGpu&) = delete;

    // Reserva los dos buffers (conteos y mapa de calor) y dos cudaEvent; libera lo anterior.
    std::string allocate(int gridWidth, int gridHeight);
    void release();
    // Copia el filtro a la memoria constante (UNA vez por filtro). Un solo filtro activo por programa.
    std::string setFilter(const std::vector<float>& filterWeights, int filterRadius);
    std::string upload(const std::vector<float>& ordersPerCell, double* elapsedMs);
    std::string launch(DemandKernel kernel, double* elapsedMs);
    std::string download(std::vector<float>& demandHeat, double* elapsedMs);

private:
    void* ordersDevice_ = nullptr;
    void* heatDevice_ = nullptr;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
    int gridWidth_ = 0;
    int gridHeight_ = 0;
    int filterRadius_ = -1;
    std::size_t cellCount_ = 0;
};

// Una pasada completa en la GPU (reserva, sube, lanza, baja, libera). Para correr DENTRO de una tarea del
// GpuWorker. Lanza GpuFatalError (error CUDA pegajoso: el worker queda no saludable) o std::runtime_error.
std::vector<float> smoothDemandGpu(const std::vector<float>& ordersPerCell, int gridWidth, int gridHeight,
                                   const std::vector<float>& filterWeights, int filterRadius,
                                   DemandKernel kernel = DemandKernel::constantFilter);

struct DemandSmoothing {
    std::vector<float> demandHeat;
    bool usedGpu = false;  // false: lo calculo la CPU (sin GPU, fallo o plazo vencido)
};

// Lo que usa la simulacion. Con `gpu` != nullptr envia el trabajo al GpuWorker con un plazo REAL de
// gpuTimeoutMs y espera hasta ese plazo; si la GPU no esta disponible, falla o vence, calcula en la CPU
// con `cpuPool` (cpuN). La tarea captura los datos POR VALOR (puede sobrevivir a esta llamada).
// Bloquea a quien llama hasta el plazo como mucho: NO se llama desde el hilo de la ventana.
DemandSmoothing smoothDemand(GpuWorker* gpu, ThreadPool& cpuPool, const std::vector<float>& ordersPerCell,
                             int gridWidth, int gridHeight, const std::vector<float>& filterWeights,
                             int filterRadius, std::int64_t gpuTimeoutMs);

// max |candidato - referencia| / (1 + |referencia|). Lanza std::invalid_argument si los tamanos difieren.
double maxRelativeDifference(const std::vector<float>& reference, const std::vector<float>& candidate);

// Indice de la celda con mas demanda; en empate, el menor (CPU y GPU informan la misma). 0 si esta vacio.
std::size_t hottestCellIndex(const std::vector<float>& demandHeat);

}  // namespace accel
