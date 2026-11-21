#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accel/ThreadPool.h"

namespace accel {

class GpuWorker;

// Aplicacion 06: campo de cobertura de la flota (stencil de 5 puntos, barridos de Jacobi con doble buffer).
// Grilla con tres tipos de celdas: FUENTES (celdas con couriers: quedan fijas con su cantidad), BORDE (el
// contorno: fijo en 0) y LIBRES (en cada barrido valen el promedio de sus 4 vecinas del barrido ANTERIOR).
// El estado inicial (barrido 0) es la grilla de couriers. Todas las versiones hacen la MISMA cantidad de
// barridos (nunca "hasta converger" cada una por su lado).
//
//   coverageFieldCpu1   un hilo
//   coverageFieldCpuN   por barrido, las filas en franjas con ThreadPool::parallelFor (la vuelta de
//                       parallelFor es la barrera entre barridos); dos buffers que se intercambian
//   coverageFieldGpu    un hilo por celda y UN lanzamiento por barrido (el fin del kernel es la barrera
//                       global); dos buffers en la GPU cuyos punteros se intercambian en el host
//   computeCoverage     lo que usa el programa: GPU por el GpuWorker con plazo, CPU si falla o vence
//
// Las tres versiones suman en el mismo orden y 0,25f * (a + b + c + d) no tiene multiplicacion seguida de
// suma (nada que fusionar en FMA): en la practica dan resultados identicos; se comparan igual con
// maxRelativeDifference y kCoverageTolerance.

constexpr double kCoverageTolerance = 1e-5;
// Tope de barridos aceptado (evita pedidos absurdos que dejen a un hilo ocupado indefinidamente).
constexpr int kMaxSweepCount = 1000000;

// Lanza std::invalid_argument si los argumentos no describen un problema valido: lados en 1..kMaxGridSide
// (el de DemandHeatmap.h), couriersCellCount == ancho * alto y sweepCount en 0..kMaxSweepCount.
void checkCoverageArguments(std::size_t couriersCellCount, int gridWidth, int gridHeight, int sweepCount);

std::vector<float> coverageFieldCpu1(const std::vector<float>& couriersPerCell, int gridWidth, int gridHeight,
                                     int sweepCount);

std::vector<float> coverageFieldCpuN(ThreadPool& pool, const std::vector<float>& couriersPerCell, int gridWidth,
                                     int gridHeight, int sweepCount);

// Memoria de la GPU para una grilla, con pasos separados para poder medir cada uno. Solo la usa el hilo dueño
// de la GPU (o el bench). Cada metodo devuelve "" si salio bien o un mensaje; ninguno lanza.
// Sin CUDA (accel/cpu/CoverageFieldCpu.cpp) todos devuelven "compilado sin CUDA".
class CoverageGpu {
public:
    CoverageGpu() = default;
    ~CoverageGpu();
    CoverageGpu(const CoverageGpu&) = delete;
    CoverageGpu& operator=(const CoverageGpu&) = delete;

    // Reserva tres buffers (couriers y los dos del doble buffer) y dos cudaEvent; libera lo anterior.
    std::string allocate(int gridWidth, int gridHeight);
    void release();
    std::string upload(const std::vector<float>& couriersPerCell, double* elapsedMs);
    // Barrido 0 (copia dispositivo -> dispositivo) y sweepCount lanzamientos con intercambio de punteros.
    // No espera a cada barrido: elapsedMs (cudaEvent) abarca todos.
    std::string run(int sweepCount, double* elapsedMs);
    // Baja el ultimo resultado (el de run); espera a que termine todo lo encolado.
    std::string download(std::vector<float>& coverage, double* elapsedMs);

private:
    void* couriersDevice_ = nullptr;
    void* bufferA_ = nullptr;
    void* bufferB_ = nullptr;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
    bool resultInA_ = true;  // tras el ultimo intercambio, donde quedo el resultado
    bool ran_ = false;
    int gridWidth_ = 0;
    int gridHeight_ = 0;
    std::size_t cellCount_ = 0;
};

// Una pasada completa en la GPU (reserva, sube, barre, baja, libera). Para correr DENTRO de una tarea del
// GpuWorker. Lanza GpuFatalError (error CUDA pegajoso) o std::runtime_error.
std::vector<float> coverageFieldGpu(const std::vector<float>& couriersPerCell, int gridWidth, int gridHeight,
                                    int sweepCount);

struct CoverageResult {
    std::vector<float> coverage;
    bool usedGpu = false;  // false: lo calculo la CPU (sin GPU, fallo o plazo vencido)
};

// Lo que usa la simulacion: igual que smoothDemand (plazo REAL, tarea por valor, respaldo cpuN).
CoverageResult computeCoverage(GpuWorker* gpu, ThreadPool& cpuPool, const std::vector<float>& couriersPerCell,
                               int gridWidth, int gridHeight, int sweepCount, std::int64_t gpuTimeoutMs);

struct CoverageZone {
    int zoneX = 0;
    int zoneY = 0;
    float meanCoverage = 0.0f;
};

// Zona de zoneSide x zoneSide celdas (las del borde de la grilla pueden ser mas chicas) con menor cobertura
// media; en empate, la primera (por filas). Lanza std::invalid_argument con argumentos invalidos. Ojo: las zonas
// del contorno salen castigadas porque el borde vale 0 por diseno.
CoverageZone leastCoveredZone(const std::vector<float>& coverage, int gridWidth, int gridHeight, int zoneSide);

}  // namespace accel
