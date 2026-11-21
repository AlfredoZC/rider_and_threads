#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Config.h"
#include "Fleet.h"
#include "SimContext.h"
#include "accel/ThreadPool.h"

// Campo de cobertura de la flota (aplicacion 06): cuenta los couriers por celda de la grilla de map.bounds (con
// margen 1, para que ninguno caiga en el contorno fijo en 0), calcula el campo con accel::computeCoverage (GPU
// por el GpuWorker, CPU de respaldo) y deja el ultimo resultado listo para que la ventana lo dibuje sin
// bloquearse. Mismo patron que DemandHeatService.

constexpr int kCoverageGridSide = 128;                    // grilla de 128 x 128
constexpr int kCoverageSweepCount = 200;                  // barridos de Jacobi (los mismos en CPU y GPU)
constexpr int kCoverageZoneSide = 16;                     // zonas de 16 x 16 celdas para "zona menos cubierta"
constexpr std::int64_t kCoverageRefreshRealMs = 2500;     // cada ~2,5 s reales
constexpr std::int64_t kCoverageGpuTimeoutRealMs = 1000;  // plazo REAL de la GPU antes del respaldo CPU

// Punto a la fraccion `ratio` (0..1, recortada) del segmento from -> to, ambos (lat, lon).
std::pair<double, double> interpolateLatLon(const std::pair<double, double>& from, const std::pair<double, double>& to,
                                            double ratio);

struct CoverageSnapshot {
    std::shared_ptr<const std::vector<float>> coverage;  // nulo hasta el primer resultado
    std::uint64_t version = 0;                           // sube con cada resultado nuevo
    float maxValue = 0.0f;
    float scaleValue = 0.0f;  // percentil 95 del campo: escala de color robusta (una celda con ~2000 couriers al arrancar domina el maximo)
    bool usedGpu = false;
};

// Linea de estadisticas del final ("[estadisticas] zona menos cubierta: ...").
std::string leastCoveredZoneLine(const Bounds& bounds, int gridWidth, int gridHeight, const std::vector<float>& coverage,
                                 int zoneSide);

class CoverageFieldService {
public:
    CoverageFieldService(const Config& cfg, const SimContext& ctx, const Fleet& fleet);
    ~CoverageFieldService();  // pide parar y une el hilo
    CoverageFieldService(const CoverageFieldService&) = delete;
    CoverageFieldService& operator=(const CoverageFieldService&) = delete;

    void start();  // hilo propio: refresca cada kCoverageRefreshRealMs, esperando bloqueado
    void requestStop();
    void join();

    // Ultimo resultado listo (barato; no espera a la GPU). Seguro desde cualquier hilo.
    CoverageSnapshot latest() const;

    // Toma las posiciones, calcula y publica ahora. Lo llama el hilo del servicio; tambien main, DESPUES de join().
    void refresh();

    std::string leastCoveredZone() const;

private:
    const Config& cfg_;
    const SimContext& ctx_;
    const Fleet& fleet_;
    std::unique_ptr<accel::ThreadPool> cpuPool_;  // se crea la primera vez que hace falta el respaldo

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    CoverageSnapshot latest_;
    std::thread thread_;
};
