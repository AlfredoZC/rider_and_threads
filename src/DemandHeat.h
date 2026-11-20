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
#include "SimContext.h"
#include "accel/ThreadPool.h"

// Mapa de calor de la demanda (aplicacion 05): cuenta los pedidos recientes por celda de una grilla sobre
// map.bounds, la suaviza con accel::smoothDemand (GPU por el GpuWorker, CPU de respaldo) y deja el ultimo
// resultado listo para que la ventana lo dibuje sin bloquearse.

constexpr int kDemandGridSide = 128;                           // grilla de 128 x 128
constexpr int kDemandFilterRadius = 3;                         // filtro gaussiano 7 x 7, sigma 1,5 celdas
constexpr std::int64_t kDemandWindowSimMs = 10LL * 60 * 1000;  // "recientes" = ultimos 10 minutos simulados
constexpr std::int64_t kDemandRefreshRealMs = 2000;            // cada ~2 s reales
constexpr std::int64_t kDemandGpuTimeoutRealMs = 1000;         // plazo REAL de la GPU antes del respaldo CPU

// Celda (fila 0 = norte, como la imagen) de un punto; fuera de los limites se recorta al borde.
std::size_t demandCellIndex(const Bounds& bounds, int gridWidth, int gridHeight, double lat, double lon);
// Centro de la celda (cellX, cellY) como (lat, lon).
std::pair<double, double> demandCellCenter(const Bounds& bounds, int gridWidth, int gridHeight, int cellX,
                                           int cellY);
// Cuantos puntos (lat, lon) caen en cada celda. borderMargin = 0 para pedidos; 1 para couriers: ninguno cae en
// el contorno de la grilla (que el campo de cobertura fija en 0).
std::vector<float> countPointsPerCell(const Bounds& bounds, int gridWidth, int gridHeight,
                                      const std::vector<std::pair<double, double>>& latLonPoints,
                                      int borderMargin = 0);

struct DemandHeatSnapshot {
    std::shared_ptr<const std::vector<float>> demandHeat;  // nulo hasta el primer resultado
    std::uint64_t version = 0;                             // sube con cada resultado nuevo
    float maxValue = 0.0f;
    bool usedGpu = false;
};

// Linea de estadisticas del final ("[estadisticas] zona mas caliente: ...").
std::string hottestZoneLine(const Bounds& bounds, int gridWidth, int gridHeight, const std::vector<float>& demandHeat);

class DemandHeatService {
public:
    DemandHeatService(const Config& cfg, const SimContext& ctx);
    ~DemandHeatService();  // pide parar y une el hilo
    DemandHeatService(const DemandHeatService&) = delete;
    DemandHeatService& operator=(const DemandHeatService&) = delete;

    void start();  // hilo propio: refresca cada kDemandRefreshRealMs, esperando bloqueado
    void requestStop();
    void join();

    // Ultimo resultado listo (barato; no espera a la GPU). Seguro desde cualquier hilo.
    DemandHeatSnapshot latest() const;

    // Cuenta, suaviza y publica ahora. Lo llama el hilo del servicio; tambien main, DESPUES de join().
    void refresh();

    std::string hottestZone() const;

private:
    const Config& cfg_;
    const SimContext& ctx_;
    std::vector<float> filterWeights_;
    std::unique_ptr<accel::ThreadPool> cpuPool_;  // se crea la primera vez que hace falta el respaldo

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;
    DemandHeatSnapshot latest_;
    std::thread thread_;
};
