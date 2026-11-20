#include "DemandHeat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "accel/DemandHeatmap.h"

std::size_t demandCellIndex(const Bounds& bounds, int gridWidth, int gridHeight, double lat, double lon) {
    const double columnFraction = (lon - bounds.west) / (bounds.east - bounds.west);
    const double rowFraction = (bounds.north - lat) / (bounds.north - bounds.south);
    const auto toCell = [](double fraction, int cellCount) {
        if (!(fraction > 0.0)) return 0;  // tambien NaN
        const double scaled = fraction * cellCount;
        return scaled >= cellCount ? cellCount - 1 : static_cast<int>(scaled);
    };
    const int cellX = toCell(columnFraction, gridWidth);
    const int cellY = toCell(rowFraction, gridHeight);
    return static_cast<std::size_t>(cellY) * static_cast<std::size_t>(gridWidth) + static_cast<std::size_t>(cellX);
}

std::pair<double, double> demandCellCenter(const Bounds& bounds, int gridWidth, int gridHeight, int cellX,
                                           int cellY) {
    const double lat = bounds.north - (cellY + 0.5) / gridHeight * (bounds.north - bounds.south);
    const double lon = bounds.west + (cellX + 0.5) / gridWidth * (bounds.east - bounds.west);
    return {lat, lon};
}

std::vector<float> countPointsPerCell(const Bounds& bounds, int gridWidth, int gridHeight,
                                      const std::vector<std::pair<double, double>>& latLonPoints, int borderMargin) {
    std::vector<float> pointsPerCell(static_cast<std::size_t>(gridWidth) * static_cast<std::size_t>(gridHeight), 0.0f);
    // En una grilla muy chica el margen se recorta para que siempre quede al menos una celda valida.
    const int marginX = std::max(0, std::min(borderMargin, (gridWidth - 1) / 2));
    const int marginY = std::max(0, std::min(borderMargin, (gridHeight - 1) / 2));
    for (const auto& point : latLonPoints) {
        const std::size_t cell = demandCellIndex(bounds, gridWidth, gridHeight, point.first, point.second);
        const int cellX = std::clamp(static_cast<int>(cell % static_cast<std::size_t>(gridWidth)), marginX, gridWidth - 1 - marginX);
        const int cellY = std::clamp(static_cast<int>(cell / static_cast<std::size_t>(gridWidth)), marginY, gridHeight - 1 - marginY);
        pointsPerCell[static_cast<std::size_t>(cellY) * static_cast<std::size_t>(gridWidth) + static_cast<std::size_t>(cellX)] += 1.0f;
    }
    return pointsPerCell;
}

std::string hottestZoneLine(const Bounds& bounds, int gridWidth, int gridHeight, const std::vector<float>& demandHeat) {
    if (demandHeat.empty() || demandHeat[accel::hottestCellIndex(demandHeat)] <= 0.0f) {
        return "[estadisticas] zona mas caliente: sin pedidos recientes";
    }
    const std::size_t hottest = accel::hottestCellIndex(demandHeat);
    const int cellX = static_cast<int>(hottest % static_cast<std::size_t>(gridWidth));
    const int cellY = static_cast<int>(hottest / static_cast<std::size_t>(gridWidth));
    const auto center = demandCellCenter(bounds, gridWidth, gridHeight, cellX, cellY);
    char line[200];
    std::snprintf(line, sizeof line, "[estadisticas] zona mas caliente: celda (%d,%d) ~ %.5f,%.5f, valor %.3f", cellX,
                  cellY, center.first, center.second, static_cast<double>(demandHeat[hottest]));
    return line;
}

DemandHeatService::DemandHeatService(const Config& cfg, const SimContext& ctx)
    : cfg_(cfg),
      ctx_(ctx),
      filterWeights_(accel::makeGaussianWeights(kDemandFilterRadius, accel::kDefaultFilterSigmaCells)) {}

DemandHeatService::~DemandHeatService() {
    requestStop();
    join();
}

void DemandHeatService::start() {
    thread_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            lock.unlock();
            refresh();
            lock.lock();
            // Espera bloqueada (sin sondeo) hasta el proximo refresco o hasta que pidan parar.
            wake_.wait_for(lock, std::chrono::milliseconds(kDemandRefreshRealMs), [this] { return stopping_; });
        }
    });
}

void DemandHeatService::requestStop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
}

void DemandHeatService::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

DemandHeatSnapshot DemandHeatService::latest() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return latest_;
}

void DemandHeatService::refresh() {
    // 1) Pedidos de los ultimos N minutos simulados, por el nodo del cliente.
    const std::int64_t sinceMs = ctx_.clock.nowMs() - kDemandWindowSimMs;
    std::vector<std::pair<double, double>> latLonPoints;
    for (const std::string& nodeId : ctx_.book.dropoffNodesCreatedSince(sinceMs)) {
        const NodeIndex node = ctx_.graph.indexOf(nodeId);
        if (node != kNoNode) {
            latLonPoints.emplace_back(ctx_.graph.latOf(node), ctx_.graph.lonOf(node));
        }
    }
    const std::vector<float> ordersPerCell =
        countPointsPerCell(cfg_.bounds, kDemandGridSide, kDemandGridSide, latLonPoints);
    // 2) Suavizado: GPU por el GpuWorker con plazo real, CPU si falla o vence. Este hilo puede esperar; el de la
    //    ventana nunca llama aqui, solo lee latest().
    if (!cpuPool_) {
        cpuPool_ = std::make_unique<accel::ThreadPool>(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
    }
    accel::DemandSmoothing smoothing;
    try {
        smoothing = accel::smoothDemand(ctx_.gpu, *cpuPool_, ordersPerCell, kDemandGridSide, kDemandGridSide,
                                        filterWeights_, kDemandFilterRadius, kDemandGpuTimeoutRealMs);
    } catch (const std::exception&) {
        return;  // con estas constantes no hay argumentos invalidos; ante la duda se conserva el ultimo mapa
    }
    // 3) Publicar.
    auto published = std::make_shared<const std::vector<float>>(std::move(smoothing.demandHeat));
    const float maxValue = published->empty() ? 0.0f : (*published)[accel::hottestCellIndex(*published)];
    const std::lock_guard<std::mutex> guard(mutex_);
    latest_.demandHeat = std::move(published);
    latest_.maxValue = maxValue;
    latest_.usedGpu = smoothing.usedGpu;
    ++latest_.version;
}

std::string DemandHeatService::hottestZone() const {
    const DemandHeatSnapshot snapshot = latest();
    static const std::vector<float> none;
    return hottestZoneLine(cfg_.bounds, kDemandGridSide, kDemandGridSide,
                           snapshot.demandHeat ? *snapshot.demandHeat : none);
}
