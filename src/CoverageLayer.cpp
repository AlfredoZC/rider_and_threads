#include "CoverageLayer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "DemandHeat.h"
#include "accel/CoverageField.h"
#include "accel/DemandHeatmap.h"

std::pair<double, double> interpolateLatLon(const std::pair<double, double>& from, const std::pair<double, double>& to,
                                            double ratio) {
    const double clamped = std::clamp(ratio, 0.0, 1.0);
    return {from.first + (to.first - from.first) * clamped, from.second + (to.second - from.second) * clamped};
}

std::string leastCoveredZoneLine(const Bounds& bounds, int gridWidth, int gridHeight, const std::vector<float>& coverage,
                                 int zoneSide) {
    if (coverage.empty()) {
        return "[estadisticas] zona menos cubierta: sin datos";
    }
    const accel::CoverageZone zone = accel::leastCoveredZone(coverage, gridWidth, gridHeight, zoneSide);
    // Celda central de la zona (la del borde de la grilla puede ser mas chica: se recorta).
    const int cellX = std::min(gridWidth - 1, zone.zoneX * zoneSide + zoneSide / 2);
    const int cellY = std::min(gridHeight - 1, zone.zoneY * zoneSide + zoneSide / 2);
    const auto center = demandCellCenter(bounds, gridWidth, gridHeight, cellX, cellY);
    char line[200];
    std::snprintf(line, sizeof line, "[estadisticas] zona menos cubierta: celda (%d,%d) ~ %.5f,%.5f, valor %.3f", cellX,
                  cellY, center.first, center.second, static_cast<double>(zone.meanCoverage));
    return line;
}

CoverageFieldService::CoverageFieldService(const Config& cfg, const SimContext& ctx, const Fleet& fleet)
    : cfg_(cfg), ctx_(ctx), fleet_(fleet) {}

CoverageFieldService::~CoverageFieldService() {
    requestStop();
    join();
}

void CoverageFieldService::start() {
    thread_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!stopping_) {
            lock.unlock();
            refresh();
            lock.lock();
            // Espera bloqueada (sin sondeo) hasta el proximo refresco o hasta que pidan parar.
            wake_.wait_for(lock, std::chrono::milliseconds(kCoverageRefreshRealMs), [this] { return stopping_; });
        }
    });
}

void CoverageFieldService::requestStop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
}

void CoverageFieldService::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

CoverageSnapshot CoverageFieldService::latest() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return latest_;
}

void CoverageFieldService::refresh() {
    // 1) Posicion de cada courier activo (interpolada sobre su tramo actual), como la dibuja la ventana.
    const std::int64_t nowMs = ctx_.clock.nowMs();
    std::vector<std::pair<double, double>> latLonPoints;
    for (CourierActor* actor : fleet_.activeSnapshot()) {
        const Courier view = actor->snapshot();
        const NodeIndex fromNode = ctx_.graph.indexOf(view.fromNode);
        if (fromNode == kNoNode) continue;
        const std::pair<double, double> from{ctx_.graph.latOf(fromNode), ctx_.graph.lonOf(fromNode)};
        const NodeIndex toNode = ctx_.graph.indexOf(view.toNode);
        const std::pair<double, double> to = toNode == kNoNode ? from : std::make_pair(ctx_.graph.latOf(toNode), ctx_.graph.lonOf(toNode));
        latLonPoints.push_back(interpolateLatLon(from, to, positionRatio(view, nowMs)));
    }
    const std::vector<float> couriersPerCell =
        countPointsPerCell(cfg_.bounds, kCoverageGridSide, kCoverageGridSide, latLonPoints, 1);
    // 2) Campo: GPU por el GpuWorker con plazo real, CPU si falla o vence. Solo este hilo espera; la ventana lee latest().
    if (!cpuPool_) {
        cpuPool_ = std::make_unique<accel::ThreadPool>(std::max(1u, std::min(4u, std::thread::hardware_concurrency())));
    }
    accel::CoverageResult result;
    try {
        result = accel::computeCoverage(ctx_.gpu, *cpuPool_, couriersPerCell, kCoverageGridSide, kCoverageGridSide,
                                        kCoverageSweepCount, kCoverageGpuTimeoutRealMs);
    } catch (const std::exception&) {
        return;  // con estas constantes no hay argumentos invalidos; ante la duda se conserva el ultimo campo
    }
    // 3) Publicar.
    auto published = std::make_shared<const std::vector<float>>(std::move(result.coverage));
    const float maxValue = published->empty() ? 0.0f : *std::max_element(published->begin(), published->end());
    float scaleValue = maxValue;
    if (!published->empty()) {
        std::vector<float> sorted(*published);
        const auto percentile95 = sorted.begin() + static_cast<std::ptrdiff_t>((sorted.size() - 1) * 95 / 100);
        std::nth_element(sorted.begin(), percentile95, sorted.end());
        if (*percentile95 > 0.0f) scaleValue = *percentile95;
    }
    const std::lock_guard<std::mutex> guard(mutex_);
    latest_.coverage = std::move(published);
    latest_.maxValue = maxValue;
    latest_.scaleValue = scaleValue;
    latest_.usedGpu = result.usedGpu;
    ++latest_.version;
}

std::string CoverageFieldService::leastCoveredZone() const {
    const CoverageSnapshot snapshot = latest();
    static const std::vector<float> none;
    return leastCoveredZoneLine(cfg_.bounds, kCoverageGridSide, kCoverageGridSide,
                                snapshot.coverage ? *snapshot.coverage : none, kCoverageZoneSide);
}
