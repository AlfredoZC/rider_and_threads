#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <vector>

#include "Config.h"
#include "CourierActor.h"
#include "Fleet.h"
#include "accel/CandidateFilter.h"
#include "Quote.h"
#include "QuotePool.h"
#include "SimContext.h"

class Dispatcher {
public:
    using QuoteFn = std::function<std::int64_t(Router&, const CityGraph&, const Courier&,
                                               const Order&)>;

    Dispatcher(const Config& cfg, const SimContext& ctx,
               const std::vector<std::unique_ptr<CourierActor>>& fleet);
    Dispatcher(const Config& cfg, const SimContext& ctx, Fleet& fleet);

    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;

    void run();
    void requestStop();

    void setQuoteFunction(QuoteFn fn) { quote_ = std::move(fn); }

    // Radio del filtro de candidatos (aplicacion 10): antes de cotizar un pedido se cotiza primero a los
    // couriers que llegan al restaurante en a lo sumo este tiempo segun la tabla de tiempos; 0 lo apaga.
    // Por defecto 10 minutos: a la velocidad de los couriers cubre de sobra la zona de un restaurante, asi
    // que el filtro solo deja de lado a los que estan muy lejos.
    static constexpr std::int64_t kDefaultCandidateRadiusMs = 10LL * 60 * 1000;
    void setCandidateRadiusMs(std::int64_t radiusMs) { candidateRadiusMs_ = radiusMs; }
    std::int64_t candidateRadiusMs() const { return candidateRadiusMs_; }
    const accel::CandidateFilter& candidateFilter() const { return filter_; }

    std::int64_t quotesRequested() const;
    std::size_t quoteWorkers() const { return pool_.workers(); }
    std::int64_t quotesLate() const;

private:
    struct Candidate {
        CourierActor* actor;
        std::future<std::int64_t> estimate;
    };

    enum class DispatchResult { assigned, rejected, noRoom, retryLater, unquotable };
    struct FleetMember {
        CourierActor* actor;
        Courier view;
    };
    DispatchResult dispatchOne(const Order& order);
    DispatchResult quoteRound(const Order& order, const std::vector<FleetMember>& members,
                             std::chrono::steady_clock::time_point deadline);
    void splitByRadius(const Order& order, const std::vector<FleetMember>& available,
                       std::vector<FleetMember>& nearby, std::vector<FleetMember>& others,
                       std::chrono::steady_clock::time_point deadline);
    void spawnIfNeeded(std::size_t waiting);
    void sweepLateQuotes();
    bool stopped() const;

    bool destinoInalcanzable(const Order& order);
    std::vector<CourierActor*> activeCouriers() const;

    const SimContext ctx_;
    const std::vector<std::unique_ptr<CourierActor>>* legacyFleet_ = nullptr;
    Fleet* fleet_ = nullptr;
    const std::int64_t quoteTimeoutMs_;
    const std::int64_t acceptTimeoutMs_;
    const std::int64_t agingMs_;
    QuoteFn quote_;

    Router router_;

    QuotePool pool_;

    std::vector<std::future<std::int64_t>> inFlight_;

    mutable std::mutex mutex_;
    bool stopRequested_ = false;
    std::int64_t requested_ = 0;
    std::int64_t late_ = 0;
    std::int64_t candidateRadiusMs_ = kDefaultCandidateRadiusMs;
    accel::CandidateFilter filter_;
};
