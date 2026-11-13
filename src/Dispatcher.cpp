#include "Dispatcher.h"

#include <algorithm>
#include <optional>
#include <unordered_set>

namespace {
// Piso del plazo de reintento: con quoteTimeoutMs = 0 no se debe girar sin dormir.
constexpr std::int64_t kMinRetryMs = 50;
constexpr std::int64_t kMaxWaitMs = 24LL * 3600 * 1000;  // evita now()+ms desbordado
}  // namespace

Dispatcher::Dispatcher(const Config& cfg, const SimContext& ctx,
                       const std::vector<std::unique_ptr<CourierActor>>& fleet)
    : ctx_(ctx),
      legacyFleet_(&fleet),
      quoteTimeoutMs_(cfg.quoteTimeoutMs),
      acceptTimeoutMs_(cfg.acceptTimeoutMs),
      agingMs_(cfg.agingMs),
      quote_(estimateDeliveryMs),
      router_(ctx.graph),
      pool_(ctx.graph, static_cast<unsigned>(cfg.couriers) +
                           static_cast<unsigned>(cfg.reinforcements.max)) {}

Dispatcher::Dispatcher(const Config& cfg, const SimContext& ctx, Fleet& fleet)
    : ctx_(ctx),
      fleet_(&fleet),
      quoteTimeoutMs_(cfg.quoteTimeoutMs),
      acceptTimeoutMs_(cfg.acceptTimeoutMs),
      agingMs_(cfg.agingMs),
      quote_(estimateDeliveryMs),
      router_(ctx.graph),
      pool_(ctx.graph, static_cast<unsigned>(cfg.couriers) +
                           static_cast<unsigned>(cfg.reinforcements.max)) {}

std::vector<CourierActor*> Dispatcher::activeCouriers() const {
    if (fleet_ != nullptr) {
        return fleet_->activeSnapshot();
    }
    std::vector<CourierActor*> actors;
    actors.reserve(legacyFleet_->size());
    for (const auto& actor : *legacyFleet_) {
        actors.push_back(actor.get());
    }
    return actors;
}

bool Dispatcher::destinoInalcanzable(const Order& order) {
    const NodeIndex pickup = ctx_.graph.indexOf(order.pickupNode);
    const NodeIndex dropoff = ctx_.graph.indexOf(order.dropoffNode);
    if (pickup == kNoNode || dropoff == kNoNode) {
        return true;
    }
    return !router_.shortestPath(pickup, dropoff).found;
}

void Dispatcher::run() {
    constexpr auto kFarFuture = std::chrono::hours(24);
    while (!stopped()) {
        std::uint64_t observed = ctx_.book.workVersion();
        sweepLateQuotes();
        ctx_.book.rejectExpired(acceptTimeoutMs_);
        spawnIfNeeded(ctx_.book.waitingCount());

        std::unordered_set<OrderId> attempted;
        bool retry = false;
        while (!stopped()) {
            // los pedidos que vencieron mientras se cotizaban otros no se eligen
            ctx_.book.rejectExpired(acceptTimeoutMs_);
            const std::optional<Order> order =
                ctx_.book.pickNextUnassigned(agingMs_, attempted);
            if (!order.has_value()) {
                break;
            }
            attempted.insert(order->id);
            const DispatchResult result = dispatchOne(*order);
            if (result == DispatchResult::noRoom) {
                break;
            }
            if (result == DispatchResult::retryLater) {
                retry = true;
            }
        }
        spawnIfNeeded(ctx_.book.waitingCount());

        auto deadline = std::chrono::steady_clock::now() + kFarFuture;
        const std::optional<std::int64_t> due =
            ctx_.book.nextDeadlineSimMs(acceptTimeoutMs_, agingMs_);
        if (due.has_value()) {
            deadline = std::min(deadline, ctx_.clock.timePointAt(*due));
        }
        if (retry) {
            // piso de reintento: con quoteTimeoutMs = 0 el reintento no debe girar
            const std::int64_t retryMs = std::clamp<std::int64_t>(quoteTimeoutMs_, kMinRetryMs,
                                                                  kMaxWaitMs);
            deadline = std::min(deadline, std::chrono::steady_clock::now() +
                                              std::chrono::milliseconds(retryMs));
        }
        if (!ctx_.book.waitForChange(observed, deadline)) {
            break;
        }
    }

    for (std::future<std::int64_t>& f : inFlight_) {
        if (f.valid()) {
            f.wait();
        }
    }
    inFlight_.clear();

    pool_.stop();
}

void Dispatcher::spawnIfNeeded(std::size_t waiting) {
    if (fleet_ == nullptr) {
        return;
    }
    bool spawned = false;
    while (fleet_->maybeSpawn(waiting)) {
        spawned = true;
    }
    if (spawned) {
        ctx_.book.notifyCapacityChanged();
    }
}

Dispatcher::DispatchResult Dispatcher::dispatchOne(const Order& order) {
    const OrderId orderId = order.id;

    if (destinoInalcanzable(order)) {
        ctx_.book.reject(orderId, RejectReason::unreachable);
        return DispatchResult::rejected;
    }

    std::vector<Candidate> candidates;
    for (CourierActor* actor : activeCouriers()) {
        const Courier view = actor->snapshot();
        if (view.state == CourierState::brokenDown || view.state == CourierState::onBreak ||
            !hasRoom(view)) {
            continue;
        }
        candidates.push_back(Candidate{
            actor, pool_.submit([this, view, order](Router& router) {
                return quote_(router, ctx_.graph, view, order);
            })});
    }

    if (candidates.empty()) {
        return DispatchResult::noRoom;
    }
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        requested_ += static_cast<std::int64_t>(candidates.size());
    }

    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(std::clamp<std::int64_t>(quoteTimeoutMs_, 0, kMaxWaitMs));

    std::vector<std::pair<std::int64_t, CourierActor*>> ranked;
    std::int64_t lateNow = 0;
    bool failed = false;

    for (Candidate& c : candidates) {
        if (c.estimate.wait_until(deadline) != std::future_status::ready) {
            inFlight_.push_back(std::move(c.estimate));
            ++lateNow;
            continue;
        }
        std::int64_t value = kUnreachable;
        try {
            value = c.estimate.get();
        } catch (...) {
            failed = true;
            continue;
        }
        if (value != kUnreachable) {
            ranked.emplace_back(value, c.actor);
        }
    }
    if (lateNow > 0) {
        const std::lock_guard<std::mutex> guard(mutex_);
        late_ += lateNow;
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& x, const auto& y) {
                  if (x.first != y.first) {
                      return x.first < y.first;
                  }
                  return x.second->id() < y.second->id();
              });

    // Si un courier rechaza la reserva (por ejemplo, justo empezo a descansar), se prueba
    // con el siguiente mejor.
    for (const auto& entry : ranked) {
        CourierActor& elegido = *entry.second;
        if (!elegido.reserve(order)) {
            continue;
        }
        // La cotizacion pudo tardar: el plazo se revalida bajo el candado del libro.
        if (ctx_.book.assignUnlessExpired(orderId, elegido.id(), acceptTimeoutMs_) !=
            OrderBook::AssignResult::assigned) {
            elegido.cancelReservation();
            return DispatchResult::rejected;
        }
        elegido.commitReservation(order);
        return DispatchResult::assigned;
    }

    if (lateNow > 0 || failed) {
        return DispatchResult::retryLater;
    }
    if (ranked.empty()) {
        // nadie puede llegar a este pedido (todo kUnreachable): no frena a los demas
        return DispatchResult::unquotable;
    }
    return DispatchResult::noRoom;
}

void Dispatcher::sweepLateQuotes() {
    inFlight_.erase(
        std::remove_if(inFlight_.begin(), inFlight_.end(),
                       [](std::future<std::int64_t>& f) {
                           return !f.valid() ||
                                  f.wait_for(std::chrono::seconds(0)) ==
                                      std::future_status::ready;
                       }),
        inFlight_.end());
}

void Dispatcher::requestStop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopRequested_ = true;
    }
    ctx_.book.stopWaiting();
}

bool Dispatcher::stopped() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return stopRequested_;
}

std::int64_t Dispatcher::quotesRequested() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return requested_;
}

std::int64_t Dispatcher::quotesLate() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return late_;
}
