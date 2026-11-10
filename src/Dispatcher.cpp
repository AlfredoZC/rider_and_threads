#include "Dispatcher.h"

#include <algorithm>

namespace {
// Sin pedidos esperando no hay plazo que vencer: el dispatcher duerme hasta que lo avisen.
constexpr std::chrono::hours kNoDeadline{24};
}

Dispatcher::Dispatcher(const Config& cfg, const SimContext& ctx,
                       const std::vector<std::unique_ptr<CourierActor>>& fleet)
    : ctx_(ctx),
      legacyFleet_(&fleet),
      quoteTimeoutMs_(cfg.quoteTimeoutMs),
      acceptTimeoutMs_(cfg.acceptTimeoutMs),
      quote_(estimateDeliveryMs),
      router_(ctx.graph),
      pool_(ctx.graph, static_cast<unsigned>(cfg.couriers) +
                           static_cast<unsigned>(cfg.reinforcements.max)) {}

Dispatcher::Dispatcher(const Config& cfg, const SimContext& ctx, Fleet& fleet)
    : ctx_(ctx),
      fleet_(&fleet),
      quoteTimeoutMs_(cfg.quoteTimeoutMs),
      acceptTimeoutMs_(cfg.acceptTimeoutMs),
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
    std::uint64_t observed = ctx_.book.workVersion();
    while (!stopped()) {
        sweepLateQuotes();
        const std::vector<OrderId> waiting = ctx_.book.unassignedIds();
        if (fleet_ != nullptr) {
            while (fleet_->maybeSpawn(waiting.size())) {}
        }
        bool retry = false;
        for (const OrderId id : waiting) {
            if (stopped()) {
                break;
            }
            if (dispatchOne(id)) {
                retry = true;
            }
        }
        if (fleet_ != nullptr) {
            const std::size_t stillWaiting = ctx_.book.unassignedIds().size();
            bool spawned = false;
            while (fleet_->maybeSpawn(stillWaiting)) {
                spawned = true;
            }
            if (spawned) {
                continue;
            }
        }
        if (!ctx_.book.waitForChange(observed, nextWakeUp(retry))) {
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

std::chrono::milliseconds Dispatcher::nextWakeUp(bool retry) {
    std::chrono::milliseconds wait = kNoDeadline;
    const std::int64_t now = ctx_.clock.nowMs();
    for (const OrderId id : ctx_.book.unassignedIds()) {
        const auto order = ctx_.book.find(id);
        if (!order.has_value()) {
            continue;
        }
        const std::int64_t simLeft =
            std::max<std::int64_t>(order->createdMs + acceptTimeoutMs_ + 1 - now, 0);
        const auto real =
            std::chrono::ceil<std::chrono::milliseconds>(ctx_.clock.realDuration(simLeft));
        wait = std::min(wait, std::max(real, std::chrono::milliseconds(1)));
    }
    if (retry) {
        wait = std::min(wait, std::chrono::milliseconds(quoteTimeoutMs_));
    }
    return wait;
}

bool Dispatcher::dispatchOne(OrderId orderId) {
    const auto maybe = ctx_.book.find(orderId);
    if (!maybe.has_value() || maybe->state != OrderState::created) {
        return false;
    }
    const Order order = *maybe;

    if (destinoInalcanzable(order)) {
        ctx_.book.reject(orderId, RejectReason::unreachable);
        return false;
    }

    if (ctx_.clock.nowMs() - order.createdMs > acceptTimeoutMs_) {
        ctx_.book.reject(orderId, RejectReason::noCourier);
        return false;
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
        return false;
    }
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        requested_ += static_cast<std::int64_t>(candidates.size());
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(quoteTimeoutMs_);

    std::vector<std::pair<std::int64_t, CourierActor*>> ranked;
    std::int64_t lateNow = 0;
    bool failed = false;

    for (Candidate& c : candidates) {
        if (c.estimate.wait_until(deadline) != std::future_status::ready) {
            inFlight_.push_back(std::move(c.estimate));
            ++lateNow;
            continue;
        }
        try {
            const std::int64_t value = c.estimate.get();
            if (value != kUnreachable) {
                ranked.emplace_back(value, c.actor);
            }
        } catch (...) {
            failed = true;
        }
    }
    if (lateNow > 0) {
        const std::lock_guard<std::mutex> guard(mutex_);
        late_ += lateNow;
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second->id() < b.second->id();
    });

    // Si un courier rechaza la reserva (por ejemplo, justo empezo a descansar), se prueba
    // con el siguiente mejor.
    for (const auto& entry : ranked) {
        CourierActor& elegido = *entry.second;
        if (!elegido.reserve(order)) {
            continue;
        }
        if (!ctx_.book.assign(orderId, elegido.id())) {
            elegido.cancelReservation();
            return false;
        }
        elegido.commitReservation(order);
        return false;
    }
    return lateNow > 0 || failed;
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
