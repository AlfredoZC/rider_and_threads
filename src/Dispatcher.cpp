#include "Dispatcher.h"

#include <algorithm>
#include <limits>

namespace {
constexpr std::chrono::milliseconds kIdleCheck{200};
}

Dispatcher::Dispatcher(const Config& cfg, const SimContext& ctx,
                       const std::vector<std::unique_ptr<CourierActor>>& fleet)
    : ctx_(ctx),
      fleet_(fleet),
      quoteTimeoutMs_(cfg.quoteTimeoutMs),
      acceptTimeoutMs_(cfg.acceptTimeoutMs),
      quote_(estimateDeliveryMs),
      router_(ctx.graph),
      pool_(ctx.graph) {}

bool Dispatcher::destinoInalcanzable(const Order& order) {
    const NodeIndex pickup = ctx_.graph.indexOf(order.pickupNode);
    const NodeIndex dropoff = ctx_.graph.indexOf(order.dropoffNode);
    if (pickup == kNoNode || dropoff == kNoNode) {
        return true;
    }
    return !router_.shortestPath(pickup, dropoff).found;
}

void Dispatcher::run() {
    while (!stopped()) {
        if (!ctx_.book.waitForUnassigned(kIdleCheck)) {
            break;
        }
        sweepLateQuotes();
        for (const OrderId id : ctx_.book.unassignedIds()) {
            if (stopped()) {
                break;
            }
            dispatchOne(id);
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

void Dispatcher::dispatchOne(OrderId orderId) {
    const auto maybe = ctx_.book.find(orderId);
    if (!maybe.has_value() || maybe->state != OrderState::created) {
        return;
    }
    const Order order = *maybe;

    if (destinoInalcanzable(order)) {
        ctx_.book.reject(orderId, RejectReason::unreachable);
        return;
    }

    if (ctx_.clock.nowMs() - order.createdMs > acceptTimeoutMs_) {
        ctx_.book.reject(orderId, RejectReason::noCourier);
        return;
    }

    std::vector<Candidate> candidates;
    for (std::size_t i = 0; i < fleet_.size(); ++i) {
        const Courier view = fleet_[i]->snapshot();
        if (view.state == CourierState::brokenDown || !hasRoom(view)) {
            continue;
        }
        candidates.push_back(Candidate{
            i, pool_.submit([this, view, order](Router& router) {
                return quote_(router, ctx_.graph, view, order);
            })});
    }

    if (candidates.empty()) {
        return;
    }
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        requested_ += static_cast<std::int64_t>(candidates.size());
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(quoteTimeoutMs_);

    std::size_t best = 0;
    std::int64_t bestMs = std::numeric_limits<std::int64_t>::max();
    bool found = false;
    std::int64_t lateNow = 0;

    for (Candidate& c : candidates) {
        if (c.estimate.wait_until(deadline) != std::future_status::ready) {
            inFlight_.push_back(std::move(c.estimate));
            ++lateNow;
            continue;
        }
        std::int64_t value = kUnreachable;
        try {
            value = c.estimate.get();
        } catch (const std::exception&) {
            continue;
        } catch (...) {
            continue;
        }
        if (value != kUnreachable && value < bestMs) {
            bestMs = value;
            best = c.index;
            found = true;
        }
    }
    if (lateNow > 0) {
        const std::lock_guard<std::mutex> guard(mutex_);
        late_ += lateNow;
    }

    if (!found) {
        return;
    }

    CourierActor& elegido = *fleet_[best];
    if (!elegido.reserve(order)) {
        return;
    }
    if (!ctx_.book.assign(orderId, elegido.id())) {
        elegido.cancelReservation();
        return;
    }
    elegido.commitReservation(order);
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
