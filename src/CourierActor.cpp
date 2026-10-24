#include "CourierActor.h"

#include "Fleet.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {
constexpr std::int64_t kPickupServiceMs = 1000;
}

CourierActor::CourierActor(CourierId id, const std::string& startNode, int bagCapacity,
                           const SimContext& ctx, double breakdownProbability,
                           std::uint32_t seed, bool reinforcement, std::int64_t idleMs)
    : id_(id),
      ctx_(ctx),
      breakdownProbability_(breakdownProbability),
      reinforcement_(reinforcement),
      idleMs_(idleMs),
      rng_(seed ^ (static_cast<std::uint32_t>(id) * 2654435761u)),
      state_(id, startNode, bagCapacity) {}

void CourierActor::run() {
    Router router(ctx_.graph);

    std::unique_lock<std::mutex> lock(mutex_);
    std::chrono::steady_clock::time_point idleSince;
    bool countingIdle = false;
    while (true) {
        if (stopRequested_) {
            break;
        }
        if (state_.bag.empty()) {
            if (!reinforcement_) {
                cv_.wait(lock, [this] { return stopRequested_ || !state_.bag.empty(); });
                continue;
            }
            if (!countingIdle) {
                idleSince = std::chrono::steady_clock::now();
                countingIdle = true;
            }
            const auto deadline = idleSince + ctx_.clock.realDuration(idleMs_);
            cv_.wait_until(lock, deadline, [this] {
                return stopRequested_ || !state_.bag.empty() || reserved_ > 0;
            });
            if (stopRequested_) {
                break;
            }
            if (!state_.bag.empty()) {
                countingIdle = false;
                continue;
            }
            if (reserved_ > 0) {
                cv_.wait(lock, [this] {
                    return stopRequested_ || !state_.bag.empty() || reserved_ == 0;
                });
                continue;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                retired_ = true;
                lock.unlock();
                if (fleetManager_ != nullptr) {
                    fleetManager_->markRetired(id_);
                }
                return;
            }
            continue;
        }

        countingIdle = false;

        const OrderId orderId = state_.bag.front();

        lock.unlock();
        if (!maybeBreakDown()) {
            deliver(router, orderId);
        }
        lock.lock();
    }
}

void CourierActor::deliver(Router& router, OrderId orderId) {
    const auto maybe = ctx_.book.find(orderId);
    if (!maybe.has_value()) {
        finish(orderId, false);
        return;
    }
    const Order order = *maybe;

    setCourierState(CourierState::toPickup);
    const Trip toPickup = travelTo(router, order.pickupNode);
    if (toPickup == Trip::stopped) {
        return;
    }
    if (toPickup == Trip::unreachable) {
        ctx_.book.reject(orderId, RejectReason::unreachable);
        finish(orderId, false);
        return;
    }

    setCourierState(CourierState::atRestaurant);

    const std::int64_t arrivedMs = ctx_.clock.nowMs();
    const std::int64_t readyMs = order.createdMs + order.prepMs;
    if (readyMs > arrivedMs && !waitSim(readyMs - arrivedMs)) {
        return;
    }

    const auto entry = ctx_.restaurants.find(order.restaurantId);
    Restaurant* const restaurant =
        entry == ctx_.restaurants.end() ? nullptr : entry->second.get();
    if (restaurant != nullptr && !restaurant->acquire()) {
        return;
    }

    Event started;
    started.name = EventName::kPickupStarted;
    started.order = orderId;
    started.courier = id_;
    started.restaurant = order.restaurantId;
    ctx_.log.write(started);

    if (!waitSim(kPickupServiceMs)) {
        if (restaurant != nullptr) {
            restaurant->release();
        }
        return;
    }
    ctx_.book.pickedUp(orderId, id_);
    if (restaurant != nullptr) {
        restaurant->release();
    }

    if (maybeBreakDown()) {
        return;
    }

    setCourierState(CourierState::toDropoff);
    const Trip toDropoff = travelTo(router, order.dropoffNode);
    if (toDropoff == Trip::stopped) {
        return;
    }
    if (toDropoff == Trip::unreachable) {
        finish(orderId, false);
        return;
    }

    ctx_.book.delivered(orderId, id_);
    finish(orderId, true);
}

CourierActor::Trip CourierActor::travelTo(Router& router, const std::string& destNode) {
    const NodeIndex from = ctx_.graph.indexOf(currentNode(snapshot()));
    const NodeIndex to = ctx_.graph.indexOf(destNode);
    if (from == kNoNode || to == kNoNode) {
        return Trip::unreachable;
    }
    const Route route = router.shortestPath(from, to);
    if (!route.found) {
        return Trip::unreachable;
    }

    for (std::size_t i = 0; i + 1 < route.nodes.size(); ++i) {
        std::int64_t segmentMs = 0;
        for (const Edge& edge : ctx_.graph.edgesFrom(route.nodes[i])) {
            if (edge.to == route.nodes[i + 1]) {
                segmentMs = edge.travelMs;
                break;
            }
        }

        {
            const std::lock_guard<std::mutex> guard(mutex_);
            state_.fromNode = ctx_.graph.idOf(route.nodes[i]);
            state_.toNode = ctx_.graph.idOf(route.nodes[i + 1]);
            state_.segmentStartMs = ctx_.clock.nowMs();
            state_.segmentEndMs = state_.segmentStartMs + segmentMs;
        }

        if (!waitSim(segmentMs)) {
            return Trip::stopped;
        }

        {
            const std::lock_guard<std::mutex> guard(mutex_);
            state_.fromNode = state_.toNode;
            state_.segmentStartMs = state_.segmentEndMs;
        }
    }
    return Trip::arrived;
}

bool CourierActor::waitSim(std::int64_t simMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    return !cv_.wait_for(lock, ctx_.clock.realDuration(simMs),
                         [this] { return stopRequested_; });
}

bool CourierActor::offer(const Order& order) {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (stopRequested_ || retired_ || state_.state == CourierState::brokenDown ||
            !hasRoomLocked()) {
            return false;
        }
        state_.bag.push_back(order.id);
    }
    cv_.notify_one();
    return true;
}

bool CourierActor::hasRoomLocked() const {
    return state_.bag.size() + static_cast<std::size_t>(reserved_) <
           static_cast<std::size_t>(state_.bagCapacity);
}

bool CourierActor::reserve(const Order&) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (stopRequested_ || retired_ || state_.state == CourierState::brokenDown ||
        !hasRoomLocked()) {
        return false;
    }
    ++reserved_;
    cv_.notify_one();
    return true;
}

void CourierActor::commitReservation(const Order& order) {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        --reserved_;
        state_.bag.push_back(order.id);
    }
    cv_.notify_one();
}

void CourierActor::cancelReservation() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        --reserved_;
    }
    cv_.notify_one();
    ctx_.book.notifyCapacityChanged();
}

void CourierActor::setCourierState(CourierState s) {
    const std::lock_guard<std::mutex> guard(mutex_);
    state_.state = s;
}

void CourierActor::finish(OrderId orderId, bool wasDelivered) {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        state_.bag.erase(std::remove(state_.bag.begin(), state_.bag.end(), orderId),
                         state_.bag.end());
        if (wasDelivered) {
            ++state_.delivered;
        }
        if (state_.bag.empty()) {
            state_.state = CourierState::idle;
        }
    }
    ctx_.book.notifyCapacityChanged();
}

bool CourierActor::maybeBreakDown() {
    const double probability = 1.0 - std::sqrt(1.0 - breakdownProbability_);
    if (probability <= 0.0) {
        return false;
    }
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    if (chance(rng_) >= probability) {
        return false;
    }
    return breakDownAndHandOver();
}

bool CourierActor::forceBreakdown() { return breakDownAndHandOver(); }

bool CourierActor::breakDownAndHandOver() {
    if (fleet_ == nullptr && fleetManager_ == nullptr) {
        return false;
    }

    std::vector<CourierActor*> candidates;
    if (fleetManager_ != nullptr) {
        candidates = fleetManager_->activeSnapshot();
    } else {
        candidates.reserve(fleet_->size());
        for (const auto& actor : *fleet_) {
            candidates.push_back(actor.get());
        }
    }

    CourierActor* recipient = nullptr;
    for (CourierActor* candidate : candidates) {
        if (candidate->id() == id_) {
            continue;
        }
        if (!tryHandOverTo(*candidate).empty()) {
            recipient = candidate;
            break;
        }
    }
    if (recipient == nullptr) {
        return false;
    }
    recipient->cv_.notify_one();
    return true;
}

std::vector<OrderId> CourierActor::tryHandOverTo(CourierActor& target) {
    const std::scoped_lock guard(mutex_, target.mutex_);

    if (state_.bag.empty() || state_.state == CourierState::brokenDown) {
        return {};
    }
    if (reserved_ > 0) {
        return {};
    }
    if (target.stopRequested_ || target.retired_ ||
        target.state_.state == CourierState::brokenDown) {
        return {};
    }
    const std::size_t ocupados =
        target.state_.bag.size() + static_cast<std::size_t>(target.reserved_);
    if (ocupados >= static_cast<std::size_t>(target.state_.bagCapacity)) {
        return {};
    }
    const std::size_t libre =
        static_cast<std::size_t>(target.state_.bagCapacity) - ocupados;
    if (state_.bag.size() > libre) {
        return {};
    }

    std::vector<OrderId> moved = state_.bag;
    if (!ctx_.book.handOverBatch(moved, id_, target.id_)) {
        return {};
    }
    for (const OrderId orderId : moved) {
        target.state_.bag.push_back(orderId);
    }
    state_.bag.clear();
    state_.state = CourierState::brokenDown;
    return moved;
}

void CourierActor::requestStop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopRequested_ = true;
    }
    cv_.notify_all();
}

Courier CourierActor::snapshot() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return state_;
}
