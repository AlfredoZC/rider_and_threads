#include "CourierActor.h"

#include <algorithm>

CourierActor::CourierActor(CourierId id, const std::string& startNode, int bagCapacity,
                           const SimContext& ctx, double breakdownProbability,
                           std::uint32_t seed)
    : id_(id),
      ctx_(ctx),
      breakdownProbability_(breakdownProbability),
      rng_(seed ^ (static_cast<std::uint32_t>(id) * 2654435761u)),
      state_(id, startNode, bagCapacity) {}

void CourierActor::run() {
    Router router(ctx_.graph);

    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        cv_.wait(lock, [this] { return stopRequested_ || !state_.bag.empty(); });
        if (stopRequested_) {
            break;
        }

        const OrderId orderId = state_.bag.front();

        lock.unlock();
        if (!maybeBreakDown(0.5)) {
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

    Event started;
    started.name = EventName::kPickupStarted;
    started.order = orderId;
    started.courier = id_;
    started.restaurant = order.restaurantId;
    const std::int64_t arrivedMs = ctx_.log.write(started);

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

    ctx_.book.pickedUp(orderId, id_);
    if (restaurant != nullptr) {
        restaurant->release();
    }

    if (maybeBreakDown(0.5)) {
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
        if (stopRequested_ || state_.state == CourierState::brokenDown ||
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
    if (stopRequested_ || state_.state == CourierState::brokenDown || !hasRoomLocked()) {
        return false;
    }
    ++reserved_;
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
    const std::lock_guard<std::mutex> guard(mutex_);
    --reserved_;
}

void CourierActor::setCourierState(CourierState s) {
    const std::lock_guard<std::mutex> guard(mutex_);
    state_.state = s;
}

void CourierActor::finish(OrderId orderId, bool wasDelivered) {
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

bool CourierActor::maybeBreakDown(double fraction) {
    const double probability = breakdownProbability_ * fraction;
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
    if (fleet_ == nullptr) {
        return false;
    }

    std::vector<OrderId> moved;
    CourierId receiver = kNoCourier;
    for (const auto& candidate : *fleet_) {
        if (candidate->id() == id_) {
            continue;
        }
        moved = tryHandOverTo(*candidate);
        if (!moved.empty()) {
            receiver = candidate->id();
            break;
        }
    }
    if (receiver == kNoCourier) {
        return false;
    }

    ctx_.book.recordBreakdown(id_);
    for (const OrderId orderId : moved) {
        ctx_.book.handOver(orderId, id_, receiver);
    }
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
    if (target.stopRequested_ || target.state_.state == CourierState::brokenDown) {
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
