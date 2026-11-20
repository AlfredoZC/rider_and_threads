#include "OrderBook.h"

#include <algorithm>

OrderBook::OrderBook(int maxPending, const SimClock& clock, EventLog& log)
    : maxPending_(static_cast<std::size_t>(maxPending)), clock_(clock), log_(log) {}

bool OrderBook::setState(Order& order, OrderState to) {
    if (!canTransition(order.state, to)) {
        return false;
    }
    const bool wasAlive = !isFinalState(order.state);
    order.state = to;
    if (wasAlive && isFinalState(to)) {
        --pending_;
    }
    return true;
}
namespace {
struct NotifyOnExit {
    std::condition_variable& cv;
    const bool& should;
    ~NotifyOnExit() {
        if (should) {
            cv.notify_all();
        }
    }
};
}

bool OrderBook::admit(Order order) {
    bool admitted = false;
    const NotifyOnExit notify{work_, admitted};
    const std::lock_guard<std::mutex> guard(mutex_);

    if (orders_.find(order.id) != orders_.end()) {
        return false;
    }

    Event event;
    event.order = order.id;

    event.name = EventName::kOrderCreated;
    event.restaurant = order.restaurantId;
    event.urgent = order.urgent ? 1 : 0;
    order.createdMs = log_.write(event);
    ++stats_.created;
    if (order.urgent) {
        ++stats_.urgentCreated;
    }

    if (pending_ >= maxPending_) {
        order.state = OrderState::rejected;
        order.rejectReason = RejectReason::queueFull;

        Event rejected;
        rejected.name = EventName::kOrderRejected;
        rejected.order = order.id;
        rejected.reason = toString(RejectReason::queueFull);
        log_.write(rejected);

        orders_.emplace(order.id, std::move(order));
        ++stats_.rejected;
        return false;
    }

    order.state = OrderState::created;
    orders_.emplace(order.id, std::move(order));
    ++pending_;
    ++workVersion_;
    admitted = true;
    return true;
}

bool OrderBook::reject(OrderId id, RejectReason reason) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return false;
    }
    if (!setState(it->second, OrderState::rejected)) {
        return false;
    }
    it->second.rejectReason = reason;
    ++stats_.rejected;

    Event event;
    event.name = EventName::kOrderRejected;
    event.order = id;
    event.reason = toString(reason);
    log_.write(event);
    return true;
}
bool OrderBook::unassign(OrderId id, bool countAsReassigned) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return false;
    }
    if (!setState(it->second, OrderState::created)) {
        return false;
    }
    it->second.assignedCourier = kNoCourier;
    ++workVersion_;
    if (countAsReassigned) {
        it->second.reassigned = true;
        ++stats_.reassigned;

        Event event;
        event.name = EventName::kOrderReassigned;
        event.order = id;
        log_.write(event);
    }
    work_.notify_all();
    return true;
}

std::uint64_t OrderBook::workVersion() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return workVersion_;
}

bool OrderBook::waitForChange(std::uint64_t& observed,
                              std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    work_.wait_for(lock, timeout, [this, &observed] {
        return stopWaiting_ || workVersion_ != observed;
    });
    observed = workVersion_;
    return !stopWaiting_;
}

bool OrderBook::waitForChange(std::uint64_t& observed,
                              std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    work_.wait_until(lock, deadline, [this, &observed] {
        return stopWaiting_ || workVersion_ != observed;
    });
    observed = workVersion_;
    return !stopWaiting_;
}

namespace {
bool isUrgentNow(const Order& o, std::int64_t nowMs, std::int64_t agingMs) {
    return o.urgent || (agingMs > 0 && nowMs - o.createdMs >= agingMs);
}
}

std::optional<Order> OrderBook::pickNextUnassigned(
    std::int64_t agingMs, const std::unordered_set<OrderId>& skip) const {
    const std::lock_guard<std::mutex> guard(mutex_);
    const std::int64_t now = clock_.nowMs();
    const Order* best = nullptr;
    bool bestClass0 = false;
    for (const auto& entry : orders_) {
        const Order& o = entry.second;
        if (o.state != OrderState::created || skip.count(o.id) != 0) {
            continue;
        }
        const bool class0 = isUrgentNow(o, now, agingMs);
        bool better = false;
        if (best == nullptr) {
            better = true;
        } else if (class0 != bestClass0) {
            better = class0;
        } else if (o.createdMs != best->createdMs) {
            better = o.createdMs < best->createdMs;
        } else {
            better = o.id < best->id;
        }
        if (better) {
            best = &o;
            bestClass0 = class0;
        }
    }
    if (best == nullptr) {
        return std::nullopt;
    }
    return *best;
}

std::int64_t OrderBook::rejectExpired(std::int64_t acceptTimeoutMs) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const std::int64_t now = clock_.nowMs();
    std::int64_t rejected = 0;
    for (auto& entry : orders_) {
        Order& o = entry.second;
        if (o.state != OrderState::created || now - o.createdMs <= acceptTimeoutMs) {
            continue;
        }
        if (!setState(o, OrderState::rejected)) {
            continue;
        }
        o.rejectReason = RejectReason::noCourier;
        ++stats_.rejected;
        ++rejected;

        Event event;
        event.name = EventName::kOrderRejected;
        event.order = o.id;
        event.reason = toString(RejectReason::noCourier);
        log_.write(event);
    }
    return rejected;
}

std::optional<std::int64_t> OrderBook::nextDeadlineSimMs(std::int64_t acceptTimeoutMs,
                                                         std::int64_t agingMs) const {
    const std::lock_guard<std::mutex> guard(mutex_);
    const std::int64_t now = clock_.nowMs();
    std::optional<std::int64_t> best;
    const auto consider = [&best](std::int64_t t) {
        if (!best.has_value() || t < *best) {
            best = t;
        }
    };
    for (const auto& entry : orders_) {
        const Order& o = entry.second;
        if (o.state != OrderState::created) {
            continue;
        }
        consider(saturatingAddMs(saturatingAddMs(o.createdMs, acceptTimeoutMs), 1));
        if (agingMs > 0 && !o.urgent && now - o.createdMs < agingMs) {
            consider(saturatingAddMs(o.createdMs, agingMs));
        }
    }
    return best;
}

std::size_t OrderBook::waitingCount() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    std::size_t n = 0;
    for (const auto& entry : orders_) {
        if (entry.second.state == OrderState::created) {
            ++n;
        }
    }
    return n;
}

void OrderBook::notifyCapacityChanged() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        ++workVersion_;
    }
    work_.notify_all();
}

bool OrderBook::assign(OrderId id, CourierId courier) {
    const std::lock_guard<std::mutex> guard(mutex_);
    return assignLocked(id, courier);
}

bool OrderBook::assignLocked(OrderId id, CourierId courier) {
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return false;
    }
    if (!setState(it->second, OrderState::assigned)) {
        return false;
    }
    it->second.assignedCourier = courier;

    Event event;
    event.name = EventName::kOrderAssigned;
    event.order = id;
    event.courier = courier;
    it->second.assignedMs = log_.write(event);
    return true;
}

OrderBook::AssignResult OrderBook::assignUnlessExpired(OrderId id, CourierId courier,
                                                       std::int64_t acceptTimeoutMs) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return AssignResult::failed;
    }
    Order& o = it->second;
    if (o.state == OrderState::created && clock_.nowMs() - o.createdMs > acceptTimeoutMs) {
        if (!setState(o, OrderState::rejected)) {
            return AssignResult::failed;
        }
        o.rejectReason = RejectReason::noCourier;
        ++stats_.rejected;

        Event event;
        event.name = EventName::kOrderRejected;
        event.order = o.id;
        event.reason = toString(RejectReason::noCourier);
        log_.write(event);
        return AssignResult::expired;
    }
    return assignLocked(id, courier) ? AssignResult::assigned : AssignResult::failed;
}

bool OrderBook::pickedUp(OrderId id, CourierId courier) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end() || it->second.assignedCourier != courier) {
        return false;
    }
    if (!setState(it->second, OrderState::pickedUp)) {
        return false;
    }

    Event event;
    event.name = EventName::kOrderPickedUp;
    event.order = id;
    event.courier = courier;
    event.restaurant = it->second.restaurantId;
    it->second.pickedUpMs = log_.write(event);
    return true;
}

bool OrderBook::delivered(OrderId id, CourierId courier) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end() || it->second.assignedCourier != courier) {
        return false;
    }
    if (!setState(it->second, OrderState::delivered)) {
        return false;
    }

    Event event;
    event.name = EventName::kOrderDelivered;
    event.order = id;
    event.courier = courier;
    it->second.deliveredMs = log_.write(event);

    ++stats_.delivered;
    const std::int64_t total = deliveryMs(it->second);
    deliverySumMs_ += total;
    stats_.maxDeliveryMs = std::max(stats_.maxDeliveryMs, total);
    if (it->second.urgent) {
        ++stats_.urgentDelivered;
        urgentDeliverySumMs_ += total;
    }
    return true;
}

std::int64_t OrderBook::markRemainingAsPending() {
    const std::lock_guard<std::mutex> guard(mutex_);
    std::int64_t marked = 0;
    for (auto& entry : orders_) {
        if (setState(entry.second, OrderState::pending)) {
            ++marked;
            ++markedPending_;
        }
    }
    return marked;
}

bool OrderBook::handOver(OrderId id, CourierId from, CourierId to) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end() || it->second.assignedCourier != from) {
        return false;
    }
    if (isFinalState(it->second.state)) {
        return false;
    }
    it->second.assignedCourier = to;
    it->second.reassigned = true;
    ++stats_.reassigned;

    Event event;
    event.name = EventName::kOrderReassigned;
    event.order = id;
    event.from = courierLabel(from);
    event.to = courierLabel(to);
    log_.write(event);
    return true;
}

bool OrderBook::handOverBatch(const std::vector<OrderId>& ids, CourierId from,
                              CourierId to) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (ids.empty()) {
        return false;
    }
    for (const OrderId id : ids) {
        const auto it = orders_.find(id);
        if (it == orders_.end() || it->second.assignedCourier != from ||
            isFinalState(it->second.state)) {
            return false;
        }
    }

    Event breakdown;
    breakdown.name = EventName::kCourierBreakdown;
    breakdown.courier = from;
    log_.write(breakdown);
    ++stats_.breakdowns;
    for (const OrderId id : ids) {
        Order& order = orders_.at(id);
        order.assignedCourier = to;
        order.reassigned = true;
        ++stats_.reassigned;

        Event event;
        event.name = EventName::kOrderReassigned;
        event.order = id;
        event.from = courierLabel(from);
        event.to = courierLabel(to);
        log_.write(event);
    }
    return true;
}

void OrderBook::recordBreakdown(CourierId courier) {
    const std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.breakdowns;

    Event event;
    event.name = EventName::kCourierBreakdown;
    event.courier = courier;
    log_.write(event);
}

std::optional<Order> OrderBook::find(OrderId id) const {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return std::nullopt;
    }
    return it->second;
}
bool OrderBook::waitForUnassigned(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    work_.wait_for(lock, timeout, [this] {
        if (stopWaiting_) {
            return true;
        }
        for (const auto& entry : orders_) {
            if (entry.second.state == OrderState::created) {
                return true;
            }
        }
        return false;
    });
    return !stopWaiting_;
}

void OrderBook::stopWaiting() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopWaiting_ = true;
    }
    work_.notify_all();
}

std::vector<OrderId> OrderBook::unassignedIds() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    std::vector<OrderId> ids;
    for (const auto& entry : orders_) {
        if (entry.second.state == OrderState::created) {
            ids.push_back(entry.first);
        }
    }
    return ids;
}

OrderStats OrderBook::stats() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    OrderStats copy = stats_;
    copy.pending = static_cast<std::int64_t>(pending_) + markedPending_;
    copy.avgDeliveryMs =
        stats_.delivered > 0 ? deliverySumMs_ / stats_.delivered : 0;
    copy.avgUrgentDeliveryMs =
        stats_.urgentDelivered > 0 ? urgentDeliverySumMs_ / stats_.urgentDelivered : 0;
    return copy;
}
