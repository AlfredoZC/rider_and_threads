#include "OrderBook.h"

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

bool OrderBook::admit(Order order) {
    const std::lock_guard<std::mutex> guard(mutex_);

    Event event;
    event.order = order.id;
    event.restaurant = order.restaurantId;
    event.from = order.pickupNode;
    event.to = order.dropoffNode;

    // Verificar y actuar, los dos adentro del mismo candado.
    if (pending_ >= maxPending_) {
        order.state = OrderState::rejected;
        order.rejectReason = RejectReason::queueFull;
        event.name = EventName::kOrderRejected;
        event.reason = toString(RejectReason::queueFull);
        order.createdMs = log_.write(event);
        orders_.emplace(order.id, std::move(order));
        ++stats_.created;
        ++stats_.rejected;
        return false;
    }

    event.name = EventName::kOrderCreated;
    // write() devuelve la t que uso: se guarda como createdMs para que el log y la
    // orden cuenten exactamente la misma historia, no dos que difieren en unos ms.
    order.createdMs = log_.write(event);
    order.state = OrderState::created;
    orders_.emplace(order.id, std::move(order));
    ++pending_;
    ++stats_.created;
    return true;
}

bool OrderBook::reject(OrderId id, RejectReason reason) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return false;
    }
    if (!setState(it->second, OrderState::rejected)) {
        return false;   // ya estaba en un estado final: no se rechaza dos veces
    }
    it->second.rejectReason = reason;
    ++stats_.rejected;

    Event event;
    event.name = EventName::kOrderRejected;
    event.order = id;
    event.restaurant = it->second.restaurantId;
    event.reason = toString(reason);
    log_.write(event);
    return true;
}

std::optional<Order> OrderBook::find(OrderId id) const {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return std::nullopt;
    }
    return it->second;   // se copia adentro del candado
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
    copy.pending = static_cast<std::int64_t>(pending_);
    copy.avgDeliveryMs =
        stats_.delivered > 0 ? deliverySumMs_ / stats_.delivered : 0;
    return copy;
}
