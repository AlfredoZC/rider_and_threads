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

bool OrderBook::admit(Order order) {
    const std::lock_guard<std::mutex> guard(mutex_);

    // Un id repetido se rechaza antes de tocar nada.
    //
    // Sin esta guarda el defecto era silencioso: emplace() no hace nada si la clave ya
    // existe —no falla ni avisa, devuelve .second == false y sigue—, pero el resto del
    // metodo igual escribia el evento y hacia ++stats_.created. Quedaba una orden
    // contada como creada que no estaba en el libro, el invariante
    // created = delivered + rejected + pending dejaba de cerrar, y el sintoma aparecia
    // recien en el resumen final (§6), a mil lineas de la causa.
    //
    // Hoy no puede ocurrir porque OrderFactory garantiza ids unicos, pero esa garantia
    // vive en otro archivo: el libro la verifica donde la usa, igual que el
    // assert(timeScale > 0) del SimClock. No se aborta el programa porque de esto se
    // sale limpio: no se registra nada, no se cuenta nada y el llamador se entera por
    // el false.
    if (orders_.find(order.id) != orders_.end()) {
        return false;
    }

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

bool OrderBook::assign(OrderId id, CourierId courier) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    if (it == orders_.end()) {
        return false;
    }
    // La orden pudo cambiar desde que el dispatcher la miro: pudo vencer su plazo o
    // haberla tomado otro. Por eso se vuelve a validar ACA, dentro del candado. Un dato
    // leido afuera del candado es una foto del pasado: sirve para decidir, no para
    // suponer.
    if (!setState(it->second, OrderState::assigned)) {
        return false;
    }
    it->second.assignedCourier = courier;

    Event event;
    event.name = EventName::kOrderAssigned;
    event.order = id;
    event.courier = courier;
    event.restaurant = it->second.restaurantId;
    it->second.assignedMs = log_.write(event);
    return true;
}

bool OrderBook::pickedUp(OrderId id, CourierId courier) {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = orders_.find(id);
    // Solo el courier que la tiene asignada puede retirarla: sin este chequeo, el log
    // podria decir que la retiro alguien que nunca la tuvo.
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
    event.to = it->second.dropoffNode;
    it->second.deliveredMs = log_.write(event);

    // Los numeros del resumen (§6) se acumulan en el MISMO candado que el cambio de
    // estado. Si se actualizaran por separado, habria un instante en que la orden esta
    // entregada y el resumen todavia no la cuenta, y el §6 exige que coincidan siempre.
    ++stats_.delivered;
    const std::int64_t total = deliveryMs(it->second);
    deliverySumMs_ += total;
    stats_.maxDeliveryMs = std::max(stats_.maxDeliveryMs, total);
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
