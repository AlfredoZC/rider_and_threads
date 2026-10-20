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
// Avisa al salir del alcance, es decir YA SOLTADO el candado: si se notificara con el
// candado tomado, el hilo que despierta se encontraria con el candado ocupado y se
// volveria a dormir.
//
// Es un objeto y no tres llamadas sueltas porque admit() tiene tres caminos de salida.
// Asi el aviso ocurre una sola vez, salga por donde salga, y no se puede olvidar al
// agregar un cuarto camino. Es la misma idea de lock_guard aplicada a algo que no es un
// recurso.
struct NotifyOnExit {
    std::condition_variable& cv;
    const bool& should;   // referencia: lee el valor que tenga AL FINAL, no al construir
    ~NotifyOnExit() {
        if (should) {
            cv.notify_all();
        }
    }
};
}  // namespace

bool OrderBook::admit(Order order) {
    bool admitted = false;
    const NotifyOnExit notify{work_, admitted};   // se declara PRIMERO: destruye ULTIMO
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

    // Toda orden que entra se registra como creada, incluida la que se va a rechazar
    // enseguida por cola llena. El §6 exige que cada contador del resumen sea igual a
    // la cantidad de lineas que coinciden en el log, y el resumen la cuenta en created
    // porque la identidad created = delivered + rejected + pending lo obliga. Si el log
    // no tuviera su orderCreated, el resumen diria 86 y el log 68.
    //
    // Y ademas cuenta la historia verdadera: la orden existio, se le asigno un id y un
    // restaurante, y recien despues se rechazo.
    event.name = EventName::kOrderCreated;
    // El §6 lista restaurant solo en orderCreated, pickupStarted y orderPickedUp.
    event.restaurant = order.restaurantId;
    // write() devuelve la t que uso: se guarda como createdMs para que el log y la
    // orden cuenten exactamente la misma historia, no dos que difieren en unos ms.
    order.createdMs = log_.write(event);
    ++stats_.created;

    // Verificar y actuar, los dos adentro del mismo candado.
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
        return false;   // ya estaba en un estado final: no se rechaza dos veces
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
    // assigned -> created es la unica transicion "hacia atras" que permite canTransition,
    // y existe justamente para esto: una orden asignada que su courier no pudo tomar
    // quedaria colgada para siempre, porque de assigned no se sale solo.
    if (!setState(it->second, OrderState::created)) {
        return false;
    }
    it->second.assignedCourier = kNoCourier;
    if (countAsReassigned) {
        it->second.reassigned = true;
        ++stats_.reassigned;

        Event event;
        event.name = EventName::kOrderReassigned;
        event.order = id;
        log_.write(event);
    }
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

std::int64_t OrderBook::markRemainingAsPending() {
    const std::lock_guard<std::mutex> guard(mutex_);
    std::int64_t marked = 0;
    for (auto& entry : orders_) {
        // No hace falta preguntar si la orden esta viva: setState consulta
        // canTransition, que prohibe salir de un estado final. Las ya entregadas o
        // rechazadas no se tocan.
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
    // Solo puede traspasarla quien la tiene: sin este chequeo, un courier podria
    // quedarse con ordenes de otro.
    if (it == orders_.end() || it->second.assignedCourier != from) {
        return false;
    }
    if (isFinalState(it->second.state)) {
        return false;   // ya termino: no hay nada que traspasar
    }
    it->second.assignedCourier = to;
    it->second.reassigned = true;
    ++stats_.reassigned;

    Event event;
    event.name = EventName::kOrderReassigned;
    event.order = id;
    // El §6 define from y to como los ids de los COURIERS que entregan y reciben, con
    // el mismo formato que el campo courier del resto de los eventos.
    event.from = "c" + std::to_string(from);
    event.to = "c" + std::to_string(to);
    log_.write(event);
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
    return it->second;   // se copia adentro del candado
}
bool OrderBook::waitForUnassigned(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    // Espera con plazo y predicado: duerme sin consumir CPU, despierta cuando admit()
    // avisa, y el predicado la protege de los despertares espurios.
    //
    // El plazo NO es una encuesta disfrazada: hace falta porque una orden puede vencer su
    // acceptTimeoutMs sin que entre ninguna orden nueva, y alguien tiene que ir a
    // rechazarla. Sin ese requisito, el wait no llevaria plazo.
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
    // Durante la corrida son las vivas; despues de cerrar, las que quedaron en
    // estado pending. Sumar ambas da el valor correcto en los dos momentos, porque
    // al marcarlas una baja exactamente lo que la otra sube.
    copy.pending = static_cast<std::int64_t>(pending_) + markedPending_;
    copy.avgDeliveryMs =
        stats_.delivered > 0 ? deliverySumMs_ / stats_.delivered : 0;
    return copy;
}
