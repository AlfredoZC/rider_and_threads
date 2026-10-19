#include "CourierActor.h"

#include <algorithm>

CourierActor::CourierActor(CourierId id, const std::string& startNode, int bagCapacity,
                           const SimContext& ctx)
    : id_(id), ctx_(ctx), state_(id, startNode, bagCapacity) {}

void CourierActor::run() {
    // La memoria de trabajo de Dijkstra es de ESTE hilo. Se crea una vez, aca, y se
    // reutiliza en todos los trayectos: asi ninguna ruta pide memoria al sistema.
    Router router(ctx_.graph);

    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        // wait() libera el candado y duerme de forma atomica: 0% de CPU mientras no hay
        // trabajo, y el renderer puede llamar a snapshot() mientras tanto. El predicado
        // cubre los despertares espurios.
        cv_.wait(lock, [this] { return stopRequested_ || !state_.bag.empty(); });
        if (stopRequested_) {
            break;
        }

        const OrderId orderId = state_.bag.front();

        // El reparto entero ocurre SIN el candado tomado. Dura segundos; sostenerlo
        // congelaria la ventana y bloquearia al dispatcher.
        lock.unlock();
        deliver(router, orderId);
        lock.lock();
    }
}

void CourierActor::deliver(Router& router, OrderId orderId) {
    const auto maybe = ctx_.book.find(orderId);
    if (!maybe.has_value()) {
        finish(orderId, false);   // desaparecio del libro: no es nuestra
        return;
    }
    const Order order = *maybe;   // copia propia: el libro puede cambiar mientras viajamos

    setCourierState(CourierState::toPickup);
    const Trip toPickup = travelTo(router, order.pickupNode);
    if (toPickup == Trip::stopped) {
        return;   // la orden queda viva; al cerrar se marcara como pendiente (§7)
    }
    if (toPickup == Trip::unreachable) {
        ctx_.book.reject(orderId, RejectReason::unreachable);
        finish(orderId, false);
        return;
    }

    // Llego al restaurante. pickupStarted se registra directo en el log porque no es un
    // cambio de estado de la orden: es un instante del recorrido del courier.
    setCourierState(CourierState::atRestaurant);
    Event started;
    started.name = EventName::kPickupStarted;
    started.order = orderId;
    started.courier = id_;
    started.restaurant = order.restaurantId;
    const std::int64_t arrivedMs = ctx_.log.write(started);

    // Si llego antes de que la comida estuviera lista, espera lo que falte. La
    // preparacion se cuenta desde que la orden se creo.
    const std::int64_t readyMs = order.createdMs + order.prepMs;
    if (readyMs > arrivedMs && !waitSim(readyMs - arrivedMs)) {
        return;
    }

    ctx_.book.pickedUp(orderId, id_);

    setCourierState(CourierState::toDropoff);
    const Trip toDropoff = travelTo(router, order.dropoffNode);
    if (toDropoff == Trip::stopped) {
        return;
    }
    if (toDropoff == Trip::unreachable) {
        finish(orderId, false);   // ya la retiro: no se puede rechazar, queda pendiente
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

    // Se recorre arista por arista. El courier nunca "salta" a destino: pasa por cada
    // nodo del camino, que es lo que impide que cruce una manzana (§3).
    for (std::size_t i = 0; i + 1 < route.nodes.size(); ++i) {
        std::int64_t segmentMs = 0;
        for (const Edge& edge : ctx_.graph.edgesFrom(route.nodes[i])) {
            if (edge.to == route.nodes[i + 1]) {
                segmentMs = edge.travelMs;
                break;
            }
        }

        {
            // Anotar el tramo es lo unico que necesita el candado, y son cuatro
            // asignaciones. El dibujo deduce la posicion de estas marcas con el reloj,
            // asi que el courier no tiene que despertarse a actualizar nada.
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
            state_.fromNode = state_.toNode;              // llego: queda quieto en el nodo
            state_.segmentStartMs = state_.segmentEndMs;  // start == end => ratio 1
        }
    }
    return Trip::arrived;
}

bool CourierActor::waitSim(std::int64_t simMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    // Esperar con plazo, no dormir. Un hilo dormido no se entera de la señal de parada, y
    // un tramo puede durar segundos reales: el §7 da 5 para cerrar todo. wait_for devuelve
    // el valor del predicado, asi que true significa "nos pidieron parar".
    return !cv_.wait_for(lock, ctx_.clock.realDuration(simMs),
                         [this] { return stopRequested_; });
}

bool CourierActor::offer(const Order& order) {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        // Verificar y agregar, los dos bajo el mismo candado: si el dispatcher pudiera
        // consultar primero y ofrecer despues, dos ofertas simultaneas se pasarian de
        // bagCapacity. Es el mismo check-then-act del maxPending del OrderBook.
        if (stopRequested_ || state_.state == CourierState::brokenDown ||
            !hasRoom(state_)) {
            return false;
        }
        state_.bag.push_back(order.id);
    }
    cv_.notify_one();   // fuera del candado: si no, despertaria para volver a dormirse
    return true;
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

void CourierActor::requestStop() {
    {
        // La bandera se escribe CON el candado tomado: si no, el hilo podria evaluar el
        // predicado como falso y, antes de dormirse, perderse el aviso y quedar dormido
        // para siempre.
        const std::lock_guard<std::mutex> guard(mutex_);
        stopRequested_ = true;
    }
    cv_.notify_all();   // puede estar esperando en el wait del bucle o en un waitSim
}

Courier CourierActor::snapshot() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return state_;   // se copia adentro del candado y se devuelve afuera
}
