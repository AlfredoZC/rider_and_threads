#include "CourierActor.h"

#include <algorithm>

CourierActor::CourierActor(CourierId id, const std::string& startNode, int bagCapacity,
                           const SimContext& ctx, double breakdownProbability,
                           std::uint32_t seed)
    : id_(id),
      ctx_(ctx),
      breakdownProbability_(breakdownProbability),
      // Semilla derivada del id con una constante de mezcla: cada courier tiene su
      // propia corriente de azar y la corrida sigue siendo reproducible.
      rng_(seed ^ (static_cast<std::uint32_t>(id) * 2654435761u)),
      state_(id, startNode, bagCapacity) {}

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
        // La averia se sortea ANTES de salir a repartir. Si ocurre, el courier queda
        // sin ordenes, y la vuelta siguiente lo encuentra esperando para siempre:
        // esta fuera de servicio.
        if (!maybeBreakDown(0.5)) {
            deliver(router, orderId);
        }
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

    // PRIMERO se espera a que la comida este lista, y recien DESPUES se hace fila por un
    // lugar de retiro.
    //
    // Al reves -que era como estaba- el courier ocupaba un lugar del restaurante para
    // quedarse parado esperando una preparacion que puede durar minutos, mientras otro
    // con la comida ya lista hacia fila afuera. Un recurso escaso retenido sin usarlo es
    // el error clasico al administrar recursos compartidos: no alcanza con tomarlos y
    // soltarlos bien, hay que tomarlos lo mas tarde posible.
    //
    // La preparacion se cuenta desde que la orden se creo.
    const std::int64_t readyMs = order.createdMs + order.prepMs;
    if (readyMs > arrivedMs && !waitSim(readyMs - arrivedMs)) {
        return;   // le pidieron parar esperando la comida; todavia no tomo ningun lugar
    }

    // Hace fila por un lugar. Es el unico punto donde el courier se bloquea esperando a
    // OTRO courier, y por eso el estado atRestaurant existe separado de toPickup: en la
    // ventana, un courier naranja es uno esperando su turno.
    //
    // Su propio candado NO esta tomado mientras espera: acquire() bloquea sobre el
    // candado del restaurante, asi que el renderer puede seguir pidiendole snapshot().
    const auto entry = ctx_.restaurants.find(order.restaurantId);
    Restaurant* const restaurant =
        entry == ctx_.restaurants.end() ? nullptr : entry->second.get();
    if (restaurant != nullptr && !restaurant->acquire()) {
        return;   // le pidieron parar mientras hacia fila
    }

    ctx_.book.pickedUp(orderId, id_);
    if (restaurant != nullptr) {
        restaurant->release();   // retiro: libera el lugar para el siguiente
    }

    // La otra mitad del sorteo, ya con la comida encima. Es el caso interesante:
    // el traspaso tiene que mover una orden RETIRADA, y el receptor termina una
    // entrega que empezo otro. Sorteando solo antes de salir, eso no pasaba nunca.
    //
    // Se reparte la probabilidad entre los dos puntos para no cambiar la tasa de
    // averias por viaje: lo que cambia es CUANDO ocurren, no cuantas.
    if (maybeBreakDown(0.5)) {
        return;   // el traspaso ya vacio la bolsa; la orden la termina otro
    }

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
            !hasRoomLocked()) {
            return false;
        }
        state_.bag.push_back(order.id);
    }
    cv_.notify_one();   // fuera del candado: si no, despertaria para volver a dormirse
    return true;
}

bool CourierActor::hasRoomLocked() const {
    // Los lugares reservados cuentan como ocupados: si no, dos reservas simultaneas
    // pasarian las dos la verificacion y la bolsa terminaria con una orden de mas.
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
    // rng_ es de este hilo: ningun otro lo toca, asi que no necesita candado.
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    if (chance(rng_) >= probability) {
        return false;
    }
    return breakDownAndHandOver();
}

bool CourierActor::forceBreakdown() { return breakDownAndHandOver(); }

bool CourierActor::breakDownAndHandOver() {
    if (fleet_ == nullptr) {
        return false;   // sin companeros no hay a quien traspasarle nada
    }

    // Todas las ordenes van a UN solo companero, o la averia no ocurre. Eso es lo que
    // significa todo-o-nada, y es el modelo que define el §6: orderReassigned lleva from
    // y to obligatorios, los dos ids de courier, asi que las ordenes de un averiado van
    // a otro courier y no de vuelta a la cola.
    //
    // Devolverlas a la cola parecia mejor -mas averias se concretan- pero implica mover
    // la orden de assigned a created sin ningun evento que lo represente, y el §6 abre
    // exigiendo que todo cambio de estado quede registrado. Un cambio silencioso rompe
    // la validacion del log.
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

    // El registro ocurre YA SOLTADOS los dos candados de courier. La jerarquia permite
    // courier -> libro -> log, pero no hay motivo para sostenerlos mientras el libro
    // escribe.
    ctx_.book.recordBreakdown(id_);
    for (const OrderId orderId : moved) {
        ctx_.book.handOver(orderId, id_, receiver);
    }
    return true;
}

std::vector<OrderId> CourierActor::tryHandOverTo(CourierActor& target) {
    // DOS candados a la vez. Tomarlos uno por uno, cada hilo en el orden que le toque,
    // es el interbloqueo de manual: si A se averia y elige a B mientras B se averia y
    // elige a A, cada uno tendria el suyo esperando el del otro para siempre.
    //
    // scoped_lock los toma con un algoritmo que evita ese ciclo, sin importar el orden
    // en que se le pasen. Por eso las averias simultaneas no pueden bloquear el sistema,
    // que es lo que exige el §4.
    const std::scoped_lock guard(mutex_, target.mutex_);

    if (state_.bag.empty() || state_.state == CourierState::brokenDown) {
        return {};
    }
    // Con una reserva a medio camino no se puede averiar: el dispatcher ya la registro
    // en el libro y va a confirmarla en microsegundos, y esa orden no esta en la bolsa,
    // asi que el traspaso la dejaria en manos de un courier fuera de servicio.
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
        return {};   // todo-o-nada: si no entran todas, no se mueve ninguna
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
