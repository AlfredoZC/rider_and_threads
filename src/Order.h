// Una orden de delivery (secciones 4 y 6 del brief).
// Es un dato puro, sin sincronización propia: la protege el OrderBook, que es su único
// dueño y el único que cambia su estado, dentro de su candado y validando la transición.
// Se diseña barata de copiar: los otros hilos copian la orden bajo candado y después
// trabajan con la copia, para no sostener el candado más de lo necesario.
#pragma once

#include <cstdint>
#include <string>

#include "Ids.h"

enum class OrderState {
    created,     // creada, sin courier
    assigned,    // un courier la aceptó
    pickedUp,    // retirada del restaurante, va en la bolsa
    delivered,   // entregada  (FINAL)
    rejected,    // rechazada  (FINAL)
    pending,     // viva cuando terminó la simulación  (FINAL)
};

enum class RejectReason {
    none,
    queueFull,     // la cola llegó a maxPending
    unreachable,   // no hay camino entre el restaurante y el destino
    noCourier,     // nadie la aceptó antes de acceptTimeoutMs
};

// El instante 0 es un tiempo válido, así que "todavía no ocurrió" necesita otro valor.
inline constexpr std::int64_t kNoTime = -1;

struct Order {
    OrderId id = kNoOrder;

    std::string restaurantId;   // "r0": va al campo restaurant del log
    std::string pickupNode;     // nodo del restaurante: campo from del log
    std::string dropoffNode;    // nodo de entrega: campo to del log

    OrderState state = OrderState::created;
    RejectReason rejectReason = RejectReason::none;
    CourierId assignedCourier = kNoCourier;
    bool reassigned = false;    // alimenta el contador reassigned del resumen

    // Todos los tiempos son tiempo SIMULADO en ms; kNoTime mientras no hayan ocurrido.
    std::int64_t createdMs = kNoTime;
    std::int64_t prepMs = 0;        // cuánto tarda la comida en estar lista
    std::int64_t assignedMs = kNoTime;
    std::int64_t pickedUpMs = kNoTime;
    std::int64_t deliveredMs = kNoTime;
};

inline const char* toString(OrderState s) {
    switch (s) {
        case OrderState::created:   return "created";
        case OrderState::assigned:  return "assigned";
        case OrderState::pickedUp:  return "pickedUp";
        case OrderState::delivered: return "delivered";
        case OrderState::rejected:  return "rejected";
        case OrderState::pending:   return "pending";
    }
    return "unknown";
}

inline const char* toString(RejectReason r) {
    switch (r) {
        case RejectReason::none:        return "none";
        case RejectReason::queueFull:   return "queueFull";
        case RejectReason::unreachable: return "unreachable";
        case RejectReason::noCourier:   return "noCourier";
    }
    return "unknown";
}

// delivered, rejected y pending son finales: created = delivered + rejected + pending.
inline bool isFinalState(OrderState s) {
    return s == OrderState::delivered || s == OrderState::rejected ||
           s == OrderState::pending;
}

// Única definición de qué cambios de estado son legales. El OrderBook la consulta
// dentro de su candado antes de mover una orden, y así se garantiza la regla del §4:
// cada orden termina en exactamente un estado final y nunca vuelve hacia atrás.
inline bool canTransition(OrderState from, OrderState to) {
    if (isFinalState(from)) {
        return false;   // de un estado final no se sale nunca más
    }
    if (to == OrderState::pending) {
        return true;    // al parar la simulación, cualquier orden viva queda pendiente
    }
    switch (from) {
        case OrderState::created:
            return to == OrderState::assigned || to == OrderState::rejected;
        case OrderState::assigned:
            // created: el courier se averió y la orden vuelve a repartirse (reassigned)
            return to == OrderState::pickedUp || to == OrderState::created;
        case OrderState::pickedUp:
            return to == OrderState::delivered;
        case OrderState::delivered:
        case OrderState::rejected:
        case OrderState::pending:
            return false;
    }
    return false;
}

inline std::int64_t deliveryMs(const Order& o) {
    if (o.state != OrderState::delivered) {
        return kNoTime;
    }
    return o.deliveredMs - o.createdMs;
}


