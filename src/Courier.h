// Un courier de la flota (sección 4 del brief).
//
// Es solo el dato: copiable y sin sincronización propia. Quien lo posee y le pone un
// hilo es CourierActor, que lo protege con su mutex y entrega copias por snapshot().
// Misma separación que Order frente a OrderBook: el dato viaja, el actor no se copia.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Ids.h"

// El §6 pide que los estados se distingan en la ventana: un color por estado.
enum class CourierState {
    idle,           // sin órdenes; espera sin consumir CPU
    toPickup,       // en camino al restaurante
    atRestaurant,   // quieto: hace fila por un pickupSlot o espera que salga la comida
    toDropoff,      // llevando la comida al cliente
    brokenDown,     // averiado; sus órdenes se traspasan todo-o-nada
};

struct Courier {
    CourierId id = kNoCourier;
    CourierState state = CourierState::idle;

    std::vector<OrderId> bag;   // nunca más de bagCapacity elementos
    int bagCapacity = 0;
    int delivered = 0;          // alimenta deliveredPerCourier del resumen

    // Posición: el courier va de fromNode a toNode, y sale y llega en estos instantes de
    // tiempo SIMULADO. No se guarda un "avance" que alguien tenga que ir moviendo: quien
    // necesite la posición la calcula con el reloj (ver positionRatio).
    //
    // Así el hilo del courier duerme el tramo entero en vez de despertarse veinte veces
    // por segundo a actualizar un número, y el dibujo queda continuo en vez de a saltos.
    //
    // Invariante: quieto en un nodo  =>  fromNode == toNode y segmentStartMs ==
    // segmentEndMs, que es lo que hace que positionRatio devuelva 1.
    std::string fromNode;
    std::string toNode;
    std::int64_t segmentStartMs = 0;
    std::int64_t segmentEndMs = 0;

    Courier(CourierId courierId, const std::string& startNode, int capacity)
        : id(courierId), bagCapacity(capacity), fromNode(startNode), toNode(startNode) {
        bag.reserve(static_cast<std::size_t>(capacity));
    }
};

inline const char* toString(CourierState s) {
    switch (s) {
        case CourierState::idle:         return "idle";
        case CourierState::toPickup:     return "toPickup";
        case CourierState::atRestaurant: return "atRestaurant";
        case CourierState::toDropoff:    return "toDropoff";
        case CourierState::brokenDown:   return "brokenDown";
    }
    return "unknown";
}

inline bool hasRoom(const Courier& c) {
    return c.bag.size() < static_cast<std::size_t>(c.bagCapacity);
}

inline bool isBusy(CourierState s) {
    return s == CourierState::toPickup || s == CourierState::atRestaurant ||
           s == CourierState::toDropoff;
}

// El último nodo por el que pasó de verdad. A mitad de un tramo es el nodo del que
// salió: es desde donde tiene sentido calcular cualquier ruta nueva.
inline const std::string& currentNode(const Courier& c) { return c.fromNode; }

// Qué fracción del tramo actual lleva recorrida, entre 0 y 1, según el reloj. La usa el
// dibujo para interpolar la posición, y por eso no necesita que nadie la actualice.
//
// El resultado se acota a propósito: si el hilo tardó en despertarse y el reloj se pasó
// del instante de llegada, devuelve 1 y no 1,3, que dibujaría al courier pasado de la
// esquina.
inline double positionRatio(const Courier& c, std::int64_t nowMs) {
    if (c.segmentEndMs <= c.segmentStartMs) {
        return 1.0;   // quieto, o un tramo de duración cero
    }
    const double total = static_cast<double>(c.segmentEndMs - c.segmentStartMs);
    const double done = static_cast<double>(nowMs - c.segmentStartMs);
    if (done <= 0.0) {
        return 0.0;
    }
    return done >= total ? 1.0 : done / total;
}
