
// Un courier de la flota (sección 4 del brief).
// Por ahora es solo el dato: no tiene hilo propio ni sincronización. El std::mutex y
// la std::condition_variable se agregan cuando el Courier pase a ser un monitor.

#pragma once

#include <cstddef>
#include <string>
#include <vector> 

#include "Ids.h"

enum class CourierState{
    idle,
    toPickup,
    atRestaurant,
    toDropoff,
    brokenDown
};

struct Courier{
    CourierId id = kNoCourier;
    CourierState state = CourierState::idle;
    std::vector<OrderId> bag;
    int bagCapacity = 0; 
    int delivered  = 0;
    // Posición: va de fromNode a toNode y lleva recorrido progress del segmento.
    // Invariante: quieto en un nodo  =>  fromNode == toNode && progress == 0.
    std::string fromNode;
    std::string toNode;
    double progress = 0.0;

    Courier(CourierId courierId, const std::string& startNode, int capacity)
        : id(courierId),
          bagCapacity(capacity),
          fromNode(startNode),
          toNode(startNode) {
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

// El nodo donde está el courier. A mitad de una calle devuelve el nodo del que salió:
// el último por el que pasó de verdad.
inline const std::string& currentNode(const Courier& c) {
    return c.progress >= 1.0 ? c.toNode : c.fromNode;
}
