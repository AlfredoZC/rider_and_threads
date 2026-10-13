// Fábrica de órdenes: el único lugar del programa que construye una Order.
//
// Es dueña del estado mutable de la creación (el próximo id y el generador aleatorio),
// y por eso es un objeto y no una función libre: ese estado tiene que sobrevivir entre
// llamadas y tener exactamente un dueño.
//
// NO es thread-safe, y no necesita serlo: la posee el hilo generador de órdenes, que es
// el único que crea órdenes (§4). Un dato con un único dueño no requiere sincronización.
// Si algún día crearan órdenes varios hilos, haría falta volver nextId_ un
// std::atomic<OrderId> y proteger rng_ (std::mt19937 no es thread-safe).
#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

#include "Config.h"
#include "Ids.h"
#include "Order.h"

class OrderFactory {
public:
    // cfg tiene que vivir más que la factory: se guarda por referencia porque es
    // inmutable y vive en main durante toda la simulación.
    OrderFactory(const Config& cfg, std::uint32_t seed);

    // Devuelve una orden válida, con id nuevo, fechada en nowMs (tiempo SIMULADO).
    Order create(std::int64_t nowMs);

private:
    const Config& cfg_;
    std::mt19937 rng_;
    OrderId nextId_ = 1;    // el 0 es kNoOrder
    std::uniform_int_distribution<std::size_t> restaurantPick_;
    std::uniform_int_distribution<std::size_t> nodePick_;
};
