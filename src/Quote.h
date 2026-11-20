#pragma once

#include <cstdint>

#include "CityGraph.h"
#include "Courier.h"
#include "Order.h"
#include "Router.h"

namespace accel {
class TravelTimeTable;
}

inline constexpr std::int64_t kUnreachable = -1;

std::int64_t estimateDeliveryMs(Router& router, const CityGraph& graph,
                                const Courier& courier, const Order& order);

// Igual que estimateDeliveryMs, pero los tiempos de viaje salen de la tabla de todos los pares
// (dos lecturas) en vez de correr Dijkstra. Con table == nullptr, o si la tabla no es de este
// grafo, usa el router: el comportamiento es el de siempre.
std::int64_t estimateDeliveryMsWithTable(Router& router, const CityGraph& graph,
                                         const Courier& courier, const Order& order,
                                         const accel::TravelTimeTable* table);
