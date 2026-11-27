#pragma once

#include <cstdint>

#include "CityGraph.h"
#include "Courier.h"
#include "Order.h"
#include "Router.h"
#include "accel/TravelTimeTable.h"

inline constexpr std::int64_t kUnreachable = -1;

std::int64_t estimateDeliveryMs(Router& router, const CityGraph& graph,
                                const Courier& courier, const Order& order);

// Lo mismo, leyendo la tabla de tiempos entre todos los pares (dos lecturas en vez de dos Dijkstra).
// Da exactamente los mismos ms que la version con Router, incluido kUnreachable cuando no hay camino
// o un nodo no existe. `table` debe haberse armado con `graph` (mismos indices de nodo).
std::int64_t estimateDeliveryMs(const accel::TravelTimeTable& table, const CityGraph& graph,
                                const Courier& courier, const Order& order);
