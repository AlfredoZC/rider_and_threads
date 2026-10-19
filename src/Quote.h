// Estimacion del tiempo de entrega: cuanto tardaria un courier en hacerse cargo de una
// orden (§4, "el dispatcher elige por mejor tiempo estimado de entrega").
//
// Es una funcion libre y no un metodo del CourierActor, y la razon es de concurrencia.
// El Router tiene buffers que se escriben en cada consulta; si viviera dentro del actor,
// lo escribirian a la vez el hilo del dispatcher (estimando) y el del propio courier
// (calculando su ruta para moverse). Siendo parametro, cada hilo trae EL SUYO.
//
// Asi la funcion no toca ningun dato compartido mutable: el CityGraph es inmutable, y
// el Courier y la Order son copias. Varios hilos pueden estimar a la vez sin un solo
// candado, que es lo que hace posible calcular las estimaciones simultaneamente.
#pragma once

#include <cstdint>

#include "CityGraph.h"
#include "Courier.h"
#include "Order.h"
#include "Router.h"

// No hay camino respetando los sentidos de las calles: es el rechazo unreachable del §4.
inline constexpr std::int64_t kUnreachable = -1;

// Tiempo SIMULADO que tardaria este courier: ir a buscar la orden, esperar a que la
// comida este lista si llega antes, y llevarla al destino.
std::int64_t estimateDeliveryMs(Router& router, const CityGraph& graph,
                                const Courier& courier, const Order& order);
