#pragma once

#include <cstddef>
#include <cstdint>

#include "accel/CandidatePredicate.h"  // ACCEL_HOST_DEVICE
#include "accel/TravelTimeTable.h"     // kUnreachableMs

namespace accel {

// Aplicacion 04: la ETA de un courier para un pedido. UNA sola definicion para la CPU y la GPU, y la
// MISMA formula que estimateDeliveryMsWithTable (Quote.cpp):
//
//     ETA = max(libreDesde + viajeAlRestaurante, pedidoListo) + viajeAlCliente
//
// con los dos viajes leidos de la tabla de tiempos. Si alguno no tiene camino la ETA es kUnreachableMs
// (se propaga). Se calcula en 64 bits y se satura a kUnreachableMs, asi nunca desborda un int32.
// courierNode, restaurantNode y customerNode deben ser < nodeCount (lo garantiza quien arma las columnas).
ACCEL_HOST_DEVICE inline std::int32_t etaForPair(std::int32_t courierNode, std::int32_t courierFreeAtMs,
                                                 std::int32_t restaurantNode, std::int32_t customerNode,
                                                 std::int32_t readyAtMs, const std::int32_t* travelTimeMs,
                                                 std::size_t nodeCount) {
    const std::int32_t toRestaurantMs =
        travelTimeMs[static_cast<std::size_t>(courierNode) * nodeCount + static_cast<std::size_t>(restaurantNode)];
    const std::int32_t toCustomerMs =
        travelTimeMs[static_cast<std::size_t>(restaurantNode) * nodeCount + static_cast<std::size_t>(customerNode)];
    if (toRestaurantMs >= kUnreachableMs || toCustomerMs >= kUnreachableMs) {
        return kUnreachableMs;  // no puede hacer este pedido
    }
    const long long arrivalMs = static_cast<long long>(courierFreeAtMs) + toRestaurantMs;  // llega al restaurante
    const long long pickupMs = arrivalMs > readyAtMs ? arrivalMs : readyAtMs;              // espera si la comida no esta
    const long long etaMs = pickupMs + toCustomerMs;
    return etaMs >= kUnreachableMs ? kUnreachableMs : static_cast<std::int32_t>(etaMs);
}

}  // namespace accel
