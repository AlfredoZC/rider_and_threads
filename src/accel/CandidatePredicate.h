#pragma once

#include <cstddef>
#include <cstdint>

// Con nvcc, ACCEL_HOST_DEVICE marca una funcion que compila para la CPU y para la GPU; con g++ no
// significa nada. Asi el filtro de candidatos usa la MISMA definicion en los dos lados.
#ifdef __CUDACC__
#define ACCEL_HOST_DEVICE __host__ __device__
#else
#define ACCEL_HOST_DEVICE
#endif

namespace accel {

// Aplicacion 10: un courier es candidato de un pedido si esta disponible (no averiado, no en descanso
// y con lugar en la bolsa) y llega al restaurante en a lo sumo maxTravelMs segun la tabla de tiempos.
// La disponibilidad se mira primero: es la prueba mas barata y evita leer la tabla.
// `travelTimeMs` es la tabla nodeCount x nodeCount por filas; courierNode[courier] < nodeCount y
// restaurantNode < nodeCount (lo garantiza quien arma las columnas).
ACCEL_HOST_DEVICE inline bool isCandidate(std::size_t courier, const std::int32_t* courierNode,
                                          const std::uint8_t* courierAvailable, const std::int32_t* travelTimeMs,
                                          std::size_t nodeCount, std::int32_t restaurantNode,
                                          std::int32_t maxTravelMs) {
    if (courierAvailable[courier] == 0) {
        return false;
    }
    const std::size_t cell = static_cast<std::size_t>(courierNode[courier]) * nodeCount +
                             static_cast<std::size_t>(restaurantNode);
    return travelTimeMs[cell] <= maxTravelMs;
}

}  // namespace accel
