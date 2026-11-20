#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace accel {

// Scan inclusivo jerarquico en la GPU (Kogge-Stone con doble buffer por tramo de 512 + scan recursivo de
// los totales de tramo), compartido por las aplicaciones que lo usan (pedidos en curso, radix sort).
// Solo para codigo de dispositivo (.cu); no lanza: devuelve "" o un mensaje de error.
constexpr int kScanBlockSize = 512;  // hilos por bloque = elementos por tramo (medido mas rapido que 1024)

// Enteros auxiliares (en la GPU) que necesita el scan de `count` elementos: por nivel, los totales de
// tramo y su scan. Siempre >= 1.
std::size_t scanScratchCount(std::size_t count);

// Scan inclusivo de `count` enteros que ya estan en la GPU. `scratch` tiene scanScratchCount(count) enteros.
std::string scanInclusiveOnDevice(const std::int32_t* input, std::int32_t* output, std::size_t count,
                                  std::int32_t* scratch);

}  // namespace accel
