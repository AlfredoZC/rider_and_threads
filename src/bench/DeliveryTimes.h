#pragma once

#include <cstdint>
#include <vector>

namespace bench {

// Tiempos de entrega sinteticos y deterministas segun la semilla (compartidos por las aplicaciones de bench
// del histograma y de los percentiles): log-normal con mediana de 28 min (casi todo entre 10 y 60 min, cola
// larga hacia horas) y cada 1024 entregas un valor exacto en un borde de minuto (0, 1, ... 70 min).
void fillDeliveryTimes(std::vector<std::int32_t>& deliveryTimesMs, std::uint32_t seed);

}  // namespace bench
