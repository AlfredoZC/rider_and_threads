#pragma once

#include <cstddef>

#include "Config.h"
#include "bench/BenchApp.h"

namespace bench {

// Grafo de entrada de las aplicaciones del bench que trabajan sobre la ciudad.
//  - Si el contexto trae ciudad (context.config.nodes no esta vacio): el subgrafo conexo real de `size`
//    nodos de CitySubgraph (extractRegion desde el nodo de partida). Lanza std::invalid_argument si la
//    ciudad no tiene `size` nodos alcanzables.
//  - Si no (contexto vacio, p. ej. en los tests): un grafo sintetico DETERMINISTA de exactamente `size`
//    nodos: una rejilla de ancho ceil(sqrt(size)) con calles entre vecinos, una de cada cinco de un solo
//    sentido, nodos con una pequena variacion fija de posicion (tiempos de viaje distintos) y un
//    restaurante en el nodo de partida "n0". Siempre da el mismo grafo para el mismo `size`.
// size == 0 lanza std::invalid_argument.
Config cityRegionOrSynthetic(const BenchContext& context, std::size_t size);

}  // namespace bench
