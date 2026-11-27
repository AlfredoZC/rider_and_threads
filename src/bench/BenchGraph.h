#pragma once

#include <cstddef>
#include <cstdint>

#include "Config.h"

namespace bench {

// Grafo de calles de `nodeCount` nodos para las aplicaciones de bench que trabajan sobre el grafo (tabla de
// tiempos, etc.). Si `city` tiene al menos `nodeCount` nodos y velocidad valida, es la region de la ciudad
// real que da extractRegion (BFS desde el nodo de partida, regiones anidadas). Si no (sin ciudad cargada,
// ciudad mas chica, region invalida) es un grafo SINTETICO determinista segun `seed`: una grilla de
// calles con coordenadas levemente desordenadas, donde ~35 % de las calles son de un solo sentido (con
// el sentido al azar) y el ultimo nodo es un sumidero (solo se llega a el), asi que siempre hay pares sin camino.
// En ambos casos el resultado sirve para CityGraph y Router.
// `usedCity` (opcional) dice cual de los dos fue. Requiere nodeCount >= 1.
Config cityRegionOrSynthetic(const Config& city, std::size_t nodeCount, std::uint32_t seed, bool* usedCity);

}  // namespace bench
