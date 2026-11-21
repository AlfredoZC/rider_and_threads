#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include "Config.h"

namespace bench {

// Subgrafo conexo de la ciudad con exactamente `nodeCount` nodos: una region que crece desde `startNodeId`
// por busqueda en anchura (BFS), tratando las calles como no dirigidas para crecer (conexidad
// debil) y visitando los vecinos en el orden de `city.nodes`, de modo que el resultado es
// determinista y las regiones son anidadas (la de 100 esta dentro de la de 400).
//
// Conserva, en su orden original, los nodos elegidos y todas las calles de la ciudad cuyos dos
// extremos quedaron dentro, con su id, su sentido (`oneWay`) y su from -> to. Los restaurantes
// que quedan sobre nodos elegidos se conservan; todo lo demas del config se copia tal cual. El
// resultado se usa como cualquier Config (CityGraph, Router...). Si el inicio no esta en la
// componente suficiente, nodeCount == 0 o nodeCount supera los nodos alcanzables, lanza std::invalid_argument.
Config extractRegion(const Config& city, const std::string& startNodeId, std::size_t nodeCount);

}  // namespace bench
