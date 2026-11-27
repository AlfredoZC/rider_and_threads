#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "CityGraph.h"

struct Route {
    bool found = false;
    std::int64_t travelMs = 0;
    std::vector<NodeIndex> nodes;
};

class Router {
public:
    explicit Router(const CityGraph& graph);

    Route shortestPath(NodeIndex from, NodeIndex to);

    // Dijkstra completo (sin cortar al llegar a un destino) desde `from`: tiempo minimo en ms a CADA
    // nodo, kInfinityMs si no hay camino. El vector es del Router (se pisa en la llamada siguiente) y
    // queda vacio si `from` no existe. Es la fila `from` de la tabla de tiempos entre todos los pares.
    const std::vector<std::int64_t>& travelTimesFrom(NodeIndex from);

    static constexpr std::int64_t kInfinityMs = std::numeric_limits<std::int64_t>::max();

private:
    // Dijkstra desde `from`; se detiene al sacar `stopAt` de la cola (kNoNode = no se detiene).
    void runDijkstra(NodeIndex from, NodeIndex stopAt);

    const CityGraph& graph_;
    std::vector<std::int64_t> dist_;
    std::vector<NodeIndex> prev_;
};
