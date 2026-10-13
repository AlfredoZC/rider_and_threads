// Caminos minimos sobre el CityGraph (Dijkstra).
//
// Vive aparte del CityGraph a proposito. Dijkstra necesita memoria de trabajo que
// cambia en cada consulta (distancias, predecesores, cola de prioridad). Si esa memoria
// fuera parte del CityGraph, el grafo dejaria de ser inmutable: dos hilos consultando a
// la vez se pisarian, habria que ponerle un candado y las consultas se harian una tras
// otra. Justo lo contrario de lo que pide el §4.
//
// Separamos entonces lo compartido de lo propio: el CityGraph es inmutable y lo leen
// todos sin candado; cada hilo tiene SU PROPIO Router con su propia memoria de trabajo.
// Nada compartido que se escriba, ningun candado, y los hilos no se estorban.
#pragma once

#include <cstdint>
#include <vector>

#include "CityGraph.h"

// found = false significa que no existe camino respetando los sentidos de las calles.
// No es un error del programa: es el motivo de rechazo "unreachable" del §4.
struct Route {
    bool found = false;
    std::int64_t travelMs = 0;        // tiempo SIMULADO
    std::vector<NodeIndex> nodes;     // from ... to, con los dos extremos incluidos
};

class Router {
public:
    // graph tiene que vivir mas que el Router. Se guarda por referencia porque es
    // inmutable y uno solo lo comparten todos los hilos.
    explicit Router(const CityGraph& graph);

    // El camino mas rapido de from a to.
    //
    // NO es const a proposito, aunque "consultar un camino" suene a operacion de solo
    // lectura: usa y modifica la memoria de trabajo del Router. Esa firma es la que le
    // avisa a quien lo lea que un Router no se comparte entre hilos.
    Route shortestPath(NodeIndex from, NodeIndex to);

private:
    const CityGraph& graph_;
    std::vector<std::int64_t> dist_;   // buffers reutilizados entre consultas: evitan
    std::vector<NodeIndex> prev_;      // pedir memoria en cada estimacion
};
