// Las dependencias compartidas por todos los actores de la simulacion.
//
// Existe para no arrastrar cuatro parametros sueltos por cada constructor, y porque el
// Dispatcher necesita exactamente las mismas. Fijate en la naturaleza de las cuatro: dos
// son inmutables (el grafo y el reloj) y dos son monitores que se protegen solos. No hay
// aca ningun dato que un actor tenga que proteger por su cuenta.
#pragma once

#include "CityGraph.h"
#include "EventLog.h"
#include <memory>
#include <string>
#include <unordered_map>

#include "OrderBook.h"
#include "Restaurant.h"
#include "SimClock.h"

// Los restaurantes, indexados por su id. unique_ptr porque Restaurant tiene un mutex
// adentro: no es copiable ni movible, y el mapa no podria reubicarlo al crecer.
using Restaurants = std::unordered_map<std::string, std::unique_ptr<Restaurant>>;

struct SimContext {
    const CityGraph& graph;   // inmutable: se lee sin candado
    const SimClock& clock;    // sin estado mutable: se lee sin candado
    EventLog& log;            // monitor
    OrderBook& book;          // monitor
    Restaurants& restaurants; // un monitor por restaurante
};
