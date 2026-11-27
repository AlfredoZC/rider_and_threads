#pragma once

#include "CityGraph.h"
#include "EventLog.h"
#include <memory>
#include <string>
#include <unordered_map>

#include "OrderBook.h"
#include "Restaurant.h"
#include "SimClock.h"

namespace accel {
class GpuWorker;
class TravelTimeTable;
}

using Restaurants = std::unordered_map<std::string, std::unique_ptr<Restaurant>>;

struct SimContext {
    const CityGraph& graph;
    const SimClock& clock;
    EventLog& log;
    OrderBook& book;
    Restaurants& restaurants;
    // Hilo dueño de la GPU, NO dueño aqui: lo crea y apaga main, despues de los hilos que lo
    // usan. nullptr = respaldo CPU. Puntero y no referencia porque es opcional.
    accel::GpuWorker* gpu = nullptr;
    // Tiempos minimos entre todos los pares de nodos, calculados una vez al arrancar (antes de lanzar los
    // hilos) y de SOLO LECTURA: se comparte sin candados. nullptr = las estimaciones usan el Router.
    std::shared_ptr<const accel::TravelTimeTable> travelTimes = nullptr;
};
