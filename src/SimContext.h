#pragma once

#include "CityGraph.h"
#include "EventLog.h"
#include <memory>
#include <string>
#include <unordered_map>

#include "OrderBook.h"
#include "Restaurant.h"
#include "SimClock.h"

using Restaurants = std::unordered_map<std::string, std::unique_ptr<Restaurant>>;

struct SimContext {
    const CityGraph& graph;
    const SimClock& clock;
    EventLog& log;
    OrderBook& book;
    Restaurants& restaurants;
};
