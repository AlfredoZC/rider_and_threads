#pragma once

#include <cstdint>

#include "CityGraph.h"
#include "Courier.h"
#include "Order.h"
#include "Router.h"

inline constexpr std::int64_t kUnreachable = -1;

std::int64_t estimateDeliveryMs(Router& router, const CityGraph& graph,
                                const Courier& courier, const Order& order);
