#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "CourierActor.h"
#include "OrderBook.h"

std::string buildSummary(const OrderStats& stats, std::int64_t breakdowns,
                         const std::vector<std::unique_ptr<CourierActor>>& fleet);
