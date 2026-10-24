#pragma once

#include <cstddef>
#include <cstdint>
#include <random>

#include "Config.h"
#include "Ids.h"
#include "Order.h"

class OrderFactory {
public:
    OrderFactory(const Config& cfg, std::uint32_t seed);

    Order create(std::int64_t nowMs);

private:
    const Config& cfg_;
    std::mt19937 rng_;
    OrderId nextId_ = 1;
    std::uniform_int_distribution<std::size_t> restaurantPick_;
    std::uniform_int_distribution<std::size_t> nodePick_;
};
