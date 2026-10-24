#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "Ids.h"

enum class CourierState {
    idle,
    toPickup,
    atRestaurant,
    toDropoff,
    brokenDown,
};

struct Courier {
    CourierId id = kNoCourier;
    CourierState state = CourierState::idle;

    std::vector<OrderId> bag;
    int bagCapacity = 0;
    int delivered = 0;

    std::string fromNode;
    std::string toNode;
    std::int64_t segmentStartMs = 0;
    std::int64_t segmentEndMs = 0;

    Courier(CourierId courierId, const std::string& startNode, int capacity)
        : id(courierId), bagCapacity(capacity), fromNode(startNode), toNode(startNode) {
        bag.reserve(static_cast<std::size_t>(capacity));
    }
};

inline const char* toString(CourierState s) {
    switch (s) {
        case CourierState::idle:         return "idle";
        case CourierState::toPickup:     return "toPickup";
        case CourierState::atRestaurant: return "atRestaurant";
        case CourierState::toDropoff:    return "toDropoff";
        case CourierState::brokenDown:   return "brokenDown";
    }
    return "unknown";
}

inline bool hasRoom(const Courier& c) {
    return c.bag.size() < static_cast<std::size_t>(c.bagCapacity);
}

inline bool isBusy(CourierState s) {
    return s == CourierState::toPickup || s == CourierState::atRestaurant ||
           s == CourierState::toDropoff;
}

inline const std::string& currentNode(const Courier& c) { return c.fromNode; }

inline double positionRatio(const Courier& c, std::int64_t nowMs) {
    if (c.segmentEndMs <= c.segmentStartMs) {
        return 1.0;
    }
    const double total = static_cast<double>(c.segmentEndMs - c.segmentStartMs);
    const double done = static_cast<double>(nowMs - c.segmentStartMs);
    if (done <= 0.0) {
        return 0.0;
    }
    return done >= total ? 1.0 : done / total;
}
