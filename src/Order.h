#pragma once

#include <cstdint>
#include <string>

#include "Ids.h"

enum class OrderState {
    created,
    assigned,
    pickedUp,
    delivered,
    rejected,
    pending,
};

enum class RejectReason {
    none,
    queueFull,
    unreachable,
    noCourier,
};

inline constexpr std::int64_t kNoTime = -1;

struct Order {
    OrderId id = kNoOrder;

    std::string restaurantId;
    std::string pickupNode;
    std::string dropoffNode;

    OrderState state = OrderState::created;
    RejectReason rejectReason = RejectReason::none;
    CourierId assignedCourier = kNoCourier;
    bool reassigned = false;

    std::int64_t createdMs = kNoTime;
    std::int64_t prepMs = 0;
    std::int64_t assignedMs = kNoTime;
    std::int64_t pickedUpMs = kNoTime;
    std::int64_t deliveredMs = kNoTime;
};

inline const char* toString(OrderState s) {
    switch (s) {
        case OrderState::created:   return "created";
        case OrderState::assigned:  return "assigned";
        case OrderState::pickedUp:  return "pickedUp";
        case OrderState::delivered: return "delivered";
        case OrderState::rejected:  return "rejected";
        case OrderState::pending:   return "pending";
    }
    return "unknown";
}

inline const char* toString(RejectReason r) {
    switch (r) {
        case RejectReason::none:        return "none";
        case RejectReason::queueFull:   return "queueFull";
        case RejectReason::unreachable: return "unreachable";
        case RejectReason::noCourier:   return "noCourier";
    }
    return "unknown";
}

inline bool isFinalState(OrderState s) {
    return s == OrderState::delivered || s == OrderState::rejected ||
           s == OrderState::pending;
}

inline bool canTransition(OrderState from, OrderState to) {
    if (isFinalState(from)) {
        return false;
    }
    if (to == OrderState::pending) {
        return true;
    }
    switch (from) {
        case OrderState::created:
            return to == OrderState::assigned || to == OrderState::rejected;
        case OrderState::assigned:
            return to == OrderState::pickedUp || to == OrderState::created;
        case OrderState::pickedUp:
            return to == OrderState::delivered;
        case OrderState::delivered:
        case OrderState::rejected:
        case OrderState::pending:
            return false;
    }
    return false;
}

inline std::int64_t deliveryMs(const Order& o) {
    if (o.state != OrderState::delivered) {
        return kNoTime;
    }
    return o.deliveredMs - o.createdMs;
}

