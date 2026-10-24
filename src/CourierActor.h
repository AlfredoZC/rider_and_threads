#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <vector>
#include <string>

#include "Courier.h"
#include "Ids.h"
#include "Order.h"
#include "Router.h"
#include "SimContext.h"

class Fleet;

class CourierActor {
public:
    CourierActor(CourierId id, const std::string& startNode, int bagCapacity,
                 const SimContext& ctx, double breakdownProbability = 0.0,
                 std::uint32_t seed = 0, bool reinforcement = false,
                 std::int64_t idleMs = 0);

    CourierActor(const CourierActor&) = delete;
    CourierActor& operator=(const CourierActor&) = delete;

    void run();

    void requestStop();

    bool offer(const Order& order);

    bool reserve(const Order& order);
    void commitReservation(const Order& order);
    void cancelReservation();

    Courier snapshot() const;

    CourierId id() const { return id_; }

    void setFleet(const std::vector<std::unique_ptr<CourierActor>>* fleet) {
        fleet_ = fleet;
    }
    void setFleet(Fleet* fleet) { fleetManager_ = fleet; }

    bool forceBreakdown();

private:
    enum class Trip { arrived, stopped, unreachable };

    Trip travelTo(Router& router, const std::string& destNode);
    void deliver(Router& router, OrderId orderId);

    bool waitSim(std::int64_t simMs);

    void setCourierState(CourierState s);
    void finish(OrderId orderId, bool wasDelivered);

    bool maybeBreakDown();
    bool breakDownAndHandOver();
    std::vector<OrderId> tryHandOverTo(CourierActor& target);

    const CourierId id_;
    const SimContext ctx_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    const double breakdownProbability_;
    const bool reinforcement_;
    const std::int64_t idleMs_;
    std::mt19937 rng_;
    const std::vector<std::unique_ptr<CourierActor>>* fleet_ = nullptr;
    Fleet* fleetManager_ = nullptr;

    bool hasRoomLocked() const;

    Courier state_;
    bool stopRequested_ = false;
    bool retired_ = false;
    int reserved_ = 0;
};
