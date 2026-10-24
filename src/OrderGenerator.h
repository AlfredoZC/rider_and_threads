#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>

#include "Config.h"
#include "OrderBook.h"
#include "OrderFactory.h"
#include "SimClock.h"

class OrderGenerator {
public:
    OrderGenerator(const Config& cfg, const SimClock& clock, OrderBook& book);

    OrderGenerator(const OrderGenerator&) = delete;
    OrderGenerator& operator=(const OrderGenerator&) = delete;

    void run();
    void requestStop();

    std::int64_t generated() const;

private:
    std::int64_t nextIntervalMs();
    int nextBurstSize();

    const SimClock& clock_;
    OrderBook& book_;

    OrderFactory factory_;
    std::mt19937 rng_;
    const std::int64_t meanIntervalMs_;
    const int burstMax_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopRequested_ = false;
    std::int64_t generated_ = 0;
};
