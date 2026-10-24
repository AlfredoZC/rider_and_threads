#include "OrderGenerator.h"

#include <algorithm>

namespace {
constexpr double kBurstProbability = 0.10;
}

OrderGenerator::OrderGenerator(const Config& cfg, const SimClock& clock, OrderBook& book)
    : clock_(clock),
      book_(book),
      factory_(cfg, cfg.seed),
      rng_(cfg.seed ^ 0x9e3779b9u),
      meanIntervalMs_(cfg.meanIntervalMs),
      burstMax_(cfg.burstMax) {}

std::int64_t OrderGenerator::nextIntervalMs() {
    std::exponential_distribution<double> interval(
        1.0 / static_cast<double>(meanIntervalMs_));
    const double ms = interval(rng_);
    return std::max<std::int64_t>(1, static_cast<std::int64_t>(ms));
}

int OrderGenerator::nextBurstSize() {
    if (burstMax_ <= 1) {
        return 1;
    }
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    if (chance(rng_) >= kBurstProbability) {
        return 1;
    }
    std::uniform_int_distribution<int> size(2, burstMax_);
    return size(rng_);
}

void OrderGenerator::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopRequested_) {
        const std::int64_t waitSimMs = nextIntervalMs();

        if (cv_.wait_for(lock, clock_.realDuration(waitSimMs),
                         [this] { return stopRequested_; })) {
            break;
        }

        const int burst = nextBurstSize();

        lock.unlock();
        for (int i = 0; i < burst; ++i) {
            book_.admit(factory_.create(clock_.nowMs()));
        }
        lock.lock();
        generated_ += burst;
    }
}

void OrderGenerator::requestStop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopRequested_ = true;
    }
    cv_.notify_one();
}

std::int64_t OrderGenerator::generated() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return generated_;
}
