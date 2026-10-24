#pragma once

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>

class SimClock {
public:
    explicit SimClock(double timeScale)
        : start_(std::chrono::steady_clock::now()), timeScale_(timeScale) {
        assert(timeScale > 0.0);
    }

    std::int64_t nowMs() const {
        const std::chrono::duration<double, std::milli> real =
            std::chrono::steady_clock::now() - start_;
        return static_cast<std::int64_t>(real.count() * timeScale_);
    }

    std::chrono::microseconds realDuration(std::int64_t simMs) const {
        return std::chrono::microseconds(
            std::llround(static_cast<double>(simMs) * 1000.0 / timeScale_));
    }

    void sleepSim(std::int64_t simMs) const {
        std::this_thread::sleep_for(realDuration(simMs));
    }

    double timeScale() const { return timeScale_; }

private:
    const std::chrono::steady_clock::time_point start_;
    const double timeScale_;
};
