#include "Restaurant.h"

#include <utility>

Restaurant::Restaurant(std::string id, int slots) : id_(std::move(id)), slots_(slots) {}

bool Restaurant::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);

    const std::uint64_t myTicket = nextTicket_++;

    cv_.wait(lock, [this, myTicket] {
        return stop_ || (myTicket == nowServing_ && inUse_ < slots_);
    });

    if (stop_) {
        if (myTicket == nowServing_) {
            ++nowServing_;
            cv_.notify_all();
        }
        return false;
    }

    ++inUse_;
    ++nowServing_;
    cv_.notify_all();
    return true;
}

void Restaurant::release() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        --inUse_;
    }
    cv_.notify_all();
}

void Restaurant::stop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
}

int Restaurant::inUse() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return inUse_;
}

int Restaurant::waiting() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<int>(nextTicket_ - nowServing_);
}
