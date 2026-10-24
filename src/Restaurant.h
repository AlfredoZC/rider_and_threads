#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>

class Restaurant {
public:
    Restaurant(std::string id, int slots);

    Restaurant(const Restaurant&) = delete;
    Restaurant& operator=(const Restaurant&) = delete;

    bool acquire();
    void release();

    void stop();

    const std::string& id() const { return id_; }
    int slots() const { return slots_; }
    int inUse() const;
    int waiting() const;

private:
    const std::string id_;
    const int slots_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int inUse_ = 0;
    std::uint64_t nextTicket_ = 0;
    std::uint64_t nowServing_ = 0;
    bool stop_ = false;
};
