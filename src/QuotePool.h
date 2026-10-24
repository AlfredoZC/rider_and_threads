#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "CityGraph.h"
#include "Router.h"

class QuotePool {
public:
    using Job = std::function<std::int64_t(Router&)>;

    explicit QuotePool(const CityGraph& graph, unsigned workers = 0);
    ~QuotePool();

    QuotePool(const QuotePool&) = delete;
    QuotePool& operator=(const QuotePool&) = delete;

    std::future<std::int64_t> submit(Job job);

    void stop();

    std::size_t workers() const { return workers_.size(); }

private:
    void workerLoop();

    struct Task {
        Job job;
        std::shared_ptr<std::promise<std::int64_t>> result;
    };

    const CityGraph& graph_;
    std::vector<std::thread> workers_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Task> queue_;
    bool stop_ = false;
};
