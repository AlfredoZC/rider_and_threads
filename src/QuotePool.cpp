#include "QuotePool.h"

#include <stdexcept>
#include <utility>

QuotePool::QuotePool(const CityGraph& graph, unsigned workers) : graph_(graph) {
    unsigned count = workers;
    if (count == 0) {
        count = std::thread::hardware_concurrency();
    }
    if (count == 0) {
        count = 2;
    }

    workers_.reserve(count);
    for (unsigned i = 0; i < count; ++i) {
        workers_.emplace_back(&QuotePool::workerLoop, this);
    }
}

QuotePool::~QuotePool() { stop(); }

std::future<std::int64_t> QuotePool::submit(Job job) {
    auto result = std::make_shared<std::promise<std::int64_t>>();
    std::future<std::int64_t> future = result->get_future();
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (stop_) {
            result->set_exception(
                std::make_exception_ptr(std::runtime_error("QuotePool detenido")));
            return future;
        }
        queue_.push(Task{std::move(job), std::move(result)});
    }
    cv_.notify_one();
    return future;
}

void QuotePool::workerLoop() {
    Router router(graph_);

    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                return;
            }
            task = std::move(queue_.front());
            queue_.pop();
        }

        try {
            task.result->set_value(task.job(router));
        } catch (...) {
            task.result->set_exception(std::current_exception());
        }
    }
}

void QuotePool::stop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (stop_) {
            return;
        }
        stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}
