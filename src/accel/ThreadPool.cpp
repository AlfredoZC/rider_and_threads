#include "accel/ThreadPool.h"

namespace accel {

Range chunkRange(std::size_t itemCount, std::size_t parts, std::size_t partIndex) {
    const std::size_t baseLength = itemCount / parts;
    const std::size_t remainder = itemCount % parts;
    // partIndex * baseLength <= itemCount y min(partIndex, remainder) < parts: no desborda mientras partIndex < parts.
    const std::size_t begin = partIndex * baseLength + (partIndex < remainder ? partIndex : remainder);
    const std::size_t length = baseLength + (partIndex < remainder ? 1 : 0);
    return Range{begin, begin + length};
}

ThreadPool::ThreadPool(unsigned threads) : threads_(threads == 0 ? 1 : threads) {
    // Si crear un hilo falla (EAGAIN, limite de memoria...), el destructor NO corre: hay que
    // detener y unir los ya creados aqui, antes de que se destruyan mutex_ y wake_ (que esperan).
    try {
        workers_.reserve(threads_ - 1);
        for (unsigned workerNumber = 1; workerNumber < threads_; ++workerNumber) {
            workers_.emplace_back(&ThreadPool::workerLoop, this, static_cast<std::size_t>(workerNumber));
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for (std::thread& worker : workers_) {
            worker.join();
        }
        throw;
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::runChunk(std::size_t partIndex) {
    const Range range = chunkRange(itemCount_, threads_, partIndex);
    if (range.begin == range.end) {
        return;
    }
    try {
        (*body_)(range.begin, range.end);
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) {
            error_ = std::current_exception();
        }
    }
}

void ThreadPool::workerLoop(std::size_t partIndex) {
    std::size_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) {
                return;
            }
            seen = generation_;
        }
        runChunk(partIndex);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--pending_ == 0) {
                done_.notify_one();
            }
        }
    }
}

void ThreadPool::parallelFor(std::size_t itemCount, const std::function<void(std::size_t, std::size_t)>& body) {
    std::lock_guard<std::mutex> call(callMutex_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        body_ = &body;
        itemCount_ = itemCount;
        error_ = nullptr;
        pending_ = workers_.size();
        ++generation_;
    }
    if (!workers_.empty()) {
        wake_.notify_all();
    }
    runChunk(0);
    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [&] { return pending_ == 0; });
        body_ = nullptr;
        error = error_;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

}  // namespace accel
