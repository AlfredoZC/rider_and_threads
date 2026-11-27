#include "accel/GpuWorker.h"

#include <vector>

namespace accel {

GpuWorker::GpuWorker(InitFn init) {
    std::promise<void> ready;
    std::future<void> readyFuture = ready.get_future();
    thread_ = std::thread(&GpuWorker::loop, this, std::move(init), std::move(ready));
    readyFuture.wait();  // bloqueado, sin sondeo: el hilo avisa al terminar init()
}

GpuWorker::~GpuWorker() { stop(); }

void GpuWorker::stop() {
    std::lock_guard<std::mutex> joinLock(joinMutex_);
    std::deque<Job> abandoned;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        abandoned.swap(queue_);
    }
    wake_.notify_all();
    // Se resuelven aqui, sin esperar a la tarea en curso.
    for (Job& job : abandoned) {
        job.fail(unavailable("GpuWorker detenido"));
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

bool GpuWorker::healthy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return healthy_;
}

std::string GpuWorker::unhealthyReason() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reason_;
}

void GpuWorker::enqueue(Job job) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_) {
            throw GpuUnavailableError("GpuWorker detenido");
        }
        if (!healthy_) {
            throw GpuUnavailableError("GPU no disponible: " + reason_);
        }
        queue_.push_back(std::move(job));
    }
    wake_.notify_one();
}

void GpuWorker::markUnhealthy(const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (healthy_) {
        healthy_ = false;
        reason_ = reason;
    }
}

std::exception_ptr GpuWorker::unavailable(const std::string& why) {
    return std::make_exception_ptr(GpuUnavailableError(why));
}

void GpuWorker::loop(InitFn init, std::promise<void> ready) {
    try {
        init();
    } catch (const std::exception& e) {
        markUnhealthy(e.what());
    } catch (...) {
        markUnhealthy("error desconocido al inicializar la GPU");
    }
    ready.set_value();

    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        wake_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
            return;  // stopping_ y nada pendiente (stop() ya resolvio lo que habia)
        }
        Job job = std::move(queue_.front());
        queue_.pop_front();
        if (!healthy_) {
            const std::string why = "GPU no disponible: " + reason_;
            lock.unlock();
            job.fail(unavailable(why));
            lock.lock();
            continue;
        }
        if (job.deadline.has_value() && Clock::now() >= *job.deadline) {
            lock.unlock();
            job.fail(std::make_exception_ptr(GpuTaskExpiredError("tarea GPU vencida en la cola")));
            lock.lock();
            continue;
        }
        lock.unlock();
        job.run();  // resuelve su future; nunca lanza
        lock.lock();
    }
}

}  // namespace accel
