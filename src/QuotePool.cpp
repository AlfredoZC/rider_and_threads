#include "QuotePool.h"

#include <stdexcept>
#include <utility>

QuotePool::QuotePool(const CityGraph& graph, unsigned workers) : graph_(graph) {
    // La decision la toma el hardware. hardware_concurrency puede devolver 0 cuando no
    // puede averiguarlo, y en ese caso hay que elegir algo razonable a mano: un pool de
    // cero trabajadores seria un cuelgue garantizado.
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
            // Ya se esta cerrando: se cumple el future con una excepcion en vez de
            // dejarlo colgado. Quien espere recibe un error, no un bloqueo.
            result->set_exception(
                std::make_exception_ptr(std::runtime_error("QuotePool detenido")));
            return future;
        }
        queue_.push(Task{std::move(job), std::move(result)});
    }
    cv_.notify_one();   // fuera del candado: quien despierta no lo encuentra ocupado
    return future;
}

void QuotePool::workerLoop() {
    // Un Router por trabajador, creado una sola vez. Antes se construia uno por cada
    // estimacion, es decir dos vectores de 668 posiciones cada vez.
    Router router(graph_);

    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            // Al parar se atienden primero las tareas que quedaron. Si se descartaran,
            // sus promises moririan sin cumplirse y quien espera recibiria un
            // broken_promise: un error que nadie pidio.
            if (queue_.empty()) {
                return;   // stop_ y nada pendiente
            }
            task = std::move(queue_.front());
            queue_.pop();
        }

        // El calculo ocurre SIN el candado del pool: es el trabajo largo.
        try {
            task.result->set_value(task.job(router));
        } catch (...) {
            // La excepcion viaja al hilo que espera y se vuelve a lanzar en su get().
            // Es el mecanismo que pide el §4: una estimacion que falla llega como error
            // y no tumba el programa. Dejarla escapar de un hilo llamaria a
            // std::terminate.
            task.result->set_exception(std::current_exception());
        }
    }
}

void QuotePool::stop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (stop_) {
            return;   // idempotente: el destructor puede llamarlo despues de stop()
        }
        stop_ = true;
    }
    cv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();   // ningun hilo queda abandonado (§7)
        }
    }
}
