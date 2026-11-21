#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace accel {

struct Range {
    std::size_t begin = 0;
    std::size_t end = 0;
};

// Trozo `partIndex` de `parts` en que se parte [0, itemCount): trozos contiguos que cubren [0, itemCount) sin
// solapes y cuyos tamanos difieren en a lo sumo 1. Sin multiplicaciones que puedan desbordar.
// Requiere parts >= 1 e partIndex < parts.
Range chunkRange(std::size_t itemCount, std::size_t parts, std::size_t partIndex);

// Pool de CPU hecho solo con la biblioteca estandar (std::thread, mutex, condition_variable).
// `threads` es el paralelismo total: el hilo que llama a parallelFor ejecuta el trozo 0 y los
// threads - 1 hilos del pool los demas. Los hilos esperan bloqueados en una condition_variable
// (sin giro ni sleep) y el destructor los une a todos.
// parallelFor no es reentrante ni se llama desde dos hilos a la vez: lo serializa un mutex.
class ThreadPool {
public:
    explicit ThreadPool(unsigned threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned threads() const { return threads_; }

    // Reparte [0, itemCount) en `threads()` trozos y llama body(begin, end) por cada uno (los vacios se
    // omiten). Vuelve cuando todos terminaron. Si algun cuerpo lanza, se relanza la primera
    // excepcion en el hilo que llama, despues de que todos terminen.
    void parallelFor(std::size_t itemCount, const std::function<void(std::size_t, std::size_t)>& body);

private:
    void workerLoop(std::size_t partIndex);
    void runChunk(std::size_t partIndex);

    unsigned threads_;
    std::vector<std::thread> workers_;

    std::mutex callMutex_;  // serializa los parallelFor

    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    // Estado protegido por mutex_.
    const std::function<void(std::size_t, std::size_t)>* body_ = nullptr;
    std::size_t itemCount_ = 0;
    std::size_t generation_ = 0;
    std::size_t pending_ = 0;
    bool stop_ = false;
    std::exception_ptr error_;
};

}  // namespace accel
