// Un grupo acotado de trabajadores que calculan estimaciones.
//
// Antes cada estimacion se lanzaba con std::async(launch::async), que crea un hilo del
// sistema operativo. Con los 8 couriers de la configuracion de ejemplo son 8 hilos por
// orden; con los 64 que el §11 dice que se van a probar, 64 por orden, y en una rafaga
// cientos vivos a la vez. Crear un hilo cuesta decenas de microsegundos y una pila
// reservada: pasado cierto punto el sistema gasta mas en crear hilos que en calcular.
//
// Aca los hilos se crean UNA vez y toman tareas de una cola. Cuantos, lo decide el
// hardware: std::thread::hardware_concurrency(). Mas trabajadores que nucleos no
// agregarian paralelismo real, solo cambios de contexto.
//
// Cada trabajador tiene su PROPIO Router, la memoria de trabajo de Dijkstra, creado una
// sola vez y reutilizado en todas las estimaciones que le toquen.
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
    // La tarea recibe el Router del trabajador que la ejecuta.
    using Job = std::function<std::int64_t(Router&)>;

    // workers = 0 significa "los que diga el hardware".
    explicit QuotePool(const CityGraph& graph, unsigned workers = 0);
    ~QuotePool();

    QuotePool(const QuotePool&) = delete;
    QuotePool& operator=(const QuotePool&) = delete;

    // Encola la tarea y devuelve el future por el que llegara su resultado. Es un canal
    // de un solo uso: una escritura, una lectura.
    std::future<std::int64_t> submit(Job job);

    // Vacia la cola, despierta a los trabajadores y los espera. Idempotente.
    void stop();

    std::size_t workers() const { return workers_.size(); }

private:
    void workerLoop();

    struct Task {
        Job job;
        // shared_ptr y no el promise directo: std::function exige que lo que captura sea
        // copiable, y un std::promise no lo es. El shared_ptr hace copiable al paquete y
        // garantiza que el promise viva mientras lo necesiten tanto quien encola como el
        // trabajador que lo cumple.
        std::shared_ptr<std::promise<std::int64_t>> result;
    };

    const CityGraph& graph_;
    std::vector<std::thread> workers_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<Task> queue_;   // protegida por mutex_
    bool stop_ = false;        // protegida por mutex_
};
