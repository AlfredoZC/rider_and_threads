#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "EventLog.h"
#include "Restaurant.h"
#include "SimClock.h"

// Un unico hilo que marca el principio y el fin de cada pausa de cocina, en el instante
// exacto del calendario. Espera con plazo absoluto (wait_until) y se interrumpe con requestStop().
// RAII: el destructor pide parar y hace join.
class KitchenScheduler {
public:
    KitchenScheduler(std::vector<Restaurant*> restaurants, const SimClock& clock, EventLog& log);
    ~KitchenScheduler();

    KitchenScheduler(const KitchenScheduler&) = delete;
    KitchenScheduler& operator=(const KitchenScheduler&) = delete;

    void start();
    void requestStop();
    void join();

private:
    struct Slot {
        Restaurant* restaurant;
        bool paused = false;
        std::int64_t k = 0;
        std::int64_t nextMs() const;
    };

    void run();

    std::vector<Slot> slots_;
    const SimClock& clock_;
    EventLog& log_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};
