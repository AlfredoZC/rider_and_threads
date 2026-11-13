#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

#include "KitchenSchedule.h"

class EventLog;
class SimClock;

class Restaurant {
public:
    // Callback con el instante simulado (-1 si no hay reloj) del hecho; se invoca bajo el
    // candado del restaurante.
    using Hook = std::function<void(std::int64_t)>;
    // Igual, pero ademas informa si la cocina estaba en pausa al entrar en la fila.
    using QueuedHook = std::function<void(std::int64_t, bool)>;

    Restaurant(std::string id, int slots);
    Restaurant(std::string id, int slots, KitchenSchedule schedule, const SimClock* clock);

    Restaurant(const Restaurant&) = delete;
    Restaurant& operator=(const Restaurant&) = delete;

    // onQueued: al sacar el ticket (entrar en la fila). onGranted: al obtener el puesto.
    bool acquire(const QueuedHook& onQueued = QueuedHook(), const Hook& onGranted = Hook());
    void release();

    void stop();

    // Solo las llama el KitchenScheduler, con t = limite exacto del calendario.
    void beginPause(std::int64_t t, EventLog& log);
    void endPause(std::int64_t t, EventLog& log);

    const std::string& id() const { return id_; }
    int slots() const { return slots_; }
    const KitchenSchedule& schedule() const { return schedule_; }
    int inUse() const;
    int waiting() const;
    std::int64_t pauses() const;
    std::int64_t delayedByPause() const;

private:
    std::int64_t nowMs() const;

    const std::string id_;
    const int slots_;
    const KitchenSchedule schedule_;
    const SimClock* const clock_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int inUse_ = 0;
    std::uint64_t nextTicket_ = 0;
    std::uint64_t nowServing_ = 0;
    bool stop_ = false;
    bool paused_ = false;
    std::int64_t pauses_ = 0;
    std::int64_t delayed_ = 0;
};
