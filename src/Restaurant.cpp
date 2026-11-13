#include "Restaurant.h"

#include <utility>

#include "EventLog.h"
#include "SimClock.h"

Restaurant::Restaurant(std::string id, int slots)
    : Restaurant(std::move(id), slots, KitchenSchedule(), nullptr) {}

Restaurant::Restaurant(std::string id, int slots, KitchenSchedule schedule, const SimClock* clock)
    : id_(std::move(id)), slots_(slots), schedule_(schedule), clock_(clock) {}

std::int64_t Restaurant::nowMs() const { return clock_ != nullptr ? clock_->nowMs() : -1; }

bool Restaurant::acquire(const QueuedHook& onQueued, const Hook& onGranted) {
    std::unique_lock<std::mutex> lock(mutex_);

    const std::uint64_t myTicket = nextTicket_++;
    std::int64_t evaluatedAt = nowMs();
    // Tras stop() el reintento no es una llegada real a la fila: no se cuenta ni se informa.
    if (!stop_) {
        if (paused_) {
            ++delayed_;
        }
        if (onQueued) {
            onQueued(evaluatedAt, paused_);
        }
    }

    // Un puesto se obtiene solo si: no se pidio parar, es mi turno, hay puesto libre y la cocina
    // no esta en pausa ni segun el calendario (funcion pura del tiempo, por si el hilo marcador
    // tarda) ni segun la bandera paused_ (que solo baja el KitchenScheduler al registrar
    // restaurantResumed). Asi ningun retiro empieza dentro de la pausa ni antes de esa linea.
    cv_.wait(lock, [this, myTicket, &evaluatedAt] {
        if (stop_) {
            return true;
        }
        if (myTicket != nowServing_ || inUse_ >= slots_ || paused_) {
            return false;
        }
        evaluatedAt = nowMs();
        return !(clock_ != nullptr && schedule_.pausedAt(evaluatedAt));
    });

    if (stop_) {
        if (myTicket == nowServing_) {
            ++nowServing_;
            cv_.notify_all();
        }
        return false;
    }

    ++inUse_;
    ++nowServing_;
    if (onGranted) {
        onGranted(evaluatedAt);
    }
    cv_.notify_all();
    return true;
}

void Restaurant::release() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        --inUse_;
    }
    cv_.notify_all();
}

void Restaurant::stop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
}

void Restaurant::beginPause(std::int64_t t, EventLog& log) {
    const std::lock_guard<std::mutex> guard(mutex_);
    paused_ = true;
    ++pauses_;
    delayed_ += static_cast<std::int64_t>(nextTicket_ - nowServing_);
    Event e;
    e.name = EventName::kRestaurantPaused;
    e.restaurant = id_;
    log.write(e, t);
}

void Restaurant::endPause(std::int64_t t, EventLog& log) {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        paused_ = false;
        Event e;
        e.name = EventName::kRestaurantResumed;
        e.restaurant = id_;
        log.write(e, t);
    }
    cv_.notify_all();
}

int Restaurant::inUse() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return inUse_;
}

int Restaurant::waiting() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<int>(nextTicket_ - nowServing_);
}

std::int64_t Restaurant::pauses() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return pauses_;
}

std::int64_t Restaurant::delayedByPause() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return delayed_;
}
