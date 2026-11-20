#include "KitchenScheduler.h"

#include <algorithm>
#include <chrono>

std::int64_t KitchenScheduler::Slot::nextMs() const {
    const KitchenSchedule& s = restaurant->schedule();
    return paused ? s.endOf(k) : s.startOf(k);
}

KitchenScheduler::KitchenScheduler(std::vector<Restaurant*> restaurants, const SimClock& clock,
                                   EventLog& log)
    : clock_(clock), log_(log) {
    for (Restaurant* r : restaurants) {
        if (r != nullptr && r->schedule().enabled()) {
            slots_.push_back(Slot{r});
        }
    }
}

KitchenScheduler::~KitchenScheduler() {
    requestStop();
    join();
}

void KitchenScheduler::start() {
    if (slots_.empty() || thread_.joinable()) {
        return;
    }
    thread_ = std::thread(&KitchenScheduler::run, this);
}

void KitchenScheduler::requestStop() {
    // El flag es atomico: se publica sin tomar el candado, asi un hilo que gira sobre mutex_
    // jamas puede impedir (por inanicion del mutex) que otro hilo pida parar.
    stop_.store(true);
    {
        const std::lock_guard<std::mutex> guard(mutex_);
    }
    cv_.notify_all();
}

void KitchenScheduler::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

void KitchenScheduler::run() {
    // Tope de una espera: si el proximo limite esta mas lejos, se duerme esto desde ahora y se
    // re-evalua. Mantiene el plazo absoluto siempre cercano y representable.
    constexpr std::chrono::microseconds kMaxWait = std::chrono::hours(1);

    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_.load()) {
        Slot* next = &slots_.front();
        for (Slot& s : slots_) {
            if (s.nextMs() < next->nextMs()) {
                next = &s;
            }
        }
        const std::int64_t limit = next->nextMs();
        if (limit == KitchenSchedule::kNever) {
            return;  // ningun limite restante es representable: no ocurre durante la corrida
        }
        const std::int64_t now = clock_.nowMs();
        if (now < limit) {
            const std::chrono::microseconds wait =
                std::min(clock_.realDuration(limit - now), kMaxWait);
            const auto deadline = std::chrono::steady_clock::now() + wait;
            cv_.wait_until(lock, deadline, [this] { return stop_.load(); });
            continue;
        }
        // Jerarquia de candados: restaurante -> log. El candado propio se suelta antes de
        // tomar el del restaurante; los campos de los slots solo los toca este hilo.
        lock.unlock();
        if (next->paused) {
            next->restaurant->endPause(limit, log_);
            next->paused = false;
            ++next->k;
        } else {
            next->restaurant->beginPause(limit, log_);
            next->paused = true;
        }
        lock.lock();
    }
}
