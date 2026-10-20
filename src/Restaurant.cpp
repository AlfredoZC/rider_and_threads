#include "Restaurant.h"

#include <utility>

Restaurant::Restaurant(std::string id, int slots) : id_(std::move(id)), slots_(slots) {}

bool Restaurant::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);

    // Sacar numero es lo primero: fija el orden de llegada en el instante en que el
    // courier llega, no en el instante en que consigue lugar.
    const std::uint64_t myTicket = nextTicket_++;

    cv_.wait(lock, [this, myTicket] {
        return stop_ || (myTicket == nowServing_ && inUse_ < slots_);
    });

    if (stop_) {
        // Hay que ceder el turno igual. Si este courier se va sin avanzar nowServing_,
        // los que estan detras esperarian un numero que ya no va a atender nadie, y el
        // join() del cierre no volveria nunca.
        if (myTicket == nowServing_) {
            ++nowServing_;
            cv_.notify_all();
        }
        return false;
    }

    ++inUse_;
    ++nowServing_;   // el siguiente de la fila ya puede intentar
    cv_.notify_all();
    return true;
}

void Restaurant::release() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        --inUse_;
    }
    cv_.notify_all();   // fuera del candado: quien despierta no lo encuentra ocupado
}

void Restaurant::stop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stop_ = true;
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
