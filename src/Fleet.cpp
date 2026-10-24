#include "Fleet.h"

#include <future>
#include <stdexcept>
#include <utility>

Fleet::Fleet(const Config& cfg, const SimContext& ctx)
    : ctx_(&ctx),
      maxReinforcements_(cfg.reinforcements.max),
      threshold_(cfg.reinforcements.threshold),
      idleMs_(cfg.reinforcements.idleMs),
      startNode_(cfg.startNode),
      bagCapacity_(cfg.bagCapacity),
      breakdownProbability_(cfg.breakdownProbability),
      seed_(cfg.seed) {}

Fleet::~Fleet() {
    joinAll();
}

std::vector<CourierActor*> Fleet::activeSnapshot() const {
    std::vector<CourierActor*> actors;
    const std::lock_guard<std::mutex> guard(mutex_);
    actors.reserve(entries_.size());
    for (const Entry& entry : entries_) {
        if (entry.active) {
            actors.push_back(entry.actor.get());
        }
    }
    return actors;
}

std::vector<CourierActor*> Fleet::allSnapshot() const {
    std::vector<CourierActor*> actors;
    const std::lock_guard<std::mutex> guard(mutex_);
    actors.reserve(entries_.size());
    for (const Entry& entry : entries_) {
        actors.push_back(entry.actor.get());
    }
    return actors;
}

void Fleet::addAndStart(std::unique_ptr<CourierActor> actor) {
    if (!actor) {
        throw std::invalid_argument("Fleet::addAndStart necesita un actor");
    }
    const std::lock_guard<std::mutex> guard(mutex_);
    if (stopping_) {
        throw std::logic_error("no se pueden agregar couriers durante la parada");
    }

    CourierActor* const raw = actor.get();
    entries_.emplace_back();
    Entry& entry = entries_.back();
    entry.actor = std::move(actor);
    try {
        entry.thread = std::thread(&CourierActor::run, raw);
    } catch (...) {
        entries_.pop_back();
        throw;
    }
}

bool Fleet::maybeSpawn(std::size_t waiting) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (ctx_ == nullptr || stopping_ || maxReinforcements_ == 0) {
        return false;
    }
    if (waiting < static_cast<std::size_t>(threshold_)) {
        thresholdArmed_ = true;
        return false;
    }
    if (!thresholdArmed_) {
        return false;
    }

    int active = 0;
    for (const Entry& entry : entries_) {
        if (entry.reinforcement && entry.active) {
            ++active;
        }
    }
    if (active >= maxReinforcements_) {
        thresholdArmed_ = false;
        return false;
    }

    const CourierId id = reinforcementId(nextReinforcement_);
    auto actor = std::make_unique<CourierActor>(
        id, startNode_, bagCapacity_, *ctx_, breakdownProbability_, seed_, true, idleMs_);
    actor->setFleet(this);
    CourierActor* const raw = actor.get();
    std::promise<bool> startGate;
    auto permission = startGate.get_future();
    entries_.emplace_back();
    Entry& entry = entries_.back();
    entry.actor = std::move(actor);
    entry.reinforcement = true;

    try {
        entry.thread = std::thread([raw, permission = std::move(permission)]() mutable {
            if (permission.get()) {
                raw->run();
            }
        });
    } catch (...) {
        entries_.pop_back();
        thresholdArmed_ = true;
        throw;
    }

    try {
        Event joined;
        joined.name = EventName::kCourierJoined;
        joined.courier = id;
        ctx_->log.write(joined);
    } catch (...) {
        startGate.set_value(false);
        std::thread failed = std::move(entry.thread);
        entries_.pop_back();
        thresholdArmed_ = true;
        lock.unlock();
        failed.join();
        throw;
    }
    ++nextReinforcement_;
    ++joined_;
    thresholdArmed_ = false;
    startGate.set_value(true);
    return true;
}

void Fleet::markRetired(CourierId id) {
    const std::lock_guard<std::mutex> guard(mutex_);
    for (Entry& entry : entries_) {
        if (entry.actor->id() == id && entry.reinforcement && entry.active) {
            entry.active = false;
            ++retired_;
            Event retired;
            retired.name = EventName::kCourierRetired;
            retired.courier = id;
            ctx_->log.write(retired);
            return;
        }
    }
}

std::int64_t Fleet::reinforcementsJoined() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return joined_;
}

std::int64_t Fleet::reinforcementsRetired() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return retired_;
}

void Fleet::requestStopAll() {
    std::vector<CourierActor*> actors;
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopping_ = true;
        actors.reserve(entries_.size());
        for (const Entry& entry : entries_) {
            actors.push_back(entry.actor.get());
        }
    }
    for (CourierActor* actor : actors) {
        actor->requestStop();
    }
}

void Fleet::joinAll() {
    requestStopAll();
    std::vector<std::thread> threads;
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        threads.reserve(entries_.size());
        for (Entry& entry : entries_) {
            if (entry.thread.joinable()) {
                threads.push_back(std::move(entry.thread));
            }
        }
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}
