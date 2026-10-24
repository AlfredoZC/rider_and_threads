#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "CourierActor.h"
#include "Config.h"
#include "SimContext.h"

class Fleet {
public:
    Fleet() = default;
    Fleet(const Config& cfg, const SimContext& ctx);
    ~Fleet();
    Fleet(const Fleet&) = delete;
    Fleet& operator=(const Fleet&) = delete;

    std::vector<CourierActor*> activeSnapshot() const;
    std::vector<CourierActor*> allSnapshot() const;

    void addAndStart(std::unique_ptr<CourierActor> actor);
    bool maybeSpawn(std::size_t waiting);
    void markRetired(CourierId id);
    std::int64_t reinforcementsJoined() const;
    std::int64_t reinforcementsRetired() const;

    void requestStopAll();
    void joinAll();

private:
    struct Entry {
        std::unique_ptr<CourierActor> actor;
        std::thread thread;
        bool active = true;
        bool reinforcement = false;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
    bool stopping_ = false;
    const SimContext* ctx_ = nullptr;
    int maxReinforcements_ = 0;
    int threshold_ = 0;
    std::int64_t idleMs_ = 0;
    std::string startNode_;
    int bagCapacity_ = 0;
    double breakdownProbability_ = 0.0;
    std::uint32_t seed_ = 0;
    std::uint64_t nextReinforcement_ = 0;
    std::int64_t joined_ = 0;
    std::int64_t retired_ = 0;
};
