#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

struct Bounds {
    double north = 0.0;
    double south = 0.0;
    double west = 0.0;
    double east = 0.0;
};

struct NodeConfig {
    std::string id;
    double lat = 0.0;
    double lon = 0.0;
};

struct StreetConfig {
    std::string id;
    std::string from;
    std::string to;
    bool oneWay = false;
};

struct RestaurantConfig {
    std::string id;
    std::string name;
    std::string node;
    int pickupSlots = 1;
    std::int64_t prepMinMs = 0;
    std::int64_t prepMaxMs = 0;
};
struct ReinforcementConfig {
    int max = 0;
    int threshold = 0;
    std::int64_t idleMs = 0;
};
struct Config {
    std::filesystem::path imagePath;
    std::string attribution;
    Bounds bounds;

    std::vector<NodeConfig> nodes;
    std::vector<StreetConfig> streets;
    std::vector<RestaurantConfig> restaurants;

    int couriers = 0;
    int bagCapacity = 0;
    double speedKmh = 0.0;
    std::string startNode;

    std::int64_t meanIntervalMs = 0;
    int burstMax = 0;
    int maxPending = 0;
    std::uint32_t seed = 0;

    std::int64_t quoteTimeoutMs = 0;
    std::int64_t acceptTimeoutMs = 0;

    double breakdownProbability = 0.0;

    std::int64_t durationS = 0;
    double timeScale = 1.0;

    ReinforcementConfig reinforcements;
};

class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

Config loadConfig(const std::string& path);
