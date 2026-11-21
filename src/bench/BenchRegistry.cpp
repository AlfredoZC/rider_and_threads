#include "bench/BenchRegistry.h"

#include <algorithm>

namespace bench {

BenchRegistry& BenchRegistry::instance() {
    static BenchRegistry registry;
    return registry;
}

void BenchRegistry::add(const std::string& name, AppFactory factory) {
    entries_.emplace_back(name, std::move(factory));
    std::sort(entries_.begin(), entries_.end(), [](const auto& left, const auto& right) { return left.first < right.first; });
}

std::vector<std::string> BenchRegistry::names() const {
    std::vector<std::string> appNames;
    appNames.reserve(entries_.size());
    for (const auto& entry : entries_) {
        appNames.push_back(entry.first);
    }
    return appNames;
}

std::unique_ptr<BenchApp> BenchRegistry::create(const std::string& name) const {
    for (const auto& entry : entries_) {
        if (entry.first == name) {
            return entry.second();
        }
    }
    return nullptr;
}

std::vector<std::unique_ptr<BenchApp>> makeApps(const std::vector<std::string>& names) {
    const BenchRegistry& registry = BenchRegistry::instance();
    std::vector<std::string> wanted = names.empty() ? registry.names() : names;
    std::vector<std::unique_ptr<BenchApp>> apps;
    for (const std::string& appName : wanted) {
        if (auto app = registry.create(appName)) {
            apps.push_back(std::move(app));
        }
    }
    return apps;
}

}  // namespace bench
