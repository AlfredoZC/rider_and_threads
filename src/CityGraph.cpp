#include "CityGraph.h"
#include <cmath>

namespace {

constexpr double kPi = 3.14159265358979323846;

double haversineMeters(double lat1, double lon1, double lat2, double lon2) {
    const double kEarthRadiusM = 6371008.8;
    const double toRad = kPi / 180.0;

    const double dLat = (lat2 - lat1) * toRad;
    const double dLon = (lon2 - lon1) * toRad;
    const double a = std::sin(dLat / 2) * std::sin(dLat / 2) +
                     std::cos(lat1 * toRad) * std::cos(lat2 * toRad) *
                         std::sin(dLon / 2) * std::sin(dLon / 2);
    return 2.0 * kEarthRadiusM * std::asin(std::sqrt(a));
}

}

CityGraph::CityGraph(const Config& cfg) {
    nodes_.reserve(cfg.nodes.size());
    ids_.reserve(cfg.nodes.size());
    for (const NodeConfig& n : cfg.nodes) {
        ids_[n.id] = static_cast<NodeIndex>(nodes_.size());
        nodes_.push_back(Node{n.id, n.lat, n.lon});
    }

    adjacency_.resize(nodes_.size());

    const double msPerMeter = 3600.0 / cfg.speedKmh;

    for (const StreetConfig& s : cfg.streets) {
        const NodeIndex a = indexOf(s.from);
        const NodeIndex b = indexOf(s.to);
        if (a == kNoNode || b == kNoNode || a == b) {
            continue;
        }

        const double meters = haversineMeters(nodes_[a].lat, nodes_[a].lon,
                                              nodes_[b].lat, nodes_[b].lon);
        const std::int64_t travelMs = static_cast<std::int64_t>(meters * msPerMeter);

        adjacency_[a].push_back(Edge{b, meters, travelMs});
        ++edgeCount_;

        if (!s.oneWay) {
            adjacency_[b].push_back(Edge{a, meters, travelMs});
            ++edgeCount_;
        }
    }
}

NodeIndex CityGraph::indexOf(const std::string& id) const {
    const auto it = ids_.find(id);
    return it == ids_.end() ? kNoNode : it->second;
}
