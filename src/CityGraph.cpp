#include "CityGraph.h"
#include <cmath>

namespace {

// Distancia sobre la superficie de la Tierra entre dos puntos en grados (haversine).
// Hace falta porque el config tiene latitud y longitud, no metros, y sin metros no hay
// ni camino más corto ni estimación de tiempo.
double haversineMeters(double lat1, double lon1, double lat2, double lon2) {
    const double kEarthRadiusM = 6371008.8;
    const double toRad = M_PI / 180.0;

    const double dLat = (lat2 - lat1) * toRad;
    const double dLon = (lon2 - lon1) * toRad;
    const double a = std::sin(dLat / 2) * std::sin(dLat / 2) +
                     std::cos(lat1 * toRad) * std::cos(lat2 * toRad) *
                         std::sin(dLon / 2) * std::sin(dLon / 2);
    return 2.0 * kEarthRadiusM * std::asin(std::sqrt(a));
}

}  // namespace

CityGraph::CityGraph(const Config& cfg) {
    // 1. Los nodos, en el orden del config: el índice de un nodo es su posición acá.
    nodes_.reserve(cfg.nodes.size());
    ids_.reserve(cfg.nodes.size());
    for (const NodeConfig& n : cfg.nodes) {
        ids_[n.id] = static_cast<NodeIndex>(nodes_.size());
        nodes_.push_back(Node{n.id, n.lat, n.lon});
    }

    adjacency_.resize(nodes_.size());

    // 2. Cada calle se convierte en aristas dirigidas. El tiempo de viaje se precalcula
    // porque speedKmh es igual para toda la flota y no cambia durante la simulación.
    const double msPerMeter = 3600.0 / cfg.speedKmh;   // 30 km/h -> 120 ms por metro

    for (const StreetConfig& s : cfg.streets) {
        const NodeIndex a = indexOf(s.from);
        const NodeIndex b = indexOf(s.to);
        if (a == kNoNode || b == kNoNode || a == b) {
            continue;   // loadConfig ya lo garantiza; esto es solo para no construir basura
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
