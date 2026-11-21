#include "bench/BenchGraph.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "bench/CitySubgraph.h"

namespace bench {

namespace {

// Variacion fija en [0, 1) a partir del indice del nodo (sin generador global ni semilla externa).
double jitterOf(std::size_t nodeIndex, std::size_t salt) {
    std::size_t mixed = (nodeIndex + 1) * 2654435761u + salt * 40503u;
    mixed ^= mixed >> 13;
    mixed *= 0x5bd1e995u;
    mixed ^= mixed >> 15;
    return static_cast<double>(mixed % 1000u) / 1000.0;
}

Config syntheticRegion(std::size_t size) {
    Config region;
    region.speedKmh = 30.0;
    region.startNode = "n0";
    const std::size_t width = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(size))));
    constexpr double kCellDegrees = 0.0009;  // ~100 m entre vecinos
    for (std::size_t node = 0; node < size; ++node) {
        NodeConfig nodeConfig;
        nodeConfig.id = "n" + std::to_string(node);
        nodeConfig.lat = -17.76 + kCellDegrees * (static_cast<double>(node / width) + 0.4 * jitterOf(node, 1));
        nodeConfig.lon = -63.19 + kCellDegrees * (static_cast<double>(node % width) + 0.4 * jitterOf(node, 2));
        region.nodes.push_back(nodeConfig);
    }
    std::size_t streetIndex = 0;
    auto addStreet = [&](std::size_t fromNode, std::size_t toNode) {
        StreetConfig street;
        street.id = "s" + std::to_string(streetIndex);
        street.from = "n" + std::to_string(fromNode);
        street.to = "n" + std::to_string(toNode);
        street.oneWay = (streetIndex % 5 == 4);  // una de cada cinco, de un solo sentido
        ++streetIndex;
        region.streets.push_back(street);
    };
    for (std::size_t node = 0; node < size; ++node) {
        if (node % width + 1 < width && node + 1 < size) addStreet(node, node + 1);  // vecino de la derecha
        if (node + width < size) addStreet(node, node + width);                      // vecino de abajo
    }
    if (size > 0) {
        RestaurantConfig restaurant;
        restaurant.id = "r0";
        restaurant.name = "sintetico";
        restaurant.node = "n0";
        region.restaurants.push_back(restaurant);
    }
    return region;
}

}  // namespace

Config cityRegionOrSynthetic(const BenchContext& context, std::size_t size) {
    if (size == 0) {
        throw std::invalid_argument("cityRegionOrSynthetic: size debe ser >= 1");
    }
    if (!context.config.nodes.empty()) {
        return extractRegion(context.config, context.config.startNode, size);
    }
    return syntheticRegion(size);
}

}  // namespace bench
