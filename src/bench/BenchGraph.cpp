#include "bench/BenchGraph.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "bench/CitySubgraph.h"

namespace bench {

namespace {

// splitmix64: numeros pseudoaleatorios deterministas segun la semilla, sin depender de la biblioteca.
class SeededRandom {
public:
    explicit SeededRandom(std::uint32_t seed) : state_(seed) {}

    std::uint64_t next() {
        state_ += 0x9E3779B97F4A7C15ull;
        std::uint64_t mixed = state_;
        mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
        mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
        return mixed ^ (mixed >> 31);
    }

    // Entero en [0, bound).
    std::uint64_t below(std::uint64_t bound) { return next() % bound; }

private:
    std::uint64_t state_;
};

Config syntheticGraph(std::size_t nodeCount, std::uint32_t seed) {
    Config graph;
    graph.speedKmh = 20.0;
    const std::size_t columns = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(nodeCount))));
    constexpr double kBlockDegrees = 0.001;  // ~110 m entre intersecciones vecinas
    SeededRandom random(seed);

    graph.nodes.reserve(nodeCount);
    for (std::size_t node = 0; node < nodeCount; ++node) {
        const double rowJitter = static_cast<double>(random.below(1000)) / 1000.0 * 0.4 * kBlockDegrees;
        const double columnJitter = static_cast<double>(random.below(1000)) / 1000.0 * 0.4 * kBlockDegrees;
        graph.nodes.push_back(NodeConfig{"s" + std::to_string(node),
                                         -17.77 + static_cast<double>(node / columns) * kBlockDegrees + rowJitter,
                                         -63.19 + static_cast<double>(node % columns) * kBlockDegrees + columnJitter});
    }
    graph.startNode = graph.nodes.front().id;

    std::size_t streetCount = 0;
    auto addStreet = [&](std::size_t firstNode, std::size_t secondNode) {
        const std::size_t sinkNode = nodeCount - 1;  // el ultimo nodo es un sumidero: se llega a el pero no se sale
        const bool intoSink = secondNode == sinkNode || firstNode == sinkNode;
        const bool oneWay = intoSink || random.below(100) < 35;
        const bool reversed = intoSink ? firstNode == sinkNode : (oneWay && random.below(2) == 1);  // sentido de las calles de un sentido
        const std::size_t fromNode = reversed ? secondNode : firstNode;
        const std::size_t toNode = reversed ? firstNode : secondNode;
        graph.streets.push_back(StreetConfig{"street" + std::to_string(streetCount++), graph.nodes[fromNode].id,
                                             graph.nodes[toNode].id, oneWay});
    };
    for (std::size_t node = 0; node < nodeCount; ++node) {
        if ((node % columns) + 1 < columns && node + 1 < nodeCount) {
            addStreet(node, node + 1);  // hacia la derecha
        }
        if (node + columns < nodeCount) {
            addStreet(node, node + columns);  // hacia abajo
        }
    }
    return graph;
}

}  // namespace

Config cityRegionOrSynthetic(const Config& city, std::size_t nodeCount, std::uint32_t seed, bool* usedCity) {
    if (nodeCount == 0) {
        throw std::invalid_argument("cityRegionOrSynthetic: nodeCount debe ser >= 1");
    }
    if (city.speedKmh > 0.0 && city.nodes.size() >= nodeCount && !city.startNode.empty()) {
        try {
            Config region = extractRegion(city, city.startNode, nodeCount);
            if (usedCity != nullptr) *usedCity = true;
            return region;
        } catch (const std::invalid_argument&) {
            // la ciudad no tiene una componente conexa tan grande: respaldo sintetico
        }
    }
    if (usedCity != nullptr) *usedCity = false;
    return syntheticGraph(nodeCount, seed);
}

}  // namespace bench
