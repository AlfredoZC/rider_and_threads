#include "bench/CitySubgraph.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bench {

Config extractRegion(const Config& city, const std::string& startNodeId, std::size_t nodeCount) {
    if (nodeCount == 0) {
        throw std::invalid_argument("extractRegion: nodeCount debe ser >= 1");
    }
    std::unordered_map<std::string, std::size_t> indexById;
    indexById.reserve(city.nodes.size());
    for (std::size_t nodeIndex = 0; nodeIndex < city.nodes.size(); ++nodeIndex) {
        indexById.emplace(city.nodes[nodeIndex].id, nodeIndex);
    }
    const auto startNodeEntry = indexById.find(startNodeId);
    if (startNodeEntry == indexById.end()) {
        throw std::invalid_argument("extractRegion: el nodo de partida no existe en la ciudad");
    }
    if (nodeCount > city.nodes.size()) {
        throw std::invalid_argument("extractRegion: se piden " + std::to_string(nodeCount) + " nodos y la ciudad tiene " +
                                    std::to_string(city.nodes.size()));
    }

    // Adyacencia no dirigida por indice de nodo, con los vecinos en orden ascendente.
    std::vector<std::vector<std::size_t>> adjacency(city.nodes.size());
    for (const StreetConfig& street : city.streets) {
        const auto fromNode = indexById.find(street.from);
        const auto toNode = indexById.find(street.to);
        if (fromNode == indexById.end() || toNode == indexById.end() || fromNode->second == toNode->second) {
            continue;
        }
        adjacency[fromNode->second].push_back(toNode->second);
        adjacency[toNode->second].push_back(fromNode->second);
    }
    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    // BFS: se detiene al reunir nodeCount nodos.
    std::vector<char> chosen(city.nodes.size(), 0);
    std::vector<std::size_t> queue;
    queue.reserve(nodeCount);
    queue.push_back(startNodeEntry->second);
    chosen[startNodeEntry->second] = 1;
    for (std::size_t head = 0; head < queue.size() && queue.size() < nodeCount; ++head) {
        for (std::size_t next : adjacency[queue[head]]) {
            if (!chosen[next]) {
                chosen[next] = 1;
                queue.push_back(next);
                if (queue.size() == nodeCount) {
                    break;
                }
            }
        }
    }
    if (queue.size() < nodeCount) {
        throw std::invalid_argument("extractRegion: la componente conexa de '" + startNodeId + "' tiene solo " +
                                    std::to_string(queue.size()) + " nodos y se piden " + std::to_string(nodeCount));
    }

    Config region = city;
    region.nodes.clear();
    region.streets.clear();
    region.restaurants.clear();
    region.nodes.reserve(nodeCount);
    std::unordered_set<std::string> keptIds;
    keptIds.reserve(nodeCount);
    for (std::size_t nodeIndex = 0; nodeIndex < city.nodes.size(); ++nodeIndex) {
        if (chosen[nodeIndex]) {
            region.nodes.push_back(city.nodes[nodeIndex]);
            keptIds.insert(city.nodes[nodeIndex].id);
        }
    }
    for (const StreetConfig& street : city.streets) {
        if (keptIds.count(street.from) != 0 && keptIds.count(street.to) != 0) {
            region.streets.push_back(street);
        }
    }
    for (const RestaurantConfig& restaurant : city.restaurants) {
        if (keptIds.count(restaurant.node) != 0) {
            region.restaurants.push_back(restaurant);
        }
    }
    region.startNode = startNodeId;
    return region;
}

}  // namespace bench
