#pragma once

#include <cstdint>
#include <vector>

#include "CityGraph.h"

struct Route {
    bool found = false;
    std::int64_t travelMs = 0;
    std::vector<NodeIndex> nodes;
};

class Router {
public:
    explicit Router(const CityGraph& graph);

    Route shortestPath(NodeIndex from, NodeIndex to);

private:
    const CityGraph& graph_;
    std::vector<std::int64_t> dist_;
    std::vector<NodeIndex> prev_;
};
