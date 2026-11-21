#include "Quote.h"

#include <algorithm>

#include "accel/TravelTimeTable.h"

std::int64_t estimateDeliveryMs(Router& router, const CityGraph& graph,
                                const Courier& courier, const Order& order) {
  return estimateDeliveryMsWithTable(router, graph, courier, order, nullptr);
}

std::int64_t estimateDeliveryMsWithTable(Router& router, const CityGraph& graph,
                                         const Courier& courier, const Order& order,
                                         const accel::TravelTimeTable* table) {
  const NodeIndex at = graph.indexOf(currentNode(courier));
  const NodeIndex pickup = graph.indexOf(order.pickupNode);
  const NodeIndex dropoff = graph.indexOf(order.dropoffNode);
  if (at == kNoNode || pickup == kNoNode || dropoff == kNoNode) {
    return kUnreachable;
  }

  if (table != nullptr && table->nodeCount() == graph.nodeCount()) {
    const std::int64_t toPickupMs = table->travelTimeMs(at, pickup);
    const std::int64_t toDropoffMs = table->travelTimeMs(pickup, dropoff);
    if (toPickupMs == accel::TravelTimeTable::kNoRoute ||
        toDropoffMs == accel::TravelTimeTable::kNoRoute) {
      return kUnreachable;
    }
    return std::max(toPickupMs, order.prepMs) + toDropoffMs;
  }

  const Route toPickup = router.shortestPath(at, pickup);
  if (!toPickup.found) {
    return kUnreachable;
  }
  const Route toDropoff = router.shortestPath(pickup, dropoff);
  if (!toDropoff.found) {
    return kUnreachable;
  }

  const std::int64_t readyAt = std::max(toPickup.travelMs, order.prepMs);
  return readyAt + toDropoff.travelMs;
}
