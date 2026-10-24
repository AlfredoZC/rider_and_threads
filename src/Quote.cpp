#include "Quote.h"

#include <algorithm>

std::int64_t estimateDeliveryMs(Router& router, const CityGraph& graph,
                                const Courier& courier, const Order& order) {
  const NodeIndex at = graph.indexOf(currentNode(courier));
  const NodeIndex pickup = graph.indexOf(order.pickupNode);
  const NodeIndex dropoff = graph.indexOf(order.dropoffNode);
  if (at == kNoNode || pickup == kNoNode || dropoff == kNoNode) {
    return kUnreachable;
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
