#include "Quote.h"

#include <algorithm>

namespace {

// El courier llega al restaurante, espera a que la comida este lista si hace falta, y lleva el pedido.
std::int64_t deliveryMs(std::int64_t toPickupMs, std::int64_t toDropoffMs, const Order& order) {
  const std::int64_t readyAt = std::max(toPickupMs, order.prepMs);
  return readyAt + toDropoffMs;
}

}  // namespace

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

  return deliveryMs(toPickup.travelMs, toDropoff.travelMs, order);
}

std::int64_t estimateDeliveryMs(const accel::TravelTimeTable& table, const CityGraph& graph,
                                const Courier& courier, const Order& order) {
  const NodeIndex at = graph.indexOf(currentNode(courier));
  const NodeIndex pickup = graph.indexOf(order.pickupNode);
  const NodeIndex dropoff = graph.indexOf(order.dropoffNode);
  if (at == kNoNode || pickup == kNoNode || dropoff == kNoNode) {
    return kUnreachable;
  }

  const std::int64_t toPickupMs = table.travelTimeMs(at, pickup);  // -1 = sin camino
  if (toPickupMs < 0) {
    return kUnreachable;
  }
  const std::int64_t toDropoffMs = table.travelTimeMs(pickup, dropoff);
  if (toDropoffMs < 0) {
    return kUnreachable;
  }

  return deliveryMs(toPickupMs, toDropoffMs, order);
}
