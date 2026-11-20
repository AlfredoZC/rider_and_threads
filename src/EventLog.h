#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

#include "Ids.h"
#include "SimClock.h"

namespace EventName {
constexpr const char *kOrderCreated = "orderCreated";
constexpr const char *kOrderAssigned = "orderAssigned";
constexpr const char *kPickupStarted = "pickupStarted";
constexpr const char *kOrderPickedUp = "orderPickedUp";
constexpr const char *kOrderDelivered = "orderDelivered";
constexpr const char *kOrderRejected = "orderRejected";
constexpr const char *kCourierBreakdown = "courierBreakdown";
constexpr const char *kCourierJoined = "courierJoined";
constexpr const char *kCourierRetired = "courierRetired";
constexpr const char *kOrderReassigned = "orderReassigned";
constexpr const char *kCourierBreakStarted = "courierBreakStarted";
constexpr const char *kCourierBreakEnded = "courierBreakEnded";
constexpr const char *kSimulationStopping = "simulationStopping";
constexpr const char *kRestaurantPaused = "restaurantPaused";
constexpr const char *kRestaurantResumed = "restaurantResumed";
}

struct Event {
  const char *name = "";
  OrderId order = kNoOrder;
  CourierId courier = kNoCourier;
  std::string restaurant;
  std::string from;
  std::string to;

  std::string reason;
  int urgent = -1;
};

class EventLog {
public:
  EventLog(const std::string &path, const SimClock &clock);

  std::int64_t write(const Event &event);
  // Escribe con un t explicito (instante exacto del calendario de pausas o de la entrada en fila).
  std::int64_t write(const Event &event, std::int64_t t);

private:
  std::int64_t writeAt(const Event &event, const std::int64_t *explicitT);

  const SimClock &clock_;
  std::ofstream out_;
  std::mutex mutex_;
};
