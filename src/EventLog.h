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
constexpr const char *kSimulationStopping = "simulationStopping";
}

struct Event {
  const char *name = "";
  OrderId order = kNoOrder;
  CourierId courier = kNoCourier;
  std::string restaurant;
  std::string from;
  std::string to;

  std::string reason;
};

class EventLog {
public:
  EventLog(const std::string &path, const SimClock &clock);

  std::int64_t write(const Event &event);

private:
  const SimClock &clock_;
  std::ofstream out_;
  std::mutex mutex_;
};
