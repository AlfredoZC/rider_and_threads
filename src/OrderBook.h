#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <condition_variable>

#include "EventLog.h"
#include "Ids.h"
#include "Order.h"
#include "SimClock.h"

struct OrderStats {
    std::int64_t created = 0;
    std::int64_t delivered = 0;
    std::int64_t rejected = 0;
    std::int64_t reassigned = 0;
    std::int64_t breakdowns = 0;
    std::int64_t pending = 0;
    std::int64_t avgDeliveryMs = 0;
    std::int64_t maxDeliveryMs = 0;
    std::int64_t urgentCreated = 0;
    std::int64_t urgentDelivered = 0;
    std::int64_t avgUrgentDeliveryMs = 0;
};

class OrderBook {
public:
    OrderBook(int maxPending, const SimClock& clock, EventLog& log);

    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;

    bool admit(Order order);
    

    bool reject(OrderId id, RejectReason reason);

    bool assign(OrderId id, CourierId courier);
    // Como assign, pero si el pedido ya vencio (now - createdMs > acceptTimeoutMs) lo
    // rechaza con noCourier bajo el mismo candado y devuelve expired: nunca queda un
    // orderAssigned fuera de plazo.
    enum class AssignResult { assigned, expired, failed };
    AssignResult assignUnlessExpired(OrderId id, CourierId courier,
                                     std::int64_t acceptTimeoutMs);
    bool pickedUp(OrderId id, CourierId courier);
    bool delivered(OrderId id, CourierId courier);
    bool waitForUnassigned(std::chrono::milliseconds timeout);
    std::uint64_t workVersion() const;
    bool waitForChange(std::uint64_t& observed, std::chrono::milliseconds timeout);
    bool waitForChange(std::uint64_t& observed,
                       std::chrono::steady_clock::time_point deadline);
    std::optional<Order> pickNextUnassigned(std::int64_t agingMs,
                                            const std::unordered_set<OrderId>& skip) const;
    std::int64_t rejectExpired(std::int64_t acceptTimeoutMs);
    std::optional<std::int64_t> nextDeadlineSimMs(std::int64_t acceptTimeoutMs,
                                                  std::int64_t agingMs) const;
    std::size_t waitingCount() const;
    void notifyCapacityChanged();
    void stopWaiting();
    bool unassign(OrderId id, bool countAsReassigned);

    std::int64_t markRemainingAsPending();

    bool handOver(OrderId id, CourierId from, CourierId to);
    bool handOverBatch(const std::vector<OrderId>& ids, CourierId from, CourierId to);

    void recordBreakdown(CourierId courier);

    std::optional<Order> find(OrderId id) const;

    std::vector<OrderId> unassignedIds() const;

    OrderStats stats() const;

private:
    bool setState(Order& order, OrderState to);
    bool assignLocked(OrderId id, CourierId courier);

    const std::size_t maxPending_;
    const SimClock& clock_;
    EventLog& log_;

    mutable std::mutex mutex_;
    std::condition_variable work_;
    std::uint64_t workVersion_ = 0;
    bool stopWaiting_ = false;
    std::unordered_map<OrderId, Order> orders_;
    std::size_t pending_ = 0;
    std::int64_t markedPending_ = 0;
    OrderStats stats_;
    std::int64_t deliverySumMs_ = 0;
    std::int64_t urgentDeliverySumMs_ = 0;
};
