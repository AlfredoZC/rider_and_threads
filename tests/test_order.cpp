#include "Order.h"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    Order o{};
    o.id = 1;
    o.restaurantId = "r0";
    o.pickupNode = "n15";
    o.dropoffNode = "n420";
    o.createdMs = 0;

    assert(o.state == OrderState::created);
    assert(o.assignedCourier == kNoCourier);
    assert(o.deliveredMs == kNoTime);
    assert(!o.reassigned);
    assert(deliveryMs(o) == kNoTime);

    assert(!isFinalState(OrderState::created));
    assert(!isFinalState(OrderState::assigned));
    assert(!isFinalState(OrderState::pickedUp));
    assert(isFinalState(OrderState::delivered));
    assert(isFinalState(OrderState::rejected));
    assert(isFinalState(OrderState::pending));

    assert(canTransition(OrderState::created, OrderState::assigned));
    assert(canTransition(OrderState::assigned, OrderState::pickedUp));
    assert(canTransition(OrderState::pickedUp, OrderState::delivered));

    assert(!canTransition(OrderState::delivered, OrderState::delivered));
    assert(!canTransition(OrderState::delivered, OrderState::pickedUp));
    assert(!canTransition(OrderState::delivered, OrderState::pending));
    assert(!canTransition(OrderState::rejected, OrderState::assigned));
    assert(!canTransition(OrderState::pending, OrderState::delivered));
    assert(!canTransition(OrderState::created, OrderState::pickedUp));
    assert(!canTransition(OrderState::created, OrderState::delivered));
    assert(!canTransition(OrderState::assigned, OrderState::delivered));
    assert(!canTransition(OrderState::assigned, OrderState::rejected));

    assert(canTransition(OrderState::assigned, OrderState::created));

    assert(canTransition(OrderState::created, OrderState::pending));
    assert(canTransition(OrderState::assigned, OrderState::pending));
    assert(canTransition(OrderState::pickedUp, OrderState::pending));

    assert(std::string(toString(OrderState::pickedUp)) == "pickedUp");
    assert(std::string(toString(OrderState::delivered)) == "delivered");
    assert(std::string(toString(RejectReason::queueFull)) == "queueFull");
    assert(std::string(toString(RejectReason::unreachable)) == "unreachable");
    assert(std::string(toString(RejectReason::noCourier)) == "noCourier");

    o.state = OrderState::delivered;
    o.deliveredMs = 930000;
    assert(deliveryMs(o) == 930000);

    std::cout << "  order ok\n";
}
