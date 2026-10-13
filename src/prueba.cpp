#include "Order.h"
#include <cassert>
#include <iostream>

int main() {
    Order o{};
    o.id = 1;
    o.restaurantId = "r0";
    o.pickupNode = "n15";
    o.dropoffNode = "n420";
    o.createdMs = 0;

    assert(o.state == OrderState::created);
    assert(!isFinalState(o.state));
    assert(deliveryMs(o) == kNoTime);

    // el camino feliz
    assert(canTransition(OrderState::created, OrderState::assigned));
    assert(canTransition(OrderState::assigned, OrderState::pickedUp));
    assert(canTransition(OrderState::pickedUp, OrderState::delivered));

    // lo que el brief prohíbe
    assert(!canTransition(OrderState::delivered, OrderState::pickedUp));
    assert(!canTransition(OrderState::delivered, OrderState::delivered));
    assert(!canTransition(OrderState::rejected, OrderState::assigned));
    assert(!canTransition(OrderState::created, OrderState::delivered));
    assert(!canTransition(OrderState::created, OrderState::pickedUp));

    // reasignación por avería, y parada
    assert(canTransition(OrderState::assigned, OrderState::created));
    assert(canTransition(OrderState::pickedUp, OrderState::pending));
    assert(!canTransition(OrderState::pending, OrderState::delivered));

    o.state = OrderState::delivered;
    o.deliveredMs = 930000;
    assert(deliveryMs(o) == 930000);

    std::cout << "estado=" << toString(o.state)
              << " razon=" << toString(RejectReason::queueFull)
              << " entrega=" << deliveryMs(o) << "ms\n";
    std::cout << "TODO OK\n";
}
