#include "Courier.h"

#include <cassert>
#include <iostream>

int main() {
    Courier c(1, "n311", 3);

    assert(c.id == 1);
    assert(c.state == CourierState::idle);
    assert(c.fromNode == "n311");
    assert(c.toNode == "n311");
    assert(c.segmentStartMs == 0);
    assert(c.segmentEndMs == 0);
    assert(c.delivered == 0);
    assert(c.bag.empty());
    assert(currentNode(c) == "n311");
    assert(c.bag.capacity() >= 3);

    assert(hasRoom(c));
    c.bag.push_back(10);
    c.bag.push_back(11);
    assert(hasRoom(c));
    c.bag.push_back(12);
    assert(!hasRoom(c));

    assert(!isBusy(CourierState::idle));
    assert(!isBusy(CourierState::brokenDown));
    assert(isBusy(CourierState::toPickup));
    assert(isBusy(CourierState::atRestaurant));
    assert(isBusy(CourierState::toDropoff));

    assert(std::string(toString(CourierState::idle)) == "idle");
    assert(std::string(toString(CourierState::atRestaurant)) == "atRestaurant");
    assert(std::string(toString(CourierState::brokenDown)) == "brokenDown");

    c.fromNode = "n311";
    c.toNode = "n312";
    c.segmentStartMs = 1000;
    c.segmentEndMs = 3000;
    assert(currentNode(c) == "n311");
    assert(positionRatio(c, 1000) == 0.0);
    assert(positionRatio(c, 2000) == 0.5);
    assert(positionRatio(c, 3000) == 1.0);
    assert(positionRatio(c, 500) == 0.0);
    assert(positionRatio(c, 9999) == 1.0);

    c.fromNode = "n312";
    c.toNode = "n312";
    c.segmentStartMs = 3000;
    c.segmentEndMs = 3000;
    assert(currentNode(c) == "n312");
    assert(positionRatio(c, 0) == 1.0);

    std::cout << "  courier ok\n";
}
