// Pruebas de Courier: constructor consistente, limite de la bolsa y estados.
#include "Courier.h"

#include <cassert>
#include <iostream>

int main() {
    Courier c(1, "n311", 3);

    // un objeto recien creado tiene que cumplir ya su invariante
    assert(c.id == 1);
    assert(c.state == CourierState::idle);
    assert(c.fromNode == "n311");
    assert(c.toNode == "n311");
    assert(c.progress == 0.0);
    assert(c.delivered == 0);
    assert(c.bag.empty());
    assert(currentNode(c) == "n311");
    assert(c.bag.capacity() >= 3);      // el reserve del constructor

    // el limite de capacidad
    assert(hasRoom(c));
    c.bag.push_back(10);
    c.bag.push_back(11);
    assert(hasRoom(c));
    c.bag.push_back(12);
    assert(!hasRoom(c));                // justo en el limite

    // estados
    assert(!isBusy(CourierState::idle));
    assert(!isBusy(CourierState::brokenDown));
    assert(isBusy(CourierState::toPickup));
    assert(isBusy(CourierState::atRestaurant));
    assert(isBusy(CourierState::toDropoff));

    // los nombres que se escriben en el log
    assert(std::string(toString(CourierState::idle)) == "idle");
    assert(std::string(toString(CourierState::atRestaurant)) == "atRestaurant");
    assert(std::string(toString(CourierState::brokenDown)) == "brokenDown");

    // interpolacion: a mitad de calle sigue contando el nodo del que salio
    c.fromNode = "n311";
    c.toNode = "n312";
    c.progress = 0.5;
    assert(currentNode(c) == "n311");
    c.progress = 1.0;
    assert(currentNode(c) == "n312");

    std::cout << "  courier ok\n";
}
