// Pruebas de OrderFactory: ids unicos, ordenes coherentes al nacer y reproducibilidad.
#include "OrderFactory.h"

#include <cassert>
#include <iostream>
#include <set>

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    OrderFactory f(cfg, cfg.seed);

    std::set<OrderId> ids;
    const int kHowMany = 10000;
    for (int i = 0; i < kHowMany; ++i) {
        const Order o = f.create(i * 100);

        assert(ids.insert(o.id).second);            // nunca repite un id
        assert(o.id != kNoOrder);
        assert(o.state == OrderState::created);
        assert(o.createdMs == i * 100);
        assert(o.dropoffNode != o.pickupNode);
        assert(o.assignedCourier == kNoCourier);
        assert(o.deliveredMs == kNoTime);
        assert(o.rejectReason == RejectReason::none);

        // la orden nace coherente: su pickupNode es el nodo del restaurante que dice
        bool found = false;
        for (const RestaurantConfig& r : cfg.restaurants) {
            if (r.id == o.restaurantId) {
                assert(o.pickupNode == r.node);
                assert(o.prepMs >= r.prepMinMs && o.prepMs <= r.prepMaxMs);
                found = true;
            }
        }
        assert(found);
    }
    assert(ids.size() == static_cast<std::size_t>(kHowMany));

    // reproducibilidad: misma semilla, misma secuencia. Es lo que permite repetir un
    // bug de concurrencia que aparece una corrida entre veinte.
    OrderFactory a(cfg, 42), b(cfg, 42), c(cfg, 43);
    bool anyDifferent = false;
    for (int i = 0; i < 50; ++i) {
        const Order oa = a.create(0), ob = b.create(0), oc = c.create(0);
        assert(oa.id == ob.id);
        assert(oa.restaurantId == ob.restaurantId);
        assert(oa.dropoffNode == ob.dropoffNode);
        assert(oa.prepMs == ob.prepMs);
        if (oc.dropoffNode != oa.dropoffNode || oc.restaurantId != oa.restaurantId) {
            anyDifferent = true;
        }
    }
    assert(anyDifferent);                           // otra semilla, otra secuencia

    std::cout << "  factory ok (" << kHowMany << " ids unicos)\n";
}
