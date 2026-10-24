#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "CourierActor.h"

namespace {

Order makeOrder(OrderId id, const std::string& pickup, const std::string& dropoff) {
    Order o;
    o.id = id;
    o.restaurantId = "r0";
    o.pickupNode = pickup;
    o.dropoffNode = dropoff;
    return o;
}

double ratioDe(const Courier& c, std::int64_t nowMs) { return positionRatio(c, nowMs); }

}

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    const std::string restaurantNode = cfg.restaurants.front().node;
    const std::string clientNode = cfg.nodes.back().id;

    const SimClock clock(60.0);
    EventLog log("tests/bin/render.log", clock);
    OrderBook book(50, clock, log);
    Restaurants restaurants;
    for (const RestaurantConfig& r : cfg.restaurants) {
        restaurants.emplace(r.id, std::make_unique<Restaurant>(r.id, r.pickupSlots));
    }
    const SimContext ctx{graph, clock, log, book, restaurants};

    CourierActor courier(1, cfg.startNode, 3, ctx);
    const OrderId id = 1;
    assert(book.admit(makeOrder(id, restaurantNode, clientNode)));
    assert(book.assign(id, 1));
    assert(courier.offer(*book.find(id)));

    std::thread worker(&CourierActor::run, &courier);

    struct Muestra {
        double ratio;
        std::string tramo;
        std::int64_t simMs;
        std::int64_t duracionMs;
    };
    std::vector<Muestra> muestras;
    int muestrasEnMovimiento = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count() < 3000) {
        const Courier view = courier.snapshot();
        const std::int64_t ahora = clock.nowMs();
        const double r = ratioDe(view, ahora);
        assert(r >= 0.0 && r <= 1.0);
        if (view.fromNode != view.toNode) {
            ++muestrasEnMovimiento;
            muestras.push_back(Muestra{r, view.fromNode + ">" + view.toNode, ahora,
                                       view.segmentEndMs - view.segmentStartMs});
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    courier.requestStop();
    worker.join();

    std::cout << "  " << muestrasEnMovimiento << " muestras con el courier en movimiento\n";
    assert(muestrasEnMovimiento > 20);

    int crecientes = 0, saltosInexplicables = 0, intermedios = 0;
    for (std::size_t i = 1; i < muestras.size(); ++i) {
        if (muestras[i].ratio > 0.05 && muestras[i].ratio < 0.95) {
            ++intermedios;
        }
        if (muestras[i].tramo != muestras[i - 1].tramo) {
            continue;
        }
        if (muestras[i].ratio >= muestras[i - 1].ratio) {
            ++crecientes;
        }
        const std::int64_t dt = muestras[i].simMs - muestras[i - 1].simMs;
        const std::int64_t duracion = muestras[i].duracionMs;
        const double esperado =
            duracion > 0 ? static_cast<double>(dt) / static_cast<double>(duracion) : 1.0;
        if (muestras[i].ratio - muestras[i - 1].ratio > esperado + 0.05) {
            ++saltosInexplicables;
        }
    }
    std::cout << "  avances dentro de un tramo: " << crecientes << " crecientes, "
              << saltosInexplicables << " mayores que lo que explica el tiempo\n";
    std::cout << "  " << intermedios
              << " muestras con el punto ENTRE dos esquinas, no sobre una\n";
    assert(crecientes > 10);
    assert(saltosInexplicables == 0);
    assert(intermedios > 10);

    std::size_t distintos = 1;
    for (std::size_t i = 1; i < muestras.size(); ++i) {
        if (muestras[i].tramo != muestras[i - 1].tramo) {
            ++distintos;
        }
    }
    std::cout << "  recorrio " << distintos << " tramos distintos\n";
    assert(distintos >= 2);

    std::cout << "  render ok\n";
}
