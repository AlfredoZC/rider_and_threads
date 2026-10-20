// Pruebas de la posicion que usa el dibujo (§2).
//
// El enunciado es explicito: "un punto que salta de esquina en esquina, o que esta
// quieto mientras el log dice que el courier anda, es un defecto". Asi que hay que
// comprobar dos cosas sobre la posicion interpolada:
//
//   1. Que AVANCE de forma continua mientras el courier recorre un tramo, en vez de
//      quedarse en el nodo y saltar al llegar.
//   2. Que entre dos instantes consecutivos el salto sea pequeño: si fuera del tamaño de
//      una cuadra, el punto estaria saltando.
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

// La misma interpolacion que usa el dibujo, en coordenadas del grafo.
double ratioDe(const Courier& c, std::int64_t nowMs) { return positionRatio(c, nowMs); }

}  // namespace

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    const std::string restaurantNode = cfg.restaurants.front().node;
    const std::string clientNode = cfg.nodes.back().id;

    const SimClock clock(60.0);   // lento: los tramos duran lo suficiente para muestrear
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

    // Se muestrea la posicion mientras viaja, como haria el renderer.
    std::vector<double> ratios;
    std::vector<std::string> tramos;
    int muestrasEnMovimiento = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count() < 3000) {
        const Courier view = courier.snapshot();
        const double r = ratioDe(view, clock.nowMs());
        assert(r >= 0.0 && r <= 1.0);          // siempre acotado: nunca se pasa del nodo
        if (view.fromNode != view.toNode) {
            ++muestrasEnMovimiento;
            ratios.push_back(r);
            tramos.push_back(view.fromNode + ">" + view.toNode);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    courier.requestStop();
    worker.join();

    std::cout << "  " << muestrasEnMovimiento << " muestras con el courier en movimiento\n";
    assert(muestrasEnMovimiento > 20);         // no estuvo quieto

    // Dentro de un mismo tramo, el avance tiene que ser creciente y en pasos chicos.
    int crecientes = 0, saltosGrandes = 0;
    for (std::size_t i = 1; i < ratios.size(); ++i) {
        if (tramos[i] != tramos[i - 1]) {
            continue;                           // cambio de tramo: no comparable
        }
        if (ratios[i] >= ratios[i - 1]) {
            ++crecientes;
        }
        if (ratios[i] - ratios[i - 1] > 0.5) {
            ++saltosGrandes;                    // mas de media cuadra de un muestreo al otro
        }
    }
    std::cout << "  avances dentro de un tramo: " << crecientes << " crecientes, "
              << saltosGrandes << " saltos de mas de media cuadra\n";
    assert(crecientes > 10);
    assert(saltosGrandes == 0);                 // el punto no salta

    // Y recorrio varios tramos distintos: no se quedo en uno solo.
    std::size_t distintos = 1;
    for (std::size_t i = 1; i < tramos.size(); ++i) {
        if (tramos[i] != tramos[i - 1]) {
            ++distintos;
        }
    }
    std::cout << "  recorrio " << distintos << " tramos distintos\n";
    assert(distintos >= 2);

    std::cout << "  render ok\n";
}
