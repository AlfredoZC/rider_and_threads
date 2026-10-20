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

    // Se muestrea la posicion mientras viaja, como haria el renderer. De cada muestra se
    // guarda tambien el instante y la duracion del tramo, porque el avance esperado
    // depende de los dos: sin eso no se puede distinguir un salto del programa de una
    // muestra que llego tarde.
    struct Muestra {
        double ratio;
        std::string tramo;
        std::int64_t simMs;        // instante simulado en que se tomo
        std::int64_t duracionMs;   // cuanto dura el tramo completo
    };
    std::vector<Muestra> muestras;
    int muestrasEnMovimiento = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0).count() < 3000) {
        const Courier view = courier.snapshot();
        const std::int64_t ahora = clock.nowMs();
        const double r = ratioDe(view, ahora);
        assert(r >= 0.0 && r <= 1.0);          // siempre acotado: nunca se pasa del nodo
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
    assert(muestrasEnMovimiento > 20);         // no estuvo quieto

    // Dentro de un mismo tramo, el avance tiene que ser creciente y explicable por el
    // tiempo transcurrido.
    //
    // La version anterior comparaba contra una constante: "ningun avance mayor a media
    // cuadra entre dos muestras". Eso no mide el programa, mide la puntualidad del
    // muestreador. Con la CPU saturada el bucle se despierta tarde, el tiempo simulado
    // avanza igual -la posicion se deduce del reloj, no de quien la mire- y el avance
    // legitimo entre dos muestras pasa de media cuadra. Reproducido: la prueba fallaba
    // con seis quemadores de CPU y pasaba con la maquina libre.
    //
    // La propiedad de verdad es que la posicion sea una FUNCION CONTINUA DEL TIEMPO: el
    // avance tiene que ser el que corresponde al rato que paso, ni mas ni menos. Asi, si
    // la muestra llega tarde, el avance esperado crece con ella y la comparacion sigue
    // valiendo.
    int crecientes = 0, saltosInexplicables = 0, intermedios = 0;
    for (std::size_t i = 1; i < muestras.size(); ++i) {
        if (muestras[i].ratio > 0.05 && muestras[i].ratio < 0.95) {
            ++intermedios;
        }
        if (muestras[i].tramo != muestras[i - 1].tramo) {
            continue;                           // cambio de tramo: no comparable
        }
        if (muestras[i].ratio >= muestras[i - 1].ratio) {
            ++crecientes;
        }
        const std::int64_t dt = muestras[i].simMs - muestras[i - 1].simMs;
        const std::int64_t duracion = muestras[i].duracionMs;
        const double esperado =
            duracion > 0 ? static_cast<double>(dt) / static_cast<double>(duracion) : 1.0;
        // El margen cubre el redondeo y el desfase de medio muestreo entre leer el
        // snapshot y leer el reloj.
        if (muestras[i].ratio - muestras[i - 1].ratio > esperado + 0.05) {
            ++saltosInexplicables;
        }
    }
    std::cout << "  avances dentro de un tramo: " << crecientes << " crecientes, "
              << saltosInexplicables << " mayores que lo que explica el tiempo\n";
    std::cout << "  " << intermedios
              << " muestras con el punto ENTRE dos esquinas, no sobre una\n";
    assert(crecientes > 10);
    assert(saltosInexplicables == 0);           // el punto no salta
    // Esta es la que descarta el comportamiento que el §2 llama defecto: un punto que
    // salta de esquina en esquina estaria siempre en un ratio de 0 o de 1.
    assert(intermedios > 10);

    // Y recorrio varios tramos distintos: no se quedo en uno solo.
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
