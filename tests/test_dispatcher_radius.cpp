// Aplicacion 10 dentro del Dispatcher: con la tabla de tiempos publicada, antes de cotizar se filtra la
// flota por radio. Solo se cotiza a los cercanos; si nadie queda dentro del radio se cotiza a toda la flota
// (el filtro nunca deja un pedido sin cotizar); radio 0 lo apaga; sin tabla todo sigue como siempre.
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "Dispatcher.h"
#include "OrderFactory.h"
#include "accel/TravelTimeTable.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

template <typename Fn>
bool waitUntil(Fn condition, std::int64_t limitMs) {
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() <
           limitMs) {
        if (condition()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return condition();
}

struct Outcome {
    CourierId assignedCourier = kNoCourier;
    std::int64_t quotes = 0;
};

// Un pedido con 3 couriers en los nodos dados; radiusMs < 0 = no tocar el valor por defecto.
Outcome runOne(Config& cfg, const CityGraph& graph, const std::vector<std::string>& courierNodes,
               const std::string& pickup, const std::string& dropoff, bool publishTable, std::int64_t radiusMs,
               std::vector<CourierId>* quotedCouriers, bool customQuote = true, std::size_t* etaBatches = nullptr) {
    const SimClock clock(1.0);
    EventLog log("tests/bin/disp_radius.log", clock);
    OrderBook book(50, clock, log);
    Restaurants restaurants;
    for (const RestaurantConfig& restaurant : cfg.restaurants) {
        restaurants.emplace(restaurant.id, std::make_unique<Restaurant>(restaurant.id, restaurant.pickupSlots));
    }
    accel::TravelTimeTableSlot slot;
    if (publishTable) {
        accel::ThreadPool pool(2);
        slot.publish(accel::buildTravelTimeTableCpu(graph, pool));
    }
    const SimContext ctx{graph, clock, log, book, restaurants, nullptr, &slot};

    std::vector<std::unique_ptr<CourierActor>> fleet;
    for (std::size_t courier = 0; courier < courierNodes.size(); ++courier) {
        fleet.push_back(std::make_unique<CourierActor>(static_cast<CourierId>(courier) + 1, courierNodes[courier], 3, ctx));
    }
    cfg.quoteTimeoutMs = 5000;
    Dispatcher dispatcher(cfg, ctx, fleet);
    if (radiusMs >= 0) dispatcher.setCandidateRadiusMs(radiusMs);
    std::mutex quotedMutex;
    if (customQuote) dispatcher.setQuoteFunction([&](Router& router, const CityGraph& cityGraph, const Courier& courier, const Order& order) {
        {
            std::lock_guard<std::mutex> guard(quotedMutex);
            if (quotedCouriers != nullptr) quotedCouriers->push_back(courier.id);
        }
        return estimateDeliveryMs(router, cityGraph, courier, order);
    });

    Order order;
    order.id = 1;
    order.restaurantId = cfg.restaurants.front().id;
    order.pickupNode = pickup;
    order.dropoffNode = dropoff;
    CHECK(book.admit(order));
    std::thread worker(&Dispatcher::run, &dispatcher);
    CHECK(waitUntil([&] { return book.find(1)->state == OrderState::assigned; }, 3000));
    dispatcher.requestStop();
    worker.join();
    Outcome outcome;
    outcome.assignedCourier = book.find(1)->assignedCourier;
    outcome.quotes = dispatcher.quotesRequested();
    if (etaBatches != nullptr) *etaBatches = dispatcher.etaBatch().cpuBatches() + dispatcher.etaBatch().gpuBatches();
    return outcome;
}


// Un cotizador mas lento que el plazo: el pedido espera a lo sumo UN quoteTimeoutMs, la ronda de cercanos
// queda tarde (retryLater) y NO se lanza la ronda del resto (los couriers 1 y 3 nunca se cotizan).
void testSlowQuoterDoesNotMultiplyTimeout(Config& cfg, const CityGraph& graph, const std::string& restaurantNode,
                                          const std::string& farNode, const std::string& dropoff) {
    const SimClock clock(1.0);
    EventLog log("tests/bin/disp_radius_slow.log", clock);
    OrderBook book(50, clock, log);
    Restaurants restaurants;
    for (const RestaurantConfig& restaurant : cfg.restaurants) {
        restaurants.emplace(restaurant.id, std::make_unique<Restaurant>(restaurant.id, restaurant.pickupSlots));
    }
    accel::TravelTimeTableSlot slot;
    accel::ThreadPool pool(2);
    slot.publish(accel::buildTravelTimeTableCpu(graph, pool));
    const SimContext ctx{graph, clock, log, book, restaurants, nullptr, &slot};
    std::vector<std::unique_ptr<CourierActor>> fleet;
    const std::vector<std::string> courierNodes = {farNode, restaurantNode, farNode};
    for (std::size_t courier = 0; courier < courierNodes.size(); ++courier) {
        fleet.push_back(std::make_unique<CourierActor>(static_cast<CourierId>(courier) + 1, courierNodes[courier], 3, ctx));
    }
    cfg.quoteTimeoutMs = 300;
    Dispatcher dispatcher(cfg, ctx, fleet);
    dispatcher.setCandidateRadiusMs(1);
    std::mutex quotedMutex;
    std::vector<CourierId> quoted;
    std::atomic<bool> firstQuoteStarted{false};
    dispatcher.setQuoteFunction([&](Router&, const CityGraph&, const Courier& courier, const Order&) {
        {
            std::lock_guard<std::mutex> guard(quotedMutex);
            quoted.push_back(courier.id);
        }
        firstQuoteStarted = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1200));  // mucho mas que el plazo
        return std::int64_t{1000};
    });
    Order order;
    order.id = 1;
    order.restaurantId = cfg.restaurants.front().id;
    order.pickupNode = restaurantNode;
    order.dropoffNode = dropoff;
    CHECK(book.admit(order));
    const auto start = std::chrono::steady_clock::now();
    std::thread worker(&Dispatcher::run, &dispatcher);
    CHECK(waitUntil([&] { return firstQuoteStarted.load(); }, 3000));
    std::this_thread::sleep_for(std::chrono::milliseconds(700));  // mas de 2 plazos: la ronda 2 ya se habria lanzado
    dispatcher.requestStop();
    worker.join();
    (void)start;
    {
        std::lock_guard<std::mutex> guard(quotedMutex);
        CHECK(!quoted.empty());
        for (const CourierId courier : quoted) CHECK(courier == 2);  // nunca los lejanos
    }
    CHECK(book.find(1)->state != OrderState::assigned);
}
}  // namespace

int main() {
    Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    const std::string restaurantNode = cfg.restaurants.front().node;
    const std::string farNode = cfg.nodes.back().id;
    const std::string dropoff = cfg.startNode;
    CHECK(restaurantNode != farNode && restaurantNode != cfg.startNode);

    // Radio de 1 ms: solo el courier que ya esta en el restaurante (courier 2) queda dentro.
    {
        std::vector<CourierId> quoted;
        const Outcome outcome = runOne(cfg, graph, {farNode, restaurantNode, farNode}, restaurantNode, dropoff, true, 1, &quoted);
        CHECK(outcome.quotes == 1);
        CHECK(quoted.size() == 1 && quoted.front() == 2);
        CHECK(outcome.assignedCourier == 2);
    }
    // Nadie dentro del radio: se cotiza a toda la flota (nunca queda un pedido sin cotizar).
    {
        const Outcome outcome = runOne(cfg, graph, {farNode, farNode, cfg.startNode}, restaurantNode, dropoff, true, 1, nullptr);
        CHECK(outcome.quotes == 3);
        CHECK(outcome.assignedCourier != kNoCourier);
    }
    // Radio 0: el filtro esta apagado y se cotiza a todos.
    {
        const Outcome outcome = runOne(cfg, graph, {farNode, restaurantNode, farNode}, restaurantNode, dropoff, true, 0, nullptr);
        CHECK(outcome.quotes == 3);
        CHECK(outcome.assignedCourier == 2);
    }
    // Sin tabla publicada: se cotiza a todos, como antes.
    {
        const Outcome outcome = runOne(cfg, graph, {farNode, restaurantNode, farNode}, restaurantNode, dropoff, false, 1, nullptr);
        CHECK(outcome.quotes == 3);
        CHECK(outcome.assignedCourier == 2);
    }
    // Aplicacion 04: con la cotizacion por defecto y la tabla publicada, las cotizaciones de la ronda son una
    // fila de ETA en lote (misma eleccion que cotizando courier por courier); sin tabla no se usa el lote.
    {
        std::size_t batches = 0;
        const Outcome batched = runOne(cfg, graph, {farNode, restaurantNode, farNode}, restaurantNode, dropoff, true, 0,
                                       nullptr, false, &batches);
        CHECK(batched.quotes == 3 && batched.assignedCourier == 2);
        CHECK(batches >= 1);
        std::size_t noBatches = 0;
        const Outcome perCourier = runOne(cfg, graph, {farNode, restaurantNode, farNode}, restaurantNode, dropoff, false, 0,
                                          nullptr, false, &noBatches);
        CHECK(perCourier.quotes == 3 && perCourier.assignedCourier == batched.assignedCourier);
        CHECK(noBatches == 0);
        // Con el radio por defecto el lote se calcula solo para los cercanos.
        std::size_t radiusBatches = 0;
        const Outcome nearbyOnly = runOne(cfg, graph, {farNode, restaurantNode, farNode}, restaurantNode, dropoff, true, 1,
                                          nullptr, false, &radiusBatches);
        CHECK(nearbyOnly.quotes == 1 && nearbyOnly.assignedCourier == 2 && radiusBatches >= 1);
    }
    // Radio por defecto (10 min): el resultado es el mismo que con el filtro apagado.
    {
        const Outcome filtered = runOne(cfg, graph, {farNode, restaurantNode, cfg.startNode}, restaurantNode, dropoff, true, -1, nullptr);
        const Outcome unfiltered = runOne(cfg, graph, {farNode, restaurantNode, cfg.startNode}, restaurantNode, dropoff, false, -1, nullptr);
        CHECK(filtered.assignedCourier == unfiltered.assignedCourier);
    }
    testSlowQuoterDoesNotMultiplyTimeout(cfg, graph, restaurantNode, farNode, dropoff);
    std::cout << "TODO OK\n";
    return 0;
}
