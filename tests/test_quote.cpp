// Pruebas de estimateDeliveryMs.
//
// La prueba central es la del paralelismo: 8 hilos estimando a la vez, cada uno con SU
// Router, tienen que dar exactamente los mismos numeros que un solo hilo calculandolos
// de a uno. Si el Router estuviera compartido, los buffers se pisarian y aparecerian
// rutas mal calculadas, no solo un reporte de ThreadSanitizer.
#include "Quote.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "Config.h"

namespace {

Order orderAt(const std::string& pickup, const std::string& dropoff,
              std::int64_t prepMs = 0) {
    Order o;
    o.id = 1;
    o.restaurantId = "r0";
    o.pickupNode = pickup;
    o.dropoffNode = dropoff;
    o.prepMs = prepMs;
    return o;
}

}  // namespace

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);
    Router router(graph);

    const Courier courier(1, cfg.startNode, cfg.bagCapacity);
    const std::string restaurantNode = cfg.restaurants.front().node;

    // --- una estimacion normal: ir a buscar mas llevar
    {
        const Order order = orderAt(restaurantNode, cfg.nodes.back().id);
        const std::int64_t estimate = estimateDeliveryMs(router, graph, courier, order);
        assert(estimate > 0);

        const Route toPickup = router.shortestPath(graph.indexOf(cfg.startNode),
                                                   graph.indexOf(restaurantNode));
        const Route toDropoff = router.shortestPath(graph.indexOf(restaurantNode),
                                                    graph.indexOf(cfg.nodes.back().id));
        assert(toPickup.found && toDropoff.found);
        assert(estimate == toPickup.travelMs + toDropoff.travelMs);
        std::cout << "  estimacion = ir al restaurante (" << toPickup.travelMs / 1000
                  << "s) + llevar (" << toDropoff.travelMs / 1000 << "s)\n";
    }

    // --- si la comida tarda mas que el viaje, manda la comida: se espera, no se suma
    {
        const Order quick = orderAt(restaurantNode, cfg.nodes.back().id, 0);
        const std::int64_t base = estimateDeliveryMs(router, graph, courier, quick);

        const Order slow = orderAt(restaurantNode, cfg.nodes.back().id, 10000000);
        const std::int64_t waited = estimateDeliveryMs(router, graph, courier, slow);

        assert(waited > base);
        // el total tiene que ser prep + el tramo final, no prep + los dos tramos
        const Route toDropoff = router.shortestPath(graph.indexOf(restaurantNode),
                                                    graph.indexOf(cfg.nodes.back().id));
        assert(waited == 10000000 + toDropoff.travelMs);
        std::cout << "  con comida lenta espera en vez de sumar los dos tramos\n";
    }

    // --- un nodo inexistente no rompe nada: devuelve kUnreachable (§4)
    {
        assert(estimateDeliveryMs(router, graph, courier,
                                  orderAt("no_existe", cfg.nodes.back().id)) == kUnreachable);
        assert(estimateDeliveryMs(router, graph, courier,
                                  orderAt(restaurantNode, "no_existe")) == kUnreachable);
        Courier lost(2, "no_existe", 3);
        assert(estimateDeliveryMs(router, graph, lost,
                                  orderAt(restaurantNode, cfg.nodes.back().id)) == kUnreachable);
        std::cout << "  un nodo inalcanzable devuelve kUnreachable, no una excepcion\n";
    }

    // --- cada courier estima desde DONDE ESTA: dos posiciones, dos numeros
    {
        const Order order = orderAt(restaurantNode, cfg.nodes.back().id);
        const Courier far(2, cfg.nodes.back().id, 3);
        const std::int64_t fromStart = estimateDeliveryMs(router, graph, courier, order);
        const std::int64_t fromFar = estimateDeliveryMs(router, graph, far, order);
        assert(fromStart != fromFar);
        std::cout << "  la estimacion depende de la posicion del courier\n";
    }

    // --- LA prueba: 8 hilos estimando a la vez, cada uno con su propio Router
    {
        const int kThreads = 8;
        const int kOrders = 200;

        // referencia secuencial
        std::vector<std::int64_t> expected;
        expected.reserve(kOrders);
        for (int i = 0; i < kOrders; ++i) {
            const Courier c(static_cast<CourierId>(i) + 1,
                            cfg.nodes[static_cast<std::size_t>(i)].id, 3);
            expected.push_back(estimateDeliveryMs(
                router, graph, c,
                orderAt(restaurantNode, cfg.nodes[static_cast<std::size_t>(i) + 1].id)));
        }

        std::vector<std::vector<std::int64_t>> results(kThreads);
        std::vector<std::thread> threads;
        const auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&graph, &cfg, &results, &restaurantNode, t]() {
                // CADA hilo construye el suyo: es la memoria de trabajo de Dijkstra
                Router mine(graph);
                for (int i = 0; i < kOrders; ++i) {
                    const Courier c(static_cast<CourierId>(i) + 1,
                                    cfg.nodes[static_cast<std::size_t>(i)].id, 3);
                    results[static_cast<std::size_t>(t)].push_back(estimateDeliveryMs(
                        mine, graph, c,
                        orderAt(restaurantNode,
                                cfg.nodes[static_cast<std::size_t>(i) + 1].id)));
                }
            });
        }
        for (std::thread& th : threads) { th.join(); }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();

        for (const auto& got : results) {
            assert(got == expected);   // mismos numeros que en secuencial
        }
        std::cout << "  " << kThreads * kOrders << " estimaciones en " << kThreads
                  << " hilos simultaneos (" << ms << " ms): identicas a la secuencial\n";
    }

    std::cout << "  quote ok\n";
}
