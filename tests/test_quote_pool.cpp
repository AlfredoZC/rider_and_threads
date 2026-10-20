// Pruebas del QuotePool: paralelismo acotado por el hardware.
//
// Lo que hay que comprobar:
//   1. La cantidad de trabajadores la decide el hardware, y nunca es cero.
//   2. Las tareas se ejecutan EN PARALELO: N tareas de 100 ms con N trabajadores tardan
//      ~100 ms, no N x 100.
//   3. Nunca corren mas tareas a la vez que trabajadores hay. El pool es un limite, no
//      una sugerencia.
//   4. Una excepcion en una tarea viaja al hilo que espera y se vuelve a lanzar en su
//      get(), en vez de matar el programa (§4).
//   5. Al parar no queda ningun future colgado ni ningun hilo abandonado (§7).
#include "QuotePool.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "Config.h"
#include "Quote.h"   // kUnreachable

namespace {

std::int64_t elapsedMs(std::chrono::steady_clock::time_point from) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - from).count();
}

}  // namespace

int main() {
    const Config cfg = loadConfig("config/equipetrol.json");
    const CityGraph graph(cfg);

    // --- la cantidad de trabajadores sale del hardware
    {
        QuotePool pool(graph);
        std::cout << "  hardware_concurrency=" << std::thread::hardware_concurrency()
                  << " -> " << pool.workers() << " trabajadores\n";
        assert(pool.workers() > 0);
    }

    // --- las tareas corren en paralelo, y nunca mas de las que hay trabajadores
    {
        const unsigned kWorkers = 4;
        const int kJobs = 8;
        QuotePool pool(graph, kWorkers);

        std::atomic<int> dentro{0};
        std::atomic<int> maximo{0};

        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::future<std::int64_t>> futures;
        for (int i = 0; i < kJobs; ++i) {
            futures.push_back(pool.submit([&dentro, &maximo](Router&) {
                const int ahora = ++dentro;
                int previo = maximo.load();
                while (previo < ahora && !maximo.compare_exchange_weak(previo, ahora)) {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                --dentro;
                return std::int64_t{42};
            }));
        }
        for (auto& f : futures) {
            assert(f.get() == 42);
        }
        const std::int64_t ms = elapsedMs(t0);

        std::cout << "  " << kJobs << " tareas de 100 ms con " << kWorkers
                  << " trabajadores: " << ms << " ms, maximo simultaneo "
                  << maximo.load() << "\n";
        assert(maximo.load() <= static_cast<int>(kWorkers));   // el limite se respeta
        assert(ms < 400);   // dos tandas de 100 ms; secuencial habrian sido 800
    }

    // --- una tarea que lanza no tumba el programa: la excepcion viaja por el future
    {
        QuotePool pool(graph, 2);
        auto buena = pool.submit([](Router&) { return std::int64_t{7}; });
        auto mala = pool.submit([](Router&) -> std::int64_t {
            throw std::runtime_error("fallo al estimar");
        });

        assert(buena.get() == 7);
        bool lanzo = false;
        try {
            mala.get();
        } catch (const std::runtime_error&) {
            lanzo = true;
        }
        assert(lanzo);
        std::cout << "  la excepcion de una tarea se relanza en get(), sin matar nada\n";
    }

    // --- el Router del trabajador sirve para calcular de verdad
    {
        QuotePool pool(graph, 2);
        const NodeIndex from = graph.indexOf(cfg.startNode);
        const NodeIndex to = graph.indexOf(cfg.restaurants.front().node);
        auto f = pool.submit([from, to](Router& router) {
            const Route r = router.shortestPath(from, to);
            return r.found ? r.travelMs : kUnreachable;
        });
        const std::int64_t ms = f.get();
        std::cout << "  una tarea calculo una ruta real: " << ms / 1000 << " s simulados\n";
        assert(ms > 0);
    }

    // --- al parar, nada queda colgado
    {
        QuotePool pool(graph, 3);
        std::vector<std::future<std::int64_t>> futures;
        for (int i = 0; i < 30; ++i) {
            futures.push_back(pool.submit([](Router&) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                return std::int64_t{1};
            }));
        }
        const auto t0 = std::chrono::steady_clock::now();
        pool.stop();
        const std::int64_t ms = elapsedMs(t0);

        // ninguno quedo sin respuesta: o dio su valor o dio una excepcion
        int respondidos = 0;
        for (auto& f : futures) {
            try {
                f.get();
                ++respondidos;
            } catch (const std::exception&) {
                ++respondidos;
            }
        }
        assert(respondidos == 30);
        std::cout << "  stop() con 30 tareas en vuelo: todas respondieron, en " << ms
                  << " ms\n";

        pool.stop();   // idempotente
    }

    std::cout << "  quote pool ok\n";
}
