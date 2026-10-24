#include "Restaurant.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

int main() {
    {
        const int kSlots = 2;
        const int kCouriers = 12;
        Restaurant restaurant("r0", kSlots);

        std::atomic<int> dentro{0};
        std::atomic<int> maximo{0};
        std::atomic<int> atendidos{0};

        std::vector<std::thread> hilos;
        for (int i = 0; i < kCouriers; ++i) {
            hilos.emplace_back([&restaurant, &dentro, &maximo, &atendidos]() {
                assert(restaurant.acquire());
                const int ahora = ++dentro;
                int previo = maximo.load();
                while (previo < ahora && !maximo.compare_exchange_weak(previo, ahora)) {
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                --dentro;
                ++atendidos;
                restaurant.release();
            });
        }
        for (std::thread& h : hilos) { h.join(); }

        std::cout << "  " << kCouriers << " couriers, " << kSlots
                  << " slots: maximo simultaneo observado = " << maximo.load() << "\n";
        assert(maximo.load() <= kSlots);
        assert(atendidos.load() == kCouriers);
        assert(restaurant.inUse() == 0);
    }

    {
        Restaurant restaurant("r1", 1);

        assert(restaurant.acquire());

        std::mutex m;
        std::vector<int> ordenDeEntrada;
        std::vector<std::thread> hilos;

        for (int i = 1; i <= 5; ++i) {
            hilos.emplace_back([&restaurant, &m, &ordenDeEntrada, i]() {
                assert(restaurant.acquire());
                {
                    const std::lock_guard<std::mutex> guard(m);
                    ordenDeEntrada.push_back(i);
                }
                restaurant.release();
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        assert(restaurant.waiting() == 5);
        restaurant.release();
        for (std::thread& h : hilos) { h.join(); }

        std::cout << "  orden de llegada 1,2,3,4,5 -> orden de entrada ";
        for (const int x : ordenDeEntrada) { std::cout << x; }
        std::cout << "\n";
        const std::vector<int> esperado = {1, 2, 3, 4, 5};
        assert(ordenDeEntrada == esperado);
    }

    {
        Restaurant restaurant("r2", 1);
        assert(restaurant.acquire());

        std::atomic<int> rechazados{0};
        std::vector<std::thread> hilos;
        for (int i = 0; i < 6; ++i) {
            hilos.emplace_back([&restaurant, &rechazados]() {
                if (!restaurant.acquire()) {
                    ++rechazados;
                }
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        const auto t0 = std::chrono::steady_clock::now();
        restaurant.stop();
        for (std::thread& h : hilos) { h.join(); }
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();

        std::cout << "  stop() libero a los 6 que hacian fila en " << ms << " ms\n";
        assert(rechazados.load() == 6);
        assert(ms < 500);
    }

    std::cout << "  restaurant ok\n";
}
