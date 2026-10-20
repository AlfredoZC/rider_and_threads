// Pruebas del Restaurant: limite de slots y orden de llegada (§4).
//
// Dos condiciones que hay que comprobar juntas:
//
//   1. Nunca hay mas de pickupSlots couriers retirando a la vez. Se mide el maximo
//      observado con un contador atomico, no se supone.
//   2. Se atiende EN ORDEN DE LLEGADA. Un wait comun sobre "hay lugar" respetaria el
//      limite pero dejaria entrar al que el planificador elija, y uno podria esperar
//      indefinidamente. Aca se verifica que el orden de entrada sea el de llegada.
#include "Restaurant.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

int main() {
    // --- el limite se respeta bajo carga
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
        assert(atendidos.load() == kCouriers);   // todos pasaron, ninguno quedo afuera
        assert(restaurant.inUse() == 0);
    }

    // --- se atiende en orden de llegada
    {
        Restaurant restaurant("r1", 1);   // un solo slot: el orden se ve sin ambiguedad

        // el primero ocupa el unico lugar
        assert(restaurant.acquire());

        std::mutex m;
        std::vector<int> ordenDeEntrada;
        std::vector<std::thread> hilos;

        // los demas llegan de a uno, en orden conocido
        for (int i = 1; i <= 5; ++i) {
            hilos.emplace_back([&restaurant, &m, &ordenDeEntrada, i]() {
                assert(restaurant.acquire());
                {
                    const std::lock_guard<std::mutex> guard(m);
                    ordenDeEntrada.push_back(i);
                }
                restaurant.release();
            });
            // separacion suficiente para que el orden de llegada sea el del bucle
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        assert(restaurant.waiting() == 5);
        restaurant.release();                     // se libera el lugar: arranca la fila
        for (std::thread& h : hilos) { h.join(); }

        std::cout << "  orden de llegada 1,2,3,4,5 -> orden de entrada ";
        for (const int x : ordenDeEntrada) { std::cout << x; }
        std::cout << "\n";
        const std::vector<int> esperado = {1, 2, 3, 4, 5};
        assert(ordenDeEntrada == esperado);
    }

    // --- al parar, nadie queda colgado en la fila (§7)
    {
        Restaurant restaurant("r2", 1);
        assert(restaurant.acquire());             // el lugar queda ocupado

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
        assert(rechazados.load() == 6);   // ninguno quedo esperando un turno que no llega
        assert(ms < 500);
    }

    std::cout << "  restaurant ok\n";
}
