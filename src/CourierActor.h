// El actor que le pone un hilo a un Courier.
//
// Courier es el dato (copiable, sin sincronizacion). CourierActor es el monitor que lo
// posee: contiene el estado, el candado que lo protege y la variable de condicion con la
// que el hilo espera. Igual que Order frente a OrderBook.
//
// No es copiable ni movible: tiene un mutex y una condition_variable adentro, y otros
// hilos guardan punteros a el. Por eso la flota es vector<unique_ptr<CourierActor>>.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>
#include <vector>
#include <string>

#include "Courier.h"
#include "Ids.h"
#include "Order.h"
#include "Router.h"
#include "SimContext.h"

class CourierActor {
public:
    // breakdownProbability y seed valen cero por defecto para las pruebas que no
    // ejercitan averias. La semilla se deriva con el id, asi cada courier tiene su
    // propia corriente y la corrida sigue siendo reproducible.
    CourierActor(CourierId id, const std::string& startNode, int bagCapacity,
                 const SimContext& ctx, double breakdownProbability = 0.0,
                 std::uint32_t seed = 0);

    CourierActor(const CourierActor&) = delete;
    CourierActor& operator=(const CourierActor&) = delete;

    // El cuerpo del hilo: espera, reparte, vuelve a esperar.
    void run();

    // La llama el hilo principal. Es segura desde cualquier hilo y cuantas veces sea.
    void requestStop();

    // Recibe una orden. Devuelve false si no hay lugar en la bolsa, si esta averiado o si
    // ya le pidieron parar. Verificar y agregar ocurren bajo el MISMO candado: si el
    // llamador pudiera consultar primero y ofrecer despues, dos dispatchers podrian
    // pasarse de bagCapacity.
    bool offer(const Order& order);

    // Una copia del estado, tomada bajo candado. El dibujo y las estimaciones trabajan
    // sobre la copia, sin sostener el candado mientras calculan.
    Courier snapshot() const;

    CourierId id() const { return id_; }   // id_ es const: no necesita candado

    // La flota se conecta DESPUES de construir a todos, porque un courier necesita
    // conocer a sus companeros para traspasarles sus ordenes al averiarse, y no puede
    // recibir en el constructor una lista que todavia se esta llenando.
    void setFleet(const std::vector<std::unique_ptr<CourierActor>>* fleet) {
        fleet_ = fleet;
    }

    // Provoca una averia ahora mismo, sin sortear. La usan las pruebas, que necesitan
    // que ocurra cuando ellas deciden y no cuando lo diga el azar.
    bool forceBreakdown();

private:
    // En que termino un trayecto. Se distingue "no hay camino" de "nos pidieron parar"
    // porque la primera es un rechazo unreachable del §4 y la segunda no lo es.
    enum class Trip { arrived, stopped, unreachable };

    Trip travelTo(Router& router, const std::string& destNode);
    void deliver(Router& router, OrderId orderId);

    // Espera un lapso de tiempo SIMULADO. Devuelve false si nos pidieron parar.
    bool waitSim(std::int64_t simMs);

    void setCourierState(CourierState s);
    void finish(OrderId orderId, bool wasDelivered);

    // Sortea una averia y, si toca, traspasa TODAS sus ordenes. Devuelve si se averio.
    // fraction reparte la probabilidad entre los puntos donde se sortea, para que
    // agregar un punto nuevo cambie CUANDO ocurren las averias y no cuantas.
    bool maybeBreakDown(double fraction);
    bool breakDownAndHandOver();
    // Intenta mover TODAS las ordenes a target. Todo-o-nada: si no entran todas, no
    // se mueve ninguna. Devuelve los ids movidos, vacio si no se pudo.
    std::vector<OrderId> tryHandOverTo(CourierActor& target);

    const CourierId id_;
    const SimContext ctx_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    const double breakdownProbability_;
    std::mt19937 rng_;             // propio de este hilo: no se comparte
    const std::vector<std::unique_ptr<CourierActor>>* fleet_ = nullptr;

    Courier state_;                // protegido por mutex_
    bool stopRequested_ = false;   // protegido por mutex_
};
