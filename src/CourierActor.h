// El actor que le pone un hilo a un Courier.
//
// Courier es el dato (copiable, sin sincronizacion). CourierActor es el monitor que lo
// posee: contiene el estado, el candado que lo protege y la variable de condicion con la
// que el hilo espera. Igual que Order frente a OrderBook.
//
// No es copiable ni movible: tiene un mutex y una condition_variable adentro, y otros
// hilos van a guardar punteros a el. Por eso la flota se guarda como
// vector<unique_ptr<CourierActor>> y no como vector<CourierActor>.
#pragma once

#include <condition_variable>
#include <mutex>
#include <string>

#include "Courier.h"
#include "Ids.h"

class CourierActor {
public:
    CourierActor(CourierId id, const std::string& startNode, int bagCapacity);

    CourierActor(const CourierActor&) = delete;
    CourierActor& operator=(const CourierActor&) = delete;

    // El cuerpo del hilo. Termina cuando alguien llama a requestStop().
    void run();

    // La llama el hilo principal para que el courier termine. Es seguro llamarla
    // muchas veces y desde cualquier hilo.
    void requestStop();

    // Una copia del estado, tomada bajo candado. El renderer dibuja a partir de la
    // copia, sin sostener el candado mientras pinta.
    Courier snapshot() const;

    CourierId id() const { return id_; }   // id_ es const: no hace falta candado

private:
    const CourierId id_;

    // mutable para poder tomar el candado dentro de snapshot(), que es const: consultar
    // el estado no lo modifica conceptualmente, pero tomar un mutex si toca sus bytes.
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    Courier state_;                // protegido por mutex_
    bool stopRequested_ = false;   // protegido por mutex_
};
