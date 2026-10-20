// Un restaurante como monitor: como maximo pickupSlots couriers retirando a la vez, y
// los demas esperan EN ORDEN DE LLEGADA (§4).
//
// Las dos condiciones juntas son el motivo de los numeros de turno. Un wait comun sobre
// "hay lugar" respetaria el limite pero no el orden: al liberarse un lugar, notify_all
// despierta a todos y pasa el que el planificador elija, que puede ser el que acaba de
// llegar mientras otro lleva rato esperando. Eso es inanicion, y el §4 la prohibe al
// pedir orden de llegada.
//
// Con turnos, cada courier espera por SU numero, asi que aunque se despierten los ocho,
// siete comprueban que no les toca y se vuelven a dormir.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>

class Restaurant {
public:
    Restaurant(std::string id, int slots);

    Restaurant(const Restaurant&) = delete;
    Restaurant& operator=(const Restaurant&) = delete;

    // Espera a que le toque y haya lugar. Devuelve false si le pidieron parar, y en ese
    // caso NO hay que llamar a release().
    bool acquire();
    void release();

    // Despierta a todos los que esperan para que puedan terminar (§7).
    void stop();

    const std::string& id() const { return id_; }   // id_ es const: sin candado
    int slots() const { return slots_; }
    int inUse() const;      // para el dibujo
    int waiting() const;    // cuantos hacen fila

private:
    const std::string id_;
    const int slots_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int inUse_ = 0;                  // protegido por mutex_
    std::uint64_t nextTicket_ = 0;   // protegido por mutex_
    std::uint64_t nowServing_ = 0;   // protegido por mutex_
    bool stop_ = false;              // protegido por mutex_
};
