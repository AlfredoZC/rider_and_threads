// El hilo que crea ordenes: continuamente, a intervalos aleatorios y en rafagas (§4).
//
// Es dueño exclusivo de su OrderFactory y de su generador aleatorio, asi que esos dos
// no necesitan ninguna sincronizacion: un dato con un unico dueño no se protege, se
// posee. Lo unico compartido es la bandera de parada y el contador, protegidos por su
// candado, y el OrderBook, que se protege solo.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <random>

#include "Config.h"
#include "OrderBook.h"
#include "OrderFactory.h"
#include "SimClock.h"

class OrderGenerator {
public:
    OrderGenerator(const Config& cfg, const SimClock& clock, OrderBook& book);

    OrderGenerator(const OrderGenerator&) = delete;
    OrderGenerator& operator=(const OrderGenerator&) = delete;

    void run();           // cuerpo del hilo
    void requestStop();   // lo llama el hilo principal

    std::int64_t generated() const;   // cuantas ordenes creo, para pruebas e informe

private:
    std::int64_t nextIntervalMs();    // solo lo llama el hilo generador
    int nextBurstSize();              // idem

    const SimClock& clock_;
    OrderBook& book_;

    // Propiedad exclusiva del hilo generador: sin candado.
    OrderFactory factory_;
    std::mt19937 rng_;
    const std::int64_t meanIntervalMs_;
    const int burstMax_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopRequested_ = false;      // protegido por mutex_
    std::int64_t generated_ = 0;      // protegido por mutex_
};
