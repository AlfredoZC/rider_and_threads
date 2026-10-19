// El libro de ordenes: unico dueño de todas las Order del programa (§4).
//
// Todo el mundo referencia las ordenes por su OrderId y le pregunta al libro por su
// estado; nadie guarda una copia viva. Asi es imposible que dos partes del programa
// tengan versiones distintas de la misma orden, que es como nacen los defectos que el
// §4 prohibe: ordenes perdidas, duplicadas o en dos bolsas a la vez.
//
// Es un monitor. Las operaciones son acciones completas y no consultas sueltas:
// admit() verifica el limite y agrega DENTRO del mismo candado. Si en cambio expusiera
// un pendingCount() para que el llamador decida, entre consultar y agregar se soltaria
// el candado y una rafaga podria pasarse del limite. Eso es check-then-act, y es el
// error de concurrencia mas comun que hay.
#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <condition_variable>

#include "EventLog.h"
#include "Ids.h"
#include "Order.h"
#include "SimClock.h"

// Los numeros del resumen final (§6). created = delivered + rejected + pending.
struct OrderStats {
    std::int64_t created = 0;
    std::int64_t delivered = 0;
    std::int64_t rejected = 0;
    std::int64_t reassigned = 0;
    std::int64_t pending = 0;          // vivas en este instante
    std::int64_t avgDeliveryMs = 0;
    std::int64_t maxDeliveryMs = 0;
};

class OrderBook {
public:
    OrderBook(int maxPending, const SimClock& clock, EventLog& log);

    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;

    // Admite una orden recien creada. Si ya hay maxPending vivas la rechaza con
    // queueFull y devuelve false; en los dos casos deja registrado el evento.
    // Verificar y agregar ocurren bajo el mismo candado, por eso devuelve un bool en
    // vez de obligar al llamador a consultar antes.
    //
    // Tambien devuelve false si el id ya existe, y ahi no registra ni cuenta nada:
    // es una guarda contra un defecto silencioso, explicada en admit().
    bool admit(Order order);
    

    // Rechaza una orden viva (unreachable, noCourier). false si ya estaba en un
    // estado final.
    bool reject(OrderId id, RejectReason reason);

    // Las tres transiciones del reparto. Todas devuelven false si la orden no existe, si
    // la transicion no es legal, o si el courier que la pide no es el que la tiene
    // asignada. Ese ultimo chequeo es lo que impide que un courier entregue una orden
    // que lleva otro.
    bool assign(OrderId id, CourierId courier);
    bool pickedUp(OrderId id, CourierId courier);
    bool delivered(OrderId id, CourierId courier);
        // Espera a que haya al menos una orden esperando courier, a que venza el plazo o a
    // que alguien llame a stopWaiting(). Devuelve false solo en el ultimo caso.
    //
    // Existe para que el dispatcher no tenga que preguntar "hay trabajo?" en un bucle:
    // ese bucle consumiria una CPU entera, que es justo lo que prohibe el §4. Aca espera
    // dormido y admit() lo despierta.
    bool waitForUnassigned(std::chrono::milliseconds timeout);
    void stopWaiting();
    // Devuelve una orden asignada al estado created, para que vuelva a repartirse.
    //
    // countAsReassigned distingue los dos motivos, que el §6 trata distinto: una averia
    // del courier cuenta en el contador reassigned del resumen y registra el evento; una
    // oferta rechazada no, porque nadie llego a hacerse cargo de nada.
    bool unassign(OrderId id, bool countAsReassigned);

    // Marca como pending todas las ordenes que seguian vivas. Se llama UNA vez, al
    // cerrar, y DESPUES de que todos los hilos terminaron: si quedara alguno corriendo
    // podria mover una orden justo despues de contarla. Devuelve cuantas marco.
    std::int64_t markRemainingAsPending();

    // Una copia del estado actual, o nullopt si el id no existe. Copia: quien la recibe
    // trabaja fuera del candado.
    std::optional<Order> find(OrderId id) const;

    // Los ids de las ordenes que todavia esperan courier.
    std::vector<OrderId> unassignedIds() const;

    OrderStats stats() const;

private:
    // Cambia el estado validando la transicion con canTransition. Se llama SIEMPRE con
    // el candado ya tomado: es el unico punto del programa que mueve una orden de
    // estado, y por eso la regla de "exactamente un estado final" se cumple por
    // construccion y no por disciplina.
    bool setState(Order& order, OrderState to);

    const std::size_t maxPending_;
    const SimClock& clock_;
    EventLog& log_;

    mutable std::mutex mutex_;
    std::condition_variable work_;   // avisa que entro una orden sin asignar
    bool stopWaiting_ = false;       // protegido por mutex_
    std::unordered_map<OrderId, Order> orders_;   // protegido por mutex_
    std::size_t pending_ = 0;        // vivas en este instante
    // Las que markRemainingAsPending() dejo en estado pending. Se cuentan aparte
    // porque pending es un estado FINAL: al marcarlas, pending_ baja a cero, y sin
    // este contador el resumen reportaria siempre cero pendientes y el invariante
    // created = delivered + rejected + pending no cerraria.
    std::int64_t markedPending_ = 0;                     // protegido por mutex_
    OrderStats stats_;                            // protegido por mutex_
    std::int64_t deliverySumMs_ = 0;              // protegido por mutex_
};
