// El log de eventos (§6): un objeto JSON por linea, escrito por todos los
// hilos.
//
// Es un MONITOR: el archivo y el candado que lo protege son privados, y la
// unica forma de escribir es por write(), que toma el candado. Quien lo usa no
// ve el mutex, no puede olvidarse de tomarlo y no puede tomarlo mal. La
// correccion queda encapsulada en vez de depender de que los 10 hilos del
// programa se acuerden de hacer lo correcto.
//
// Es la HOJA de la jerarquia de candados del proyecto: write() no llama a nada
// que pueda tomar otro candado, asi que se puede registrar un evento teniendo
// tomado el candado de un courier o del OrderBook sin riesgo de interbloqueo.
// La regla inversa no vale: nunca hay que tomar otro candado mientras se tiene
// el del log.
#pragma once

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

#include "Ids.h"
#include "SimClock.h"

// Los nombres exactos del §6. Constantes y no cadenas sueltas: un error de
// tipeo en "orderPickedUp" seria invisible hasta que alguien revise el log a
// mano.
namespace EventName {
constexpr const char *kOrderCreated = "orderCreated";
constexpr const char *kOrderAssigned = "orderAssigned";
constexpr const char *kPickupStarted = "pickupStarted";
constexpr const char *kOrderPickedUp = "orderPickedUp";
constexpr const char *kOrderDelivered = "orderDelivered";
constexpr const char *kOrderRejected = "orderRejected";
constexpr const char *kCourierBreakdown = "courierBreakdown";
constexpr const char *kOrderReassigned = "orderReassigned";
constexpr const char *kSimulationStopping = "simulationStopping";
} // namespace EventName

// Un evento a registrar. Los campos que no aplican quedan vacios y no se
// escriben: una orden rechazada no tiene courier, y simulationStopping no tiene
// orden.
struct Event {
  const char *name = "";
  OrderId order = kNoOrder;
  CourierId courier = kNoCourier;
  std::string restaurant;
  std::string from;
  std::string to;
  std::string reason;
};

class EventLog {
public:
  // Lanza std::runtime_error si no se puede abrir el archivo.
  EventLog(const std::string &path, const SimClock &clock);

  // Registra el evento y devuelve la t simulada que le quedo. Se devuelve
  // porque quien crea una orden necesita guardar ese mismo valor en createdMs:
  // si volviera a consultar el reloj, el log y la orden diferirian en unos
  // milisegundos.
  std::int64_t write(const Event &event);

private:
  const SimClock &clock_;
  std::ofstream out_;
  std::mutex mutex_;
};
