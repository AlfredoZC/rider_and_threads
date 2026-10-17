#include "EventLog.h"

#include <stdexcept>

namespace {

// Lo minimo que exige JSON para una cadena entre comillas.
std::string jsonEscape(const std::string &value) {
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
}

void appendText(std::string &out, const char *key, const std::string &value) {
  if (value.empty()) {
    return; // el campo no aplica a este evento
  }
  out += ",\"";
  out += key;
  out += "\":\"";
  out += jsonEscape(value);
  out += '"';
}

void appendId(std::string &out, const char *key, std::uint64_t value) {
  if (value == 0) {
    return; // 0 es kNoOrder / kNoCourier: no aplica
  }
  out += ",\"";
  out += key;
  out += "\":";
  out += std::to_string(value);
}

} // namespace

EventLog::EventLog(const std::string &path, const SimClock &clock)
    : clock_(clock), out_(path, std::ios::out | std::ios::trunc) {
  if (!out_) {
    throw std::runtime_error("no se pudo abrir el log de eventos: " + path);
  }
}

std::int64_t EventLog::write(const Event &event) {
  // Todo el formateo ocurre FUERA del candado: construir cadenas, pedir memoria
  // y convertir numeros es la parte cara, y hacerla adentro pondria a los 10
  // hilos a hacer cola mientras cada uno arma su linea.
  std::string tail;
  tail.reserve(128);
  tail += ",\"event\":\"";
  tail += event.name;
  tail += '"';
  appendId(tail, "order", event.order);
  appendId(tail, "courier", event.courier);
  appendText(tail, "restaurant", event.restaurant);
  appendText(tail, "from", event.from);
  appendText(tail, "to", event.to);
  appendText(tail, "reason", event.reason);
  tail += "}\n";

  // Adentro del candado queda lo minimo: leer el reloj y volcar la linea.
  //
  // La t se toma ACA y no antes de tomar el candado. Si cada hilo leyera el
  // reloj afuera, dos hilos podrian escribir en orden inverso al de sus marcas
  // de tiempo y el archivo quedaria con las t desordenadas, que el §6 prohibe.
  // Tomandola adentro, el orden de las lineas es el orden de los tiempos por
  // construccion.
  const std::lock_guard<std::mutex> guard(mutex_);
  const std::int64_t t = clock_.nowMs();
  out_ << "{\"t\":" << t << tail;
  out_.flush();
  return t;
}
