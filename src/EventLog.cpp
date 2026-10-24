#include "EventLog.h"

#include <stdexcept>

namespace {

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
    return;
  }
  out += ",\"";
  out += key;
  out += "\":\"";
  out += jsonEscape(value);
  out += '"';
}

void appendId(std::string &out, const char *key, char prefix, std::uint64_t value) {
  if (value == 0) {
    return;
  }
  out += ",\"";
  out += key;
  out += "\":\"";
  out += prefix;
  out += std::to_string(value);
  out += '"';
}

}

EventLog::EventLog(const std::string &path, const SimClock &clock)
    : clock_(clock), out_(path, std::ios::out | std::ios::trunc) {
  if (!out_) {
    throw std::runtime_error("no se pudo abrir el log de eventos: " + path);
  }
}

std::int64_t EventLog::write(const Event &event) {
  std::string tail;
  tail.reserve(128);
  tail += ",\"event\":\"";
  tail += event.name;
  tail += '"';
  appendId(tail, "order", 'o', event.order);
  appendId(tail, "courier", 'c', event.courier);
  appendText(tail, "restaurant", event.restaurant);
  appendText(tail, "from", event.from);
  appendText(tail, "to", event.to);
  appendText(tail, "reason", event.reason);
  tail += "}\n";

  const std::lock_guard<std::mutex> guard(mutex_);
  const std::int64_t t = clock_.nowMs();
  out_ << "{\"t\":" << t << tail;
  out_.flush();
  return t;
}
