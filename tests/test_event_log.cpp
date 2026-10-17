// Pruebas de EventLog, el primer monitor del proyecto.
//
// Lo que de verdad importa aca no es que escriba bien un evento suelto, sino que 10
// hilos escribiendo a la vez produzcan un archivo donde CADA linea sea un JSON valido
// y completo. Sin el candado, dos escrituras se entrelazan y la linea deja de parsear:
// por eso la prueba central lanza 8 hilos y despues parsea las 4000 lineas una por una.
#include "EventLog.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

std::vector<std::string> readLines(const std::string& path) {
    std::ifstream in(path);
    assert(in);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

}  // namespace

int main() {
    const SimClock clock(60.0);
    const std::string path = "tests/bin/test_events.log";

    // --- un evento de cada forma: solo se escriben los campos que aplican
    {
        EventLog log(path, clock);

        Event created;
        created.name = EventName::kOrderCreated;
        created.order = 42;
        created.restaurant = "r0";
        created.from = "n15";
        created.to = "n420";
        const std::int64_t t = log.write(created);
        assert(t >= 0);

        Event rejected;
        rejected.name = EventName::kOrderRejected;
        rejected.order = 43;
        rejected.reason = "queueFull";
        log.write(rejected);

        Event stopping;
        stopping.name = EventName::kSimulationStopping;
        log.write(stopping);
    }

    const std::vector<std::string> lines = readLines(path);
    assert(lines.size() == 3);

    const nlohmann::json a = nlohmann::json::parse(lines[0]);
    assert(a["event"] == "orderCreated");
    assert(a["order"] == 42);
    assert(a["restaurant"] == "r0");
    assert(a["from"] == "n15");
    assert(a["to"] == "n420");
    assert(a.contains("t"));
    assert(!a.contains("courier"));     // no aplica: no se escribe
    assert(!a.contains("reason"));

    const nlohmann::json b = nlohmann::json::parse(lines[1]);
    assert(b["event"] == "orderRejected");
    assert(b["reason"] == "queueFull");
    assert(!b.contains("restaurant"));

    const nlohmann::json c = nlohmann::json::parse(lines[2]);
    assert(c["event"] == "simulationStopping");
    assert(!c.contains("order"));       // un evento global no lleva orden
    std::cout << "  un evento por linea, solo con los campos que aplican\n";

    // --- abrir un log en una ruta imposible tiene que fallar, no romperse en silencio
    {
        bool threw = false;
        try {
            EventLog bad("/no/existe/esta/carpeta/events.log", clock);
        } catch (const std::exception&) {
            threw = true;
        }
        assert(threw);
        std::cout << "  una ruta invalida lanza excepcion\n";
    }

    // --- LA prueba: 8 hilos escribiendo a la vez
    {
        const int kThreads = 8;
        const int kPerThread = 500;
        EventLog log(path, clock);

        std::vector<std::thread> threads;
        for (int id = 1; id <= kThreads; ++id) {
            threads.emplace_back([&log, id]() {
                for (int i = 0; i < kPerThread; ++i) {
                    Event e;
                    e.name = EventName::kOrderDelivered;
                    e.order = static_cast<OrderId>(id * 1000 + i);
                    e.courier = static_cast<CourierId>(id);
                    e.restaurant = "r0";
                    e.from = "n15";
                    e.to = "n420";
                    log.write(e);
                }
            });
        }
        for (std::thread& t : threads) { t.join(); }

        const std::vector<std::string> all = readLines(path);
        assert(all.size() == static_cast<std::size_t>(kThreads * kPerThread));

        // cada linea tiene que ser un JSON completo: si dos escrituras se entrelazaran,
        // parse() lanzaria aca
        std::int64_t lastT = -1;
        std::vector<int> perCourier(kThreads + 1, 0);
        for (const std::string& line : all) {
            const nlohmann::json j = nlohmann::json::parse(line);
            assert(j["event"] == "orderDelivered");
            assert(j.contains("order") && j.contains("courier"));

            // la t no puede retroceder: se toma DENTRO del candado, asi que el orden de
            // las lineas es el orden de los tiempos
            const std::int64_t t = j["t"].get<std::int64_t>();
            assert(t >= lastT);
            lastT = t;

            perCourier[j["courier"].get<std::size_t>()]++;
        }
        for (int id = 1; id <= kThreads; ++id) {
            assert(perCourier[static_cast<std::size_t>(id)] == kPerThread);
        }
        std::cout << "  " << all.size() << " lineas de 8 hilos: todas JSON valido, "
                  << "ninguna perdida, t nunca retrocede\n";
    }

    std::cout << "  event log ok\n";
}
