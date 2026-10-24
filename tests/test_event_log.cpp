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

}

int main() {
    const SimClock clock(60.0);
    const std::string path = "tests/bin/test_events.log";

    {
        EventLog log(path, clock);

        Event created;
        created.name = EventName::kOrderCreated;
        created.order = 42;
        created.restaurant = "r0";
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
    assert(a["order"] == "o42");
    assert(a["restaurant"] == "r0");
    assert(!a.contains("from"));
    assert(!a.contains("to"));
    assert(a.contains("t"));
    assert(!a.contains("courier"));
    assert(!a.contains("reason"));

    const nlohmann::json b = nlohmann::json::parse(lines[1]);
    assert(b["event"] == "orderRejected");
    assert(b["reason"] == "queueFull");
    assert(!b.contains("restaurant"));

    const nlohmann::json c = nlohmann::json::parse(lines[2]);
    assert(c["event"] == "simulationStopping");
    assert(!c.contains("order"));
    std::cout << "  un evento por linea, solo con los campos que aplican\n";

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
                                    log.write(e);
                }
            });
        }
        for (std::thread& t : threads) { t.join(); }

        const std::vector<std::string> all = readLines(path);
        assert(all.size() == static_cast<std::size_t>(kThreads * kPerThread));

        std::int64_t lastT = -1;
        std::vector<int> perCourier(kThreads + 1, 0);
        for (const std::string& line : all) {
            const nlohmann::json j = nlohmann::json::parse(line);
            assert(j["event"] == "orderDelivered");
            assert(j.contains("order") && j.contains("courier"));

            const std::int64_t t = j["t"].get<std::int64_t>();
            assert(t >= lastT);
            lastT = t;

            perCourier[std::stoul(j["courier"].get<std::string>().substr(1))]++;
        }
        for (int id = 1; id <= kThreads; ++id) {
            assert(perCourier[static_cast<std::size_t>(id)] == kPerThread);
        }
        std::cout << "  " << all.size() << " lineas de 8 hilos: todas JSON valido, "
                  << "ninguna perdida, t nunca retrocede\n";
    }

    std::cout << "  event log ok\n";
}
