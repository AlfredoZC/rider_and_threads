// Alta de refuerzos cuando el sistema no deja crear mas hilos (EAGAIN).
// Una excepcion que escapa del hilo del despachador termina el proceso con abort; en una
// simulacion con miles de hilos el volcado del abort tarda minutos y el proceso no cierra.
// Por eso un fallo al crear el hilo de un refuerzo debe ser "no hay refuerzo ahora", no un error.
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "Config.h"
#include "Fleet.h"
#include "SimContext.h"

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

#if defined(__linux__)
namespace {

// Con el tope de procesos del usuario en 1, crear cualquier hilo nuevo falla con EAGAIN (salvo
// root, que no tiene tope). Restaura el tope original al salir del alcance.
class ThreadCreationBlocker {
public:
    ThreadCreationBlocker() {
        active_ = getrlimit(RLIMIT_NPROC, &saved_) == 0;
        if (active_) {
            rlimit blocked = saved_;
            blocked.rlim_cur = 1;
            active_ = setrlimit(RLIMIT_NPROC, &blocked) == 0;
        }
    }
    ~ThreadCreationBlocker() { release(); }
    ThreadCreationBlocker(const ThreadCreationBlocker&) = delete;
    ThreadCreationBlocker& operator=(const ThreadCreationBlocker&) = delete;

    bool active() const { return active_; }
    void release() {
        if (active_) {
            setrlimit(RLIMIT_NPROC, &saved_);
            active_ = false;
        }
    }

private:
    rlimit saved_{};
    bool active_ = false;
};

}  // namespace
#endif

int main() {
#if !defined(__linux__)
    std::cout << "  omitido: necesita setrlimit(RLIMIT_NPROC) de Linux\n";
    return 0;
#else
    if (geteuid() == 0) {
        std::cout << "  omitido: root no tiene tope de procesos, no se puede forzar EAGAIN\n";
        return 0;
    }

    Config cfg = loadConfig("config/equipetrol.json");
    cfg.reinforcements.max = 3;
    cfg.reinforcements.threshold = 1;
    cfg.reinforcements.idleMs = 1000000;
    const CityGraph graph(cfg);
    const SimClock clock(1.0);
    EventLog log("tests/bin/fleet_spawn.log", clock);
    OrderBook book(50, clock, log);
    Restaurants restaurants;
    for (const RestaurantConfig& r : cfg.restaurants) {
        restaurants.emplace(r.id, std::make_unique<Restaurant>(r.id, r.pickupSlots));
    }
    const SimContext ctx{graph, clock, log, book, restaurants};
    Fleet fleet(cfg, ctx);

    {
        ThreadCreationBlocker blocker;
        CHECK(blocker.active());
        bool spawned = true;
        try {
            spawned = fleet.maybeSpawn(10);
        } catch (const std::exception& e) {
            std::cerr << "FALLO: maybeSpawn lanzo en vez de devolver false: " << e.what() << "\n";
            return 1;
        }
        CHECK(!spawned);
        CHECK(fleet.reinforcementsJoined() == 0);
        CHECK(fleet.allSnapshot().empty());
        std::cout << "  sin hilos disponibles no hay refuerzo y no se lanza\n";
    }

    // Recuperacion: con hilos disponibles de nuevo, el siguiente intento si da de alta el refuerzo.
    CHECK(fleet.maybeSpawn(10));
    CHECK(fleet.reinforcementsJoined() == 1);
    CHECK(fleet.allSnapshot().size() == 1);
    std::cout << "  al volver a haber hilos el refuerzo se da de alta\n";

    fleet.requestStopAll();
    fleet.joinAll();
    return 0;
#endif
}
