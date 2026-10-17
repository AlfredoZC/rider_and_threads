#include "OrderGenerator.h"

#include <algorithm>

namespace {
// Una de cada diez llegadas es una rafaga (§4: "en rafagas de hasta burstMax").
constexpr double kBurstProbability = 0.10;
}  // namespace

OrderGenerator::OrderGenerator(const Config& cfg, const SimClock& clock, OrderBook& book)
    : clock_(clock),
      book_(book),
      factory_(cfg, cfg.seed),
      // Semilla derivada: el generador y la factory sortean cosas distintas y conviene
      // que no compartan corriente, para que cambiar una no desplace a la otra. La
      // constante es la del numero aureo, la de mezcla habitual.
      rng_(cfg.seed ^ 0x9e3779b9u),
      meanIntervalMs_(cfg.meanIntervalMs),
      burstMax_(cfg.burstMax) {}

std::int64_t OrderGenerator::nextIntervalMs() {
    // Intervalos exponenciales: es el modelo estandar de llegadas (proceso de Poisson).
    // La mayoria son cortos y algunos muy largos, asi que los pedidos se agrupan solos,
    // que es como llegan de verdad. Con intervalos uniformes la cola casi nunca se
    // llenaria y el rechazo queueFull quedaria sin ejercitar.
    std::exponential_distribution<double> interval(
        1.0 / static_cast<double>(meanIntervalMs_));
    const double ms = interval(rng_);
    return std::max<std::int64_t>(1, static_cast<std::int64_t>(ms));
}

int OrderGenerator::nextBurstSize() {
    if (burstMax_ <= 1) {
        return 1;
    }
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    if (chance(rng_) >= kBurstProbability) {
        return 1;
    }
    std::uniform_int_distribution<int> size(2, burstMax_);
    return size(rng_);
}

void OrderGenerator::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopRequested_) {
        const std::int64_t waitSimMs = nextIntervalMs();

        // Esperar con plazo, no dormir. Si el hilo durmiera, con timeScale 1 y una
        // media de 20 s una señal de parada tardaria hasta 20 s en hacer efecto, y el
        // §7 da 5. wait_for devuelve el valor del predicado: true = nos pidieron parar.
        if (cv_.wait_for(lock, clock_.realDuration(waitSimMs),
                         [this] { return stopRequested_; })) {
            break;
        }

        const int burst = nextBurstSize();

        // Las ordenes se crean y se admiten SIN el candado del generador. admit() toma
        // el del OrderBook, y sostener dos candados a la vez sin necesidad es la receta
        // de un interbloqueo. factory_ y rng_ son de este hilo, asi que usarlos aca
        // afuera es seguro.
        lock.unlock();
        for (int i = 0; i < burst; ++i) {
            book_.admit(factory_.create(clock_.nowMs()));
        }
        lock.lock();
        generated_ += burst;
    }
}

void OrderGenerator::requestStop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopRequested_ = true;
    }
    cv_.notify_one();
}

std::int64_t OrderGenerator::generated() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return generated_;
}
