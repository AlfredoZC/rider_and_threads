#include "Dispatcher.h"

#include <algorithm>
#include <limits>

namespace {
// Cada cuanto vuelve a mirar aunque no entren ordenes nuevas. Hace falta porque una
// orden puede vencer su acceptTimeoutMs sin que nadie la toque.
constexpr std::chrono::milliseconds kIdleCheck{200};
}  // namespace

Dispatcher::Dispatcher(const Config& cfg, const SimContext& ctx,
                       const std::vector<std::unique_ptr<CourierActor>>& fleet)
    : ctx_(ctx),
      fleet_(fleet),
      quoteTimeoutMs_(cfg.quoteTimeoutMs),
      acceptTimeoutMs_(cfg.acceptTimeoutMs),
      // El valor por defecto es la funcion misma: no hace falta envolverla en una lambda
      // si la firma coincide. Las pruebas y el experimento del §12 la reemplazan.
      quote_(estimateDeliveryMs),
      pool_(ctx.graph) {}


void Dispatcher::run() {
    while (!stopped()) {
        // Espera dormido a que entre una orden. El plazo no es una encuesta: admit() lo
        // despierta al instante, y el plazo solo sirve para revisar vencimientos.
        if (!ctx_.book.waitForUnassigned(kIdleCheck)) {
            break;
        }
        sweepLateQuotes();
        for (const OrderId id : ctx_.book.unassignedIds()) {
            if (stopped()) {
                break;
            }
            dispatchOne(id);
        }
    }

    // Al cerrar no se abandona trabajo: se espera a las estimaciones que quedaron
    // sueltas. Duran microsegundos, asi que no demoran el cierre (§7).
    for (std::future<std::int64_t>& f : inFlight_) {
        if (f.valid()) {
            f.wait();
        }
    }
    inFlight_.clear();

    // El pool se para al final, cuando ya no queda nadie esperando un resultado.
    pool_.stop();
}

void Dispatcher::dispatchOne(OrderId orderId) {
    const auto maybe = ctx_.book.find(orderId);
    if (!maybe.has_value() || maybe->state != OrderState::created) {
        return;   // alguien la movio entre que la listamos y ahora
    }
    const Order order = *maybe;

    // Nadie la acepto a tiempo: se rechaza con noCourier (§4). Este plazo es en tiempo
    // SIMULADO, a diferencia del de las estimaciones.
    if (ctx_.clock.nowMs() - order.createdMs > acceptTimeoutMs_) {
        ctx_.book.reject(orderId, RejectReason::noCourier);
        return;
    }

    // --- se lanzan TODAS antes de esperar ninguna: ahi esta el paralelismo. Si se
    // esperara cada una antes de lanzar la siguiente, el tiempo total seria la suma en
    // vez del maximo.
    std::vector<Candidate> candidates;
    for (std::size_t i = 0; i < fleet_.size(); ++i) {
        const Courier view = fleet_[i]->snapshot();   // copia bajo candado
        if (view.state == CourierState::brokenDown || !hasRoom(view)) {
            continue;
        }
        candidates.push_back(Candidate{
            i, pool_.submit([this, view, order](Router& router) {
                // El Router lo aporta el trabajador: es SU memoria de trabajo, creada
                // una vez y reutilizada. Antes cada tarea construia la suya.
                return quote_(router, ctx_.graph, view, order);
            })});
    }

    if (candidates.empty()) {
        return;
    }
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        requested_ += static_cast<std::int64_t>(candidates.size());
    }

    // --- se recogen hasta el plazo. El plazo es un INSTANTE comun a todas: si fuera una
    // duracion por cada una, ocho candidatos podrian consumir ocho veces el presupuesto.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(quoteTimeoutMs_);

    std::size_t best = 0;
    std::int64_t bestMs = std::numeric_limits<std::int64_t>::max();
    bool found = false;
    std::int64_t lateNow = 0;

    for (Candidate& c : candidates) {
        if (c.estimate.wait_until(deadline) != std::future_status::ready) {
            // Llego tarde: se ignora, pero NO se destruye el future aca, porque su
            // destructor bloquearia hasta que la tarea termine.
            inFlight_.push_back(std::move(c.estimate));
            ++lateNow;
            continue;
        }
        std::int64_t value = kUnreachable;
        try {
            value = c.estimate.get();
        } catch (const std::exception&) {
            continue;   // §4: un error en una estimacion llega como error y no tumba nada
        } catch (...) {
            continue;
        }
        if (value != kUnreachable && value < bestMs) {
            bestMs = value;
            best = c.index;
            found = true;
        }
    }
    if (lateNow > 0) {
        const std::lock_guard<std::mutex> guard(mutex_);
        late_ += lateNow;
    }

    if (!found) {
        return;   // todas inalcanzables o tardias: se reintenta en la vuelta siguiente
    }

    // --- asignar y ofrecer. assign() vuelve a validar dentro de su candado, porque entre
    // que miramos la orden y ahora pasaron hasta quoteTimeoutMs.
    if (!ctx_.book.assign(orderId, fleet_[best]->id())) {
        return;
    }
    if (!fleet_[best]->offer(order)) {
        // El courier se lleno mientras estimabamos: la orden vuelve a la cola. No cuenta
        // como reasignacion, que el §6 reserva para las averias.
        ctx_.book.unassign(orderId, false);
    }
}

void Dispatcher::sweepLateQuotes() {
    // Las tardias ya terminaron casi siempre: se sacan sin bloquear, preguntando con un
    // plazo de cero.
    inFlight_.erase(
        std::remove_if(inFlight_.begin(), inFlight_.end(),
                       [](std::future<std::int64_t>& f) {
                           return !f.valid() ||
                                  f.wait_for(std::chrono::seconds(0)) ==
                                      std::future_status::ready;
                       }),
        inFlight_.end());
}

void Dispatcher::requestStop() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        stopRequested_ = true;
    }
    ctx_.book.stopWaiting();   // lo despierta si esta esperando ordenes
}

bool Dispatcher::stopped() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return stopRequested_;
}

std::int64_t Dispatcher::quotesRequested() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return requested_;
}

std::int64_t Dispatcher::quotesLate() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return late_;
}
