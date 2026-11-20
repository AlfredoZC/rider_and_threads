#include "Dispatcher.h"

#include <algorithm>
#include <optional>
#include <unordered_set>

#include "accel/TravelTimeTable.h"

namespace {
// Cotizacion por defecto: usa la tabla de tiempos si ya esta publicada y Dijkstra si no.
Dispatcher::QuoteFn defaultQuote(const SimContext& ctx) {
    const accel::TravelTimeTableSlot* slot = ctx.travelTimes;
    return [slot](Router& router, const CityGraph& graph, const Courier& courier, const Order& order) {
        const std::shared_ptr<const accel::TravelTimeTable> table = slot != nullptr ? slot->get() : nullptr;
        return estimateDeliveryMsWithTable(router, graph, courier, order, table.get());
    };
}

// Piso del plazo de reintento: con quoteTimeoutMs = 0 no se debe girar sin dormir.
constexpr std::int64_t kMinRetryMs = 50;
constexpr std::int64_t kMaxWaitMs = 24LL * 3600 * 1000;  // evita now()+ms desbordado

// Milisegundos que quedan hasta el plazo (0 si ya vencio).
std::int64_t remainingMs(std::chrono::steady_clock::time_point deadline) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
    return std::max<std::int64_t>(left.count(), 0);
}
}  // namespace

Dispatcher::Dispatcher(const Config& cfg, const SimContext& ctx,
                       const std::vector<std::unique_ptr<CourierActor>>& fleet)
    : ctx_(ctx),
      legacyFleet_(&fleet),
      quoteTimeoutMs_(cfg.quoteTimeoutMs),
      acceptTimeoutMs_(cfg.acceptTimeoutMs),
      agingMs_(cfg.agingMs),
      quote_(defaultQuote(ctx)),
      router_(ctx.graph),
      pool_(ctx.graph, static_cast<unsigned>(cfg.couriers) +
                           static_cast<unsigned>(cfg.reinforcements.max)),
      filter_(ctx.gpu) {}

Dispatcher::Dispatcher(const Config& cfg, const SimContext& ctx, Fleet& fleet)
    : ctx_(ctx),
      fleet_(&fleet),
      quoteTimeoutMs_(cfg.quoteTimeoutMs),
      acceptTimeoutMs_(cfg.acceptTimeoutMs),
      agingMs_(cfg.agingMs),
      quote_(defaultQuote(ctx)),
      router_(ctx.graph),
      pool_(ctx.graph, static_cast<unsigned>(cfg.couriers) +
                           static_cast<unsigned>(cfg.reinforcements.max)),
      filter_(ctx.gpu) {}

std::vector<CourierActor*> Dispatcher::activeCouriers() const {
    if (fleet_ != nullptr) {
        return fleet_->activeSnapshot();
    }
    std::vector<CourierActor*> actors;
    actors.reserve(legacyFleet_->size());
    for (const auto& actor : *legacyFleet_) {
        actors.push_back(actor.get());
    }
    return actors;
}

bool Dispatcher::destinoInalcanzable(const Order& order) {
    const NodeIndex pickup = ctx_.graph.indexOf(order.pickupNode);
    const NodeIndex dropoff = ctx_.graph.indexOf(order.dropoffNode);
    if (pickup == kNoNode || dropoff == kNoNode) {
        return true;
    }
    return !router_.shortestPath(pickup, dropoff).found;
}

void Dispatcher::run() {
    constexpr auto kFarFuture = std::chrono::hours(24);
    while (!stopped()) {
        std::uint64_t observed = ctx_.book.workVersion();
        sweepLateQuotes();
        ctx_.book.rejectExpired(acceptTimeoutMs_);
        spawnIfNeeded(ctx_.book.waitingCount());

        std::unordered_set<OrderId> attempted;
        bool retry = false;
        while (!stopped()) {
            // los pedidos que vencieron mientras se cotizaban otros no se eligen
            ctx_.book.rejectExpired(acceptTimeoutMs_);
            const std::optional<Order> order =
                ctx_.book.pickNextUnassigned(agingMs_, attempted);
            if (!order.has_value()) {
                break;
            }
            attempted.insert(order->id);
            const DispatchResult result = dispatchOne(*order);
            if (result == DispatchResult::noRoom) {
                break;
            }
            if (result == DispatchResult::retryLater) {
                retry = true;
            }
        }
        spawnIfNeeded(ctx_.book.waitingCount());

        auto deadline = std::chrono::steady_clock::now() + kFarFuture;
        const std::optional<std::int64_t> due =
            ctx_.book.nextDeadlineSimMs(acceptTimeoutMs_, agingMs_);
        if (due.has_value()) {
            deadline = std::min(deadline, ctx_.clock.timePointAt(*due));
        }
        if (retry) {
            // piso de reintento: con quoteTimeoutMs = 0 el reintento no debe girar
            const std::int64_t retryMs = std::clamp<std::int64_t>(quoteTimeoutMs_, kMinRetryMs,
                                                                  kMaxWaitMs);
            deadline = std::min(deadline, std::chrono::steady_clock::now() +
                                              std::chrono::milliseconds(retryMs));
        }
        if (!ctx_.book.waitForChange(observed, deadline)) {
            break;
        }
    }

    for (std::future<std::int64_t>& f : inFlight_) {
        if (f.valid()) {
            f.wait();
        }
    }
    inFlight_.clear();

    pool_.stop();
}

void Dispatcher::spawnIfNeeded(std::size_t waiting) {
    if (fleet_ == nullptr) {
        return;
    }
    bool spawned = false;
    while (fleet_->maybeSpawn(waiting)) {
        spawned = true;
    }
    if (spawned) {
        ctx_.book.notifyCapacityChanged();
    }
}

Dispatcher::DispatchResult Dispatcher::dispatchOne(const Order& order) {
    if (destinoInalcanzable(order)) {
        ctx_.book.reject(order.id, RejectReason::unreachable);
        return DispatchResult::rejected;
    }

    // UN solo plazo para todo el pedido (filtro, ronda de cercanos y ronda del resto): quoteTimeoutMs es el
    // tiempo REAL maximo que espera el dispatcher, no un plazo por etapa.
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(std::clamp<std::int64_t>(quoteTimeoutMs_, 0, kMaxWaitMs));

    std::vector<FleetMember> available;
    for (CourierActor* actor : activeCouriers()) {
        Courier view = actor->snapshot();
        if (view.state == CourierState::brokenDown || view.state == CourierState::onBreak ||
            !hasRoom(view)) {
            continue;
        }
        available.push_back(FleetMember{actor, std::move(view)});
    }
    if (available.empty()) {
        return DispatchResult::noRoom;
    }

    // Aplicacion 10: primero se cotiza a los que llegan al restaurante dentro del radio. Si nadie
    // queda dentro (o no hay tabla) se cotiza a toda la flota disponible, y si los cercanos no pueden
    // tomar el pedido se prueba despues con el resto: el filtro nunca genera un rechazo nuevo.
    std::vector<FleetMember> nearby;
    std::vector<FleetMember> others;
    splitByRadius(order, available, nearby, others, deadline);
    const DispatchResult first = quoteRound(order, nearby, deadline);
    if (others.empty() || first == DispatchResult::assigned || first == DispatchResult::rejected) {
        return first;
    }
    if (first == DispatchResult::retryLater || std::chrono::steady_clock::now() >= deadline) {
        return DispatchResult::retryLater;  // sin plazo para otra ronda: se reintenta despues
    }
    const DispatchResult second = quoteRound(order, others, deadline);
    if (second == DispatchResult::assigned || second == DispatchResult::rejected) {
        return second;
    }
    if (first == DispatchResult::retryLater || second == DispatchResult::retryLater) {
        return DispatchResult::retryLater;
    }
    return first == DispatchResult::unquotable && second == DispatchResult::unquotable
               ? DispatchResult::unquotable
               : DispatchResult::noRoom;
}

void Dispatcher::splitByRadius(const Order& order, const std::vector<FleetMember>& available,
                               std::vector<FleetMember>& nearby, std::vector<FleetMember>& others,
                               std::chrono::steady_clock::time_point deadline) {
    const std::shared_ptr<const accel::TravelTimeTable> table =
        ctx_.travelTimes != nullptr ? ctx_.travelTimes->get() : nullptr;
    const NodeIndex restaurant = ctx_.graph.indexOf(order.pickupNode);
    if (table == nullptr || candidateRadiusMs_ <= 0 || table->nodeCount() != ctx_.graph.nodeCount() ||
        restaurant == kNoNode) {
        nearby = available;
        return;
    }
    accel::FleetColumns columns;
    columns.node.reserve(available.size());
    columns.available.reserve(available.size());
    for (const FleetMember& member : available) {
        const NodeIndex node = ctx_.graph.indexOf(currentNode(member.view));
        const bool known = node != kNoNode && node < table->nodeCount();
        columns.node.push_back(known ? static_cast<std::int32_t>(node) : 0);
        columns.available.push_back(known ? 1 : 0);  // sin nodo conocido no se lee la tabla: queda para "others"
    }
    const std::int32_t maxTravelMs = static_cast<std::int32_t>(
        std::min<std::int64_t>(candidateRadiusMs_, accel::kUnreachableMs - 1));
    std::vector<std::int32_t> candidates;
    try {
        candidates = filter_.filter(table, std::move(columns), static_cast<std::int32_t>(restaurant), maxTravelMs,
                                    remainingMs(deadline));
    } catch (const std::exception&) {
        nearby = available;  // el filtro es solo una optimizacion: ante un fallo se cotiza a todos
        return;
    }
    if (candidates.empty()) {
        nearby = available;  // nadie dentro del radio: toda la flota
        return;
    }
    std::size_t nextCandidate = 0;
    for (std::size_t courier = 0; courier < available.size(); ++courier) {
        if (nextCandidate < candidates.size() && static_cast<std::size_t>(candidates[nextCandidate]) == courier) {
            nearby.push_back(available[courier]);
            ++nextCandidate;
        } else {
            others.push_back(available[courier]);
        }
    }
}

Dispatcher::DispatchResult Dispatcher::quoteRound(const Order& order, const std::vector<FleetMember>& members,
                                                  std::chrono::steady_clock::time_point deadline) {
    const OrderId orderId = order.id;

    std::vector<Candidate> candidates;
    for (const FleetMember& member : members) {
        candidates.push_back(Candidate{
            member.actor, pool_.submit([this, view = member.view, order](Router& router) {
                return quote_(router, ctx_.graph, view, order);
            })});
    }

    if (candidates.empty()) {
        return DispatchResult::noRoom;
    }
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        requested_ += static_cast<std::int64_t>(candidates.size());
    }


    std::vector<std::pair<std::int64_t, CourierActor*>> ranked;
    std::int64_t lateNow = 0;
    bool failed = false;

    for (Candidate& c : candidates) {
        if (c.estimate.wait_until(deadline) != std::future_status::ready) {
            inFlight_.push_back(std::move(c.estimate));
            ++lateNow;
            continue;
        }
        std::int64_t value = kUnreachable;
        try {
            value = c.estimate.get();
        } catch (...) {
            failed = true;
            continue;
        }
        if (value != kUnreachable) {
            ranked.emplace_back(value, c.actor);
        }
    }
    if (lateNow > 0) {
        const std::lock_guard<std::mutex> guard(mutex_);
        late_ += lateNow;
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& x, const auto& y) {
                  if (x.first != y.first) {
                      return x.first < y.first;
                  }
                  return x.second->id() < y.second->id();
              });

    // Si un courier rechaza la reserva (por ejemplo, justo empezo a descansar), se prueba
    // con el siguiente mejor.
    for (const auto& entry : ranked) {
        CourierActor& elegido = *entry.second;
        if (!elegido.reserve(order)) {
            continue;
        }
        // La cotizacion pudo tardar: el plazo se revalida bajo el candado del libro.
        if (ctx_.book.assignUnlessExpired(orderId, elegido.id(), acceptTimeoutMs_) !=
            OrderBook::AssignResult::assigned) {
            elegido.cancelReservation();
            return DispatchResult::rejected;
        }
        elegido.commitReservation(order);
        return DispatchResult::assigned;
    }

    if (lateNow > 0 || failed) {
        return DispatchResult::retryLater;
    }
    if (ranked.empty()) {
        // nadie puede llegar a este pedido (todo kUnreachable): no frena a los demas
        return DispatchResult::unquotable;
    }
    return DispatchResult::noRoom;
}

void Dispatcher::sweepLateQuotes() {
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
    ctx_.book.stopWaiting();
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
