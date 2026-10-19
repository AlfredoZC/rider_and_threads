// El dispatcher: asigna cada orden al courier con mejor tiempo estimado de entrega (§4).
//
// Es el unico lugar del programa con paralelismo de verdad: las estimaciones de todos
// los candidatos se calculan AL MISMO TIEMPO, no una tras otra, porque el §4 lo exige y
// porque el plazo es de quoteTimeoutMs milisegundos REALES.
//
// Puede hacerlo sin un solo candado gracias a como estan repartidos los datos: el grafo
// es inmutable, cada tarea construye su propio Router, y el courier y la orden que
// recibe son copias. No hay nada compartido que se escriba.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

#include "Config.h"
#include "CourierActor.h"
#include "Quote.h"
#include "SimContext.h"

class Dispatcher {
public:
    // La funcion que estima. Es inyectable por dos razones: las pruebas necesitan una que
    // tarde o que lance para ejercitar el plazo y el manejo de errores, y el experimento
    // del §12 consiste justamente en cambiarla por una que tenga en cuenta la bolsa.
    using QuoteFn = std::function<std::int64_t(Router&, const CityGraph&, const Courier&,
                                               const Order&)>;


    Dispatcher(const Config& cfg, const SimContext& ctx,
               const std::vector<std::unique_ptr<CourierActor>>& fleet);

    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;

    void run();
    void requestStop();

    void setQuoteFunction(QuoteFn fn) { quote_ = std::move(fn); }

    // Cuantas estimaciones se pidieron y cuantas llegaron tarde. Para las pruebas y para
    // la tabla de experimentos del §12.
    std::int64_t quotesRequested() const;
    std::int64_t quotesLate() const;

private:
    struct Candidate {
        std::size_t index;                   // posicion en la flota
        std::future<std::int64_t> estimate;
    };

    void dispatchOne(OrderId orderId);
    void sweepLateQuotes();
    bool stopped() const;

    const SimContext ctx_;
    const std::vector<std::unique_ptr<CourierActor>>& fleet_;
    const std::int64_t quoteTimeoutMs_;    // REAL
    const std::int64_t acceptTimeoutMs_;   // SIMULADO
    QuoteFn quote_;

    // Las estimaciones que llegaron tarde se guardan aca en vez de destruirse. El
    // destructor de un future de std::async BLOQUEA hasta que la tarea termine, asi que
    // descartarlo al vencer el plazo haria esperar exactamente lo que se decidio no
    // esperar: el plazo quedaria de adorno.
    std::vector<std::future<std::int64_t>> inFlight_;

    mutable std::mutex mutex_;
    bool stopRequested_ = false;           // protegido por mutex_
    std::int64_t requested_ = 0;           // protegido por mutex_
    std::int64_t late_ = 0;                // protegido por mutex_
};
