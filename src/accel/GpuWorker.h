#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include "accel/GpuError.h"

namespace accel {

// Hilo dueño de la GPU. Un unico hilo ejecuta todo el trabajo de GPU; los demas hilos envian
// tareas con submit() y reciben el resultado por un std::future, asi los repartidores siguen
// andando y la ventana se sigue refrescando mientras corre un kernel.
//
//  - Cola FIFO protegida por mutex + condition_variable: el hilo espera bloqueado, sin girar
//    ni dormir.
//  - init() se ejecuta en el propio hilo (ahi se crea el contexto CUDA, que queda ligado a el).
//    El constructor espera (bloqueado) a que termine. Si lanza, el worker queda "no saludable".
//  - Una tarea que lanza GpuFatalError (error CUDA pegajoso) deja el worker no saludable para
//    siempre: las tareas en cola se rechazan y los submit() siguientes lanzan GpuUnavailableError
//    de inmediato, para que quien llama use el respaldo CPU. Cualquier otra excepcion de una
//    tarea va solo a su future y el worker sigue vivo.
//  - stop(): rechaza envios nuevos, deja terminar la tarea en curso, resuelve con
//    GpuUnavailableError todas las tareas en cola (nunca queda un future sin resolver) y une el
//    hilo. Es idempotente y el destructor la llama. NO se puede llamar desde una tarea del propio
//    worker, ni destruir el worker desde una de ellas (haria join de su propio hilo).
class GpuWorker {
public:
    using InitFn = std::function<void()>;
    using Clock = std::chrono::steady_clock;

    explicit GpuWorker(InitFn init);
    ~GpuWorker();
    GpuWorker(const GpuWorker&) = delete;
    GpuWorker& operator=(const GpuWorker&) = delete;

    // Encola `fn` y devuelve su future. Lanza GpuUnavailableError si el worker esta detenido o
    // no es saludable. `fn` se ejecuta en el hilo del worker.
    //
    // `deadline` (opcional, steady_clock REAL): si al sacar la tarea de la cola ya vencio, NO se
    // ejecuta y su future se resuelve con GpuTaskExpiredError; asi la cola no se llena de trabajo
    // que nadie espera. Una tarea que ya empezo no se interrumpe.
    //
    // Reglas para quien envia (el worker las da por ciertas):
    //  - `fn` captura por VALOR o con shared_ptr: una tarea vencida puede seguir corriendo (o
    //    quedar en cola) DESPUES de que el llamador descarto su future y salio de su ambito, asi
    //    que no puede referirse a variables locales del llamador.
    //  - Una tarea NO espera futures de este mismo worker (se bloquearia a si misma).
    //  - Una tarea NO llama stop() ni destruye el worker.
    template <typename F>
    auto submit(F&& fn, std::optional<Clock::time_point> deadline = std::nullopt)
        -> std::future<std::invoke_result_t<std::decay_t<F>>> {
        using Fn = std::decay_t<F>;
        using R = std::invoke_result_t<Fn>;
        struct State {
            explicit State(F&& f) : fn(std::forward<F>(f)) {}
            Fn fn;
            std::promise<R> promise;
        };
        auto state = std::make_shared<State>(std::forward<F>(fn));
        std::future<R> future = state->promise.get_future();

        Job job;
        job.deadline = deadline;
        job.run = [this, state] {
            try {
                if constexpr (std::is_void_v<R>) {
                    state->fn();
                    state->promise.set_value();
                } else {
                    state->promise.set_value(state->fn());
                }
            } catch (const GpuFatalError& e) {
                markUnhealthy(e.what());  // antes de resolver: el llamador ya ve el estado
                state->promise.set_exception(std::current_exception());
            } catch (...) {
                state->promise.set_exception(std::current_exception());
            }
        };
        job.fail = [state](std::exception_ptr error) { state->promise.set_exception(error); };
        enqueue(std::move(job));
        return future;
    }

    void stop();

    bool healthy() const;
    // Motivo por el que quedo no saludable ("" si lo esta).
    std::string unhealthyReason() const;

private:
    struct Job {
        std::optional<Clock::time_point> deadline;
        std::function<void()> run;                      // ejecuta y resuelve el future
        std::function<void(std::exception_ptr)> fail;   // lo resuelve con una excepcion
    };

    void loop(InitFn init, std::promise<void> ready);
    void enqueue(Job job);
    void markUnhealthy(const std::string& reason);
    static std::exception_ptr unavailable(const std::string& why);

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Job> queue_;
    bool stopping_ = false;
    bool healthy_ = true;
    std::string reason_;
    std::mutex joinMutex_;  // serializa stop() concurrentes (un solo join)
    std::thread thread_;
};

// Tope de un plazo (24 h), igual que el del Dispatcher: evita now() + ms desbordado.
constexpr std::int64_t kMaxWaitMs = 24LL * 3600 * 1000;

// Plazo ABSOLUTO en tiempo REAL (steady_clock) a `timeoutMs` desde ahora, recortado a
// [0, kMaxWaitMs]. quoteTimeoutMs es tiempo real ("longest REAL time the dispatcher waits"), no
// simulado, y es el mismo patron de Dispatcher.cpp. Se calcula UNA vez por ronda y lo comparten
// todos los candidatos: nunca N x timeout.
inline GpuWorker::Clock::time_point deadlineAfterMs(std::int64_t timeoutMs) {
    return GpuWorker::Clock::now() +
           std::chrono::milliseconds(std::clamp<std::int64_t>(timeoutMs, 0, kMaxWaitMs));
}

// Espera el future hasta el plazo absoluto. true = resultado listo a tiempo; false = vencio (el
// future sigue valido: quien llama usa el respaldo CPU y lo descarta; descartar un future de una
// promesa no bloquea). No bloquea mas que el plazo; si ya paso, solo mira si esta listo.
template <typename R>
bool waitUntilDeadline(std::future<R>& future, GpuWorker::Clock::time_point deadline) {
    if (!future.valid()) {
        return false;
    }
    return future.wait_until(deadline) == std::future_status::ready;
}

}  // namespace accel
