#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "accel/GpuWorker.h"

// El hilo dueño de la GPU se prueba con tareas falsas: no hace falta GPU. Las tareas "lentas"
// se bloquean en una promesa que el test libera (sin sleeps para sincronizar).

namespace {

using accel::GpuFatalError;
using accel::GpuUnavailableError;
using accel::GpuWorker;
using namespace std::chrono_literals;

template <typename Ex, typename F>
bool throwsType(F&& f) {
    try {
        f();
    } catch (const Ex&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void fifoOrderAndResults() {
    GpuWorker worker([] {});
    assert(worker.healthy());
    std::vector<int> order;  // solo lo toca el hilo del worker hasta los get()
    std::vector<std::future<int>> results;
    for (int i = 0; i < 50; ++i) {
        results.push_back(worker.submit([i, &order] {
            order.push_back(i);
            return i * 2;
        }));
    }
    for (int i = 0; i < 50; ++i) {
        assert(results[static_cast<std::size_t>(i)].get() == i * 2);
    }
    assert(order.size() == 50);
    for (int i = 0; i < 50; ++i) {
        assert(order[static_cast<std::size_t>(i)] == i);
    }
    worker.submit([] {}).get();
    assert(worker.submit([] { return std::string("hola"); }).get() == "hola");
}

void runsOnOneDedicatedThread() {
    GpuWorker worker([] {});
    const std::thread::id a = worker.submit([] { return std::this_thread::get_id(); }).get();
    const std::thread::id b = worker.submit([] { return std::this_thread::get_id(); }).get();
    assert(a == b);
    assert(a != std::this_thread::get_id());
}

void initRunsOnWorkerThread() {
    std::thread::id initThread;
    GpuWorker worker([&initThread] { initThread = std::this_thread::get_id(); });
    const std::thread::id taskThread =
        worker.submit([] { return std::this_thread::get_id(); }).get();
    assert(initThread == taskThread);
    assert(initThread != std::this_thread::get_id());
}

void taskExceptionGoesToItsFutureOnly() {
    GpuWorker worker([] {});
    auto bad = worker.submit([]() -> int { throw std::runtime_error("boom"); });
    auto good = worker.submit([] { return 7; });
    assert(throwsType<std::runtime_error>([&] { bad.get(); }));
    assert(good.get() == 7);
    assert(worker.healthy());
}

void stopWithFullQueueResolvesEverything() {
    GpuWorker worker([] {});
    std::promise<void> started;
    std::promise<void> gate;
    std::shared_future<void> gateFuture = gate.get_future().share();
    auto running = worker.submit([&started, gateFuture] {
        started.set_value();
        gateFuture.wait();
        return 1;
    });
    started.get_future().wait();  // la tarea 0 esta en curso
    std::vector<std::future<int>> queued;
    for (int i = 0; i < 100; ++i) {
        queued.push_back(worker.submit([] { return 2; }));
    }
    // stop() bloquea hasta el join: corre en otro hilo mientras la tarea en curso sigue.
    std::thread stopper([&worker] { worker.stop(); });
    // Las de la cola se rechazan sin esperar a la tarea en curso.
    for (auto& f : queued) {
        assert(f.wait_for(5s) == std::future_status::ready);
        assert(throwsType<GpuUnavailableError>([&] { f.get(); }));
    }
    gate.set_value();
    stopper.join();
    assert(running.get() == 1);  // la tarea en curso termino normalmente
    assert(throwsType<GpuUnavailableError>([&] { worker.submit([] { return 3; }); }));
    worker.stop();  // idempotente
}

void submitAfterStopThrows() {
    GpuWorker worker([] {});
    worker.stop();
    assert(throwsType<GpuUnavailableError>([&] { worker.submit([] {}); }));
}

void destructorStopsAndResolves() {
    std::future<int> pending;
    std::future<int> running;
    std::promise<void> started;
    std::promise<void> gate;
    std::shared_future<void> gateFuture = gate.get_future().share();
    std::thread releaser;
    {
        GpuWorker worker([] {});
        running = worker.submit([&started, gateFuture] {
            started.set_value();
            gateFuture.wait();
            return 5;
        });
        started.get_future().wait();
        pending = worker.submit([] { return 6; });
        releaser = std::thread([&gate] { gate.set_value(); });
    }  // ~GpuWorker: stop + join
    releaser.join();
    assert(running.get() == 5);
    assert(throwsType<GpuUnavailableError>([&] { pending.get(); }));
}

void slowTaskAndExpiredDeadline() {
    GpuWorker worker([] {});
    std::promise<void> gate;
    std::shared_future<void> gateFuture = gate.get_future().share();
    auto slow = worker.submit([gateFuture] {
        gateFuture.wait();
        return 42;
    });
    // quoteTimeoutMs es tiempo REAL: debe esperar ~200 ms reales, no menos ni mucho mas
    // (con un reloj simulado a escala 60 habria vencido a los ~3 ms).
    const auto t0 = std::chrono::steady_clock::now();
    assert(!accel::waitUntilDeadline(slow, accel::deadlineAfterMs(200)));  // tarde
    const auto waited = std::chrono::steady_clock::now() - t0;
    assert(waited >= 190ms);
    assert(waited < 2s);
    // El llamador sigue; el worker no se colgo: al liberar, responde y atiende lo siguiente.
    gate.set_value();
    assert(slow.get() == 42);
    auto next = worker.submit([] { return 8; });
    assert(accel::waitUntilDeadline(next, accel::deadlineAfterMs(5000)));
    assert(next.get() == 8);
}

void candidatesShareOneDeadline() {
    GpuWorker worker([] {});
    std::promise<void> gate;
    std::shared_future<void> gateFuture = gate.get_future().share();
    std::vector<std::future<void>> candidates;
    for (int i = 0; i < 5; ++i) {
        candidates.push_back(worker.submit([gateFuture] { gateFuture.wait(); }));
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto deadline = accel::deadlineAfterMs(200);  // UNA vez por ronda
    for (auto& c : candidates) {
        assert(!accel::waitUntilDeadline(c, deadline));
    }
    // 5 candidatos tardios cuestan ~200 ms en total, no 5 x 200 ms.
    assert(std::chrono::steady_clock::now() - t0 < 700ms);
    gate.set_value();
    for (auto& c : candidates) {
        c.get();
    }
}

void deadlineEdgeCases() {
    GpuWorker worker([] {});
    auto f = worker.submit([] { return 1; });
    f.wait();
    assert(accel::waitUntilDeadline(f, accel::deadlineAfterMs(0)));  // listo: a tiempo con plazo 0
    std::promise<void> gate;
    std::shared_future<void> g = gate.get_future().share();
    auto blocked = worker.submit([g] { g.wait(); });
    assert(!accel::waitUntilDeadline(blocked, accel::deadlineAfterMs(-5)));  // no bloquea
    assert(!accel::waitUntilDeadline(blocked,
                                     accel::deadlineAfterMs(std::numeric_limits<std::int64_t>::min())));
    // Valor enorme: se recorta (sin desbordar) y el plazo queda en el futuro.
    const auto huge = accel::deadlineAfterMs(std::numeric_limits<std::int64_t>::max());
    assert(huge > std::chrono::steady_clock::now());
    assert(huge <= std::chrono::steady_clock::now() + std::chrono::hours(25));
    gate.set_value();
    blocked.get();
}

void expiredTaskIsNotRunWhenDequeued() {
    GpuWorker worker([] {});
    std::promise<void> started;
    std::promise<void> gate;
    std::shared_future<void> g = gate.get_future().share();
    auto blocker = worker.submit([&started, g] {
        started.set_value();
        g.wait();
    });
    started.get_future().wait();
    auto ran = std::make_shared<std::atomic<bool>>(false);
    auto expired = worker.submit(
        [ran] {
            ran->store(true);
            return 1;
        },
        accel::deadlineAfterMs(20));
    auto alive = worker.submit([ran] { return 2; }, accel::deadlineAfterMs(60000));
    // Esperamos a que el plazo de la primera pase (bloqueado, sin sleep).
    assert(!accel::waitUntilDeadline(expired, accel::deadlineAfterMs(60)));
    gate.set_value();
    blocker.get();
    assert(throwsType<accel::GpuTaskExpiredError>([&] { expired.get(); }));
    assert(!ran->load());   // no se ejecuto
    assert(alive.get() == 2);  // una con plazo vigente si
    assert(worker.healthy());
}

void unhealthyAfterInitFailureFailsFast() {
    GpuWorker worker([] { throw std::runtime_error("sin GPU de prueba"); });
    assert(!worker.healthy());
    assert(worker.unhealthyReason().find("sin GPU de prueba") != std::string::npos);
    const auto t0 = std::chrono::steady_clock::now();
    assert(throwsType<GpuUnavailableError>([&] { worker.submit([] { return 1; }); }));
    assert(std::chrono::steady_clock::now() - t0 < 1s);
    worker.stop();
}

void fatalTaskErrorMakesWorkerUnhealthyForever() {
    GpuWorker worker([] {});
    std::promise<void> started;
    std::promise<void> gate;
    std::shared_future<void> g = gate.get_future().share();
    auto first = worker.submit([&started, g]() -> int {
        started.set_value();
        g.wait();
        throw GpuFatalError("error CUDA pegajoso");
    });
    started.get_future().wait();
    auto queued = worker.submit([] { return 3; });  // en cola cuando ocurre el fallo fatal
    gate.set_value();
    assert(throwsType<GpuFatalError>([&] { first.get(); }));
    assert(throwsType<GpuUnavailableError>([&] { queued.get(); }));  // no se ejecuta
    assert(!worker.healthy());
    assert(worker.unhealthyReason().find("pegajoso") != std::string::npos);
    assert(throwsType<GpuUnavailableError>([&] { worker.submit([] { return 1; }); }));
}

void manyProducersConcurrently() {
    GpuWorker worker([] {});
    std::atomic<int> sum{0};
    std::vector<std::thread> producers;
    for (int p = 0; p < 8; ++p) {
        producers.emplace_back([&worker, &sum] {
            for (int i = 0; i < 200; ++i) {
                sum += worker.submit([] { return 1; }).get();
            }
        });
    }
    for (auto& t : producers) {
        t.join();
    }
    assert(sum.load() == 1600);
}

void stopRacingWithProducersResolvesAll() {
    GpuWorker worker([] {});
    std::vector<std::thread> producers;
    std::atomic<int> resolved{0};
    for (int p = 0; p < 4; ++p) {
        producers.emplace_back([&worker, &resolved] {
            for (int i = 0; i < 500; ++i) {
                try {
                    auto f = worker.submit([] { return 1; });
                    try {
                        f.get();
                    } catch (const GpuUnavailableError&) {
                    }
                } catch (const GpuUnavailableError&) {
                }
                ++resolved;  // ni submit ni get se cuelgan
            }
        });
    }
    worker.stop();
    for (auto& t : producers) {
        t.join();
    }
    assert(resolved.load() == 2000);
}

double processCpuSeconds() {
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

void idleWorkerDoesNotSpin() {
    GpuWorker worker([] {});
    worker.submit([] {}).get();  // ya arranco y volvio a esperar
    const double before = processCpuSeconds();
    std::promise<void> never;
    never.get_future().wait_for(400ms);  // el hilo de test espera bloqueado; solo el worker "corre"
    const double used = processCpuSeconds() - before;
    // Un spin consumiria ~0.4 s de CPU; esperando en la condition_variable, casi nada.
    assert(used < 0.05);
}

}  // namespace

int main() {
    fifoOrderAndResults();
    runsOnOneDedicatedThread();
    initRunsOnWorkerThread();
    taskExceptionGoesToItsFutureOnly();
    stopWithFullQueueResolvesEverything();
    submitAfterStopThrows();
    destructorStopsAndResolves();
    slowTaskAndExpiredDeadline();
    candidatesShareOneDeadline();
    deadlineEdgeCases();
    expiredTaskIsNotRunWhenDequeued();
    unhealthyAfterInitFailureFailsFast();
    fatalTaskErrorMakesWorkerUnhealthyForever();
    manyProducersConcurrently();
    stopRacingWithProducersResolvesAll();
    idleWorkerDoesNotSpin();
    std::cout << "test_gpu_worker OK\n";
    return 0;
}
