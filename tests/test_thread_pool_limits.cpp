#include "accel/ThreadPool.h"

#include <sys/resource.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <system_error>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

// Regresion de B1: si std::thread falla a medias en el constructor, los trabajadores ya creados
// no pueden quedar esperando en una condition_variable que se destruye (cuelgue) ni usar `this`
// destruido. Se fuerza el fallo con RLIMIT_AS; alarm() convierte un cuelgue en fallo del test.
int main() {
    alarm(60);

    rlimit originalLimit{};
    CHECK(getrlimit(RLIMIT_AS, &originalLimit) == 0);

    rlimit restrictedLimit = originalLimit;
    restrictedLimit.rlim_cur = std::size_t{600} << 20;  // 600 MiB: caben pocos hilos de 8 MiB de pila
    if (originalLimit.rlim_max != RLIM_INFINITY && restrictedLimit.rlim_cur > originalLimit.rlim_max) {
        restrictedLimit.rlim_cur = originalLimit.rlim_max;
    }
    CHECK(setrlimit(RLIMIT_AS, &restrictedLimit) == 0);

    bool threw = false;
    try {
        accel::ThreadPool pool(4000);
    } catch (const std::system_error&) {
        threw = true;
    } catch (const std::bad_alloc&) {
        threw = true;
    }
    CHECK(threw);

    // El proceso sigue sano: un pool normal funciona despues del fallo.
    CHECK(setrlimit(RLIMIT_AS, &originalLimit) == 0);
    {
        accel::ThreadPool pool(4);
        std::atomic<std::size_t> processedItems{0};
        pool.parallelFor(100, [&](std::size_t chunkBegin, std::size_t chunkEnd) { processedItems.fetch_add(chunkEnd - chunkBegin); });
        CHECK(processedItems.load() == 100u);
    }

    std::cout << "test_thread_pool_limits OK\n";
    return 0;
}
