#include "accel/ThreadPool.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

int main() {
    // chunkRange: los trozos cubren [0, itemCount) exactamente, sin huecos ni solapes, y difieren en <= 1.
    for (std::size_t itemCount : {0u, 1u, 2u, 3u, 7u, 8u, 100u, 1001u}) {
        for (std::size_t parts : {1u, 2u, 3u, 4u, 8u, 16u}) {
            std::size_t expectedBegin = 0;
            std::size_t shortestChunk = itemCount;
            std::size_t longestChunk = 0;
            for (std::size_t partIndex = 0; partIndex < parts; ++partIndex) {
                const accel::Range range = accel::chunkRange(itemCount, parts, partIndex);
                CHECK(range.begin == expectedBegin);
                CHECK(range.end >= range.begin);
                shortestChunk = std::min(shortestChunk, range.end - range.begin);
                longestChunk = std::max(longestChunk, range.end - range.begin);
                expectedBegin = range.end;
            }
            CHECK(expectedBegin == itemCount);
            CHECK(longestChunk - shortestChunk <= 1);
        }
    }
    // Sin desbordar con itemCount cercano al maximo.
    {
        const std::size_t maxItemCount = static_cast<std::size_t>(-1);
        CHECK(accel::chunkRange(maxItemCount, 4, 3).end == maxItemCount);
        CHECK(accel::chunkRange(maxItemCount, 4, 0).begin == 0);
    }

    // parallelFor: cada indice se visita exactamente una vez, para varios tamanos de pool.
    for (unsigned threadCount : {1u, 2u, 3u, 8u}) {
        accel::ThreadPool pool(threadCount);
        CHECK(pool.threads() == threadCount);
        for (std::size_t itemCount : {0u, 1u, 5u, 1000u}) {
            std::vector<std::atomic<int>> visitCounts(itemCount);
            for (auto& visitCount : visitCounts) visitCount.store(0);
            pool.parallelFor(itemCount, [&](std::size_t chunkBegin, std::size_t chunkEnd) {
                for (std::size_t item = chunkBegin; item < chunkEnd; ++item) visitCounts[item].fetch_add(1);
            });
            for (std::size_t item = 0; item < itemCount; ++item) CHECK(visitCounts[item].load() == 1);
        }
    }

    // El pool se reutiliza muchas veces (generaciones consecutivas) sin perder ni duplicar trabajo.
    {
        accel::ThreadPool pool(4);
        std::atomic<std::size_t> processedItems{0};
        for (int round = 0; round < 500; ++round) {
            pool.parallelFor(37, [&](std::size_t chunkBegin, std::size_t chunkEnd) {
                processedItems.fetch_add(chunkEnd - chunkBegin);
            });
        }
        CHECK(processedItems.load() == 500u * 37u);
    }

    // Una excepcion del cuerpo llega al llamador y el pool sigue usable.
    {
        accel::ThreadPool pool(3);
        bool caught = false;
        try {
            pool.parallelFor(30, [&](std::size_t chunkBegin, std::size_t) {
                if (chunkBegin > 0) throw std::runtime_error("boom");
            });
        } catch (const std::runtime_error&) {
            caught = true;
        }
        CHECK(caught);
        std::atomic<std::size_t> processedItems{0};
        pool.parallelFor(10, [&](std::size_t chunkBegin, std::size_t chunkEnd) {
            processedItems.fetch_add(chunkEnd - chunkBegin);
        });
        CHECK(processedItems.load() == 10u);
    }

    // threads == 0 se trata como 1 (nunca crea un pool sin nadie que trabaje).
    {
        accel::ThreadPool pool(0);
        CHECK(pool.threads() == 1);
    }

    // Destruir un pool recien creado no cuelga (todos los hilos terminan y se unen).
    { accel::ThreadPool pool(6); }

    std::cout << "test_thread_pool OK\n";
    return 0;
}
