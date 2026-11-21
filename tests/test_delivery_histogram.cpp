#include "accel/DeliveryHistogram.h"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "OrderBook.h"
#include "accel/GpuError.h"
#include "accel/GpuWorker.h"
#include "accel/ThreadPool.h"

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

using accel::kDeliveryHistogramBins;

std::uint64_t nextRandom(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t mixed = state;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
    return mixed ^ (mixed >> 31);
}

unsigned total(const std::vector<unsigned>& bins) {
    return std::accumulate(bins.begin(), bins.end(), 0u);
}

}  // namespace

int main() {
    // Cada entrega cae en la barra de su minuto (division entera) y lo que se pasa va a la ultima.
    CHECK(accel::deliveryMinuteBin(0) == 0);
    CHECK(accel::deliveryMinuteBin(59999) == 0);
    CHECK(accel::deliveryMinuteBin(60000) == 1);
    CHECK(accel::deliveryMinuteBin(899999) == 14);  // 14 min 59,999 s se trunca, no se redondea
    CHECK(accel::deliveryMinuteBin(3599999) == 59);
    CHECK(accel::deliveryMinuteBin(3600000) == kDeliveryHistogramBins - 1);
    CHECK(accel::deliveryMinuteBin(INT_MAX) == kDeliveryHistogramBins - 1);
    CHECK(accel::deliveryMinuteBin(-1) == 0);
    CHECK(accel::deliveryMinuteBin(INT_MIN) == 0);

    accel::ThreadPool pool(4);

    // Vacio: todas las barras en cero, con la cantidad de barras correcta.
    {
        const std::vector<std::int32_t> empty;
        const auto single = accel::deliveryTimeHistogramCpu1(empty);
        const auto multi = accel::deliveryTimeHistogramCpuN(empty, pool);
        CHECK(single.size() == static_cast<std::size_t>(kDeliveryHistogramBins));
        CHECK(total(single) == 0);
        CHECK(single == multi);
    }

    // Un elemento.
    {
        const std::vector<std::int32_t> one{845000};  // 14,08 min
        const auto single = accel::deliveryTimeHistogramCpu1(one);
        CHECK(single[14] == 1);
        CHECK(total(single) == 1);
        CHECK(accel::deliveryTimeHistogramCpuN(one, pool) == single);
    }

    // Todos en el mismo minuto: una sola barra con todo (peor caso de contencion).
    {
        const std::vector<std::int32_t> sameMinute(10001, 22 * 60000 + 17);
        const auto single = accel::deliveryTimeHistogramCpu1(sameMinute);
        CHECK(single[22] == 10001);
        CHECK(total(single) == 10001);
        CHECK(accel::deliveryTimeHistogramCpuN(sameMinute, pool) == single);
    }

    // Ejemplo a mano de 12 entregas (guia): barras 9, 14x4, 18x2, 22x2, 31, 47 y desborde 1.
    {
        const std::vector<std::int32_t> byHand{845000, 1102000, 871500, 1330000, 559000, 899999,
                                               1870000, 1080000, 4500000, 1345000, 840000, 2830000};
        const auto bins = accel::deliveryTimeHistogramCpu1(byHand);
        CHECK(bins[9] == 1 && bins[14] == 4 && bins[18] == 2 && bins[22] == 2);
        CHECK(bins[31] == 1 && bins[47] == 1 && bins[kDeliveryHistogramBins - 1] == 1);
        CHECK(total(bins) == 12);
        CHECK(accel::deliveryTimeHistogramCpuN(byHand, pool) == bins);
    }

    // Negativos y enormes van a los bordes; la suma de barras siempre es la cantidad.
    {
        const std::vector<std::int32_t> extremes{-5, INT_MIN, 0, INT_MAX, 3600000, 3599999, 7200000};
        const auto bins = accel::deliveryTimeHistogramCpu1(extremes);
        CHECK(bins[0] == 3);
        CHECK(bins[59] == 1);
        CHECK(bins[kDeliveryHistogramBins - 1] == 3);
        CHECK(total(bins) == extremes.size());
        CHECK(accel::deliveryTimeHistogramCpuN(extremes, pool) == bins);
    }

    // cpu1 == cpuN con datos pseudoaleatorios, tamanos chicos y no multiplos del numero de hilos.
    for (const std::size_t count : {2u, 3u, 5u, 1000u, 4099u, 100003u}) {
        std::uint64_t state = 7 + count;
        std::vector<std::int32_t> times(count);
        for (auto& deliveryTimeMs : times) {
            deliveryTimeMs = static_cast<std::int32_t>(nextRandom(state) % 5400000);  // 0..90 min
        }
        for (const unsigned threads : {1u, 2u, 3u, 8u}) {
            accel::ThreadPool threadPool(threads);
            const auto single = accel::deliveryTimeHistogramCpu1(times);
            const auto multi = accel::deliveryTimeHistogramCpuN(times, threadPool);
            CHECK(single == multi);
            CHECK(total(multi) == count);
        }
    }

    // Texto para stderr: solo barras no vacias, con la cantidad y el respaldo usado.
    {
        std::vector<unsigned> bins(kDeliveryHistogramBins, 0);
        bins[14] = 4;
        bins[22] = 2;
        bins[kDeliveryHistogramBins - 1] = 1;
        CHECK(accel::formatDeliveryHistogram(bins, "cpu") ==
              "[estadisticas] histograma de entrega (min): 14:4 22:2 60+:1 (n=7, backend=cpu)");
        const std::vector<unsigned> noBins(kDeliveryHistogramBins, 0);
        CHECK(accel::formatDeliveryHistogram(noBins, "cpu") ==
              "[estadisticas] histograma de entrega (min): sin entregas");
    }

    // Sin worker, con un worker roto y con uno sano pero sin CUDA (build CPU): el resultado es el mismo,
    // y los tiempos fuera de rango de int32 se recortan a los bordes.
    {
        const std::vector<std::int64_t> deliveryTimes{845000, 899999, 4500000, -3, 5000000000LL, 1330000};
        const std::vector<unsigned> expected = accel::deliveryTimeHistogramCpu1(
            std::vector<std::int32_t>{845000, 899999, 4500000, 0, INT_MAX, 1330000});

        const accel::DeliveryHistogramResult withoutWorker = accel::deliveryTimeHistogram(deliveryTimes, nullptr);
        CHECK(!withoutWorker.usedGpu);
        CHECK(withoutWorker.bins == expected);

        accel::GpuWorker brokenWorker([] { throw std::runtime_error("sin GPU"); });
        CHECK(!brokenWorker.healthy());
        const accel::DeliveryHistogramResult withBrokenWorker =
            accel::deliveryTimeHistogram(deliveryTimes, &brokenWorker);
        CHECK(!withBrokenWorker.usedGpu);
        CHECK(withBrokenWorker.bins == expected);
        brokenWorker.stop();

        accel::GpuWorker healthyWorker([] {});
        const accel::DeliveryHistogramResult withHealthyWorker =
            accel::deliveryTimeHistogram(deliveryTimes, &healthyWorker);
        CHECK(withHealthyWorker.bins == expected);
        healthyWorker.stop();
    }

    // OrderBook::deliveredTimesMs: solo los pedidos entregados, con su tiempo total.
    {
        const SimClock clock(60.0);
        EventLog log("tests/bin/histogram_book.log", clock);
        OrderBook book(10, clock, log);
        CHECK(book.deliveredTimesMs().empty());
        for (OrderId id = 1; id <= 3; ++id) {
            Order order;
            order.id = id;
            order.restaurantId = "r0";
            order.pickupNode = "n15";
            order.dropoffNode = "n420";
            CHECK(book.admit(order));
        }
        CHECK(book.assign(1, 7));
        CHECK(book.pickedUp(1, 7));
        CHECK(book.delivered(1, 7));
        const std::vector<std::int64_t> times = book.deliveredTimesMs();
        CHECK(times.size() == 1);
        CHECK(times[0] >= 0);
        CHECK(times[0] == book.stats().maxDeliveryMs);
    }

    std::cout << "test_delivery_histogram: OK\n";
    return 0;
}
