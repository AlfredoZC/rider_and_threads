#include "accel/BestCourier.h"

#include <limits>
#include <stdexcept>

namespace accel {

namespace {

void validateBestInput(const std::vector<std::int32_t>& etaMs, const std::vector<std::int32_t>& courierIds,
                       std::size_t orderCount) {
    if (courierIds.size() != 0 && orderCount > std::numeric_limits<std::size_t>::max() / courierIds.size()) {
        throw std::invalid_argument("argmin: la matriz no cabe en memoria");
    }
    if (etaMs.size() != orderCount * courierIds.size()) {
        throw std::invalid_argument("argmin: la matriz no tiene orderCount x courierIds.size() celdas");
    }
}

}  // namespace

BestChoice decodeBestKey(unsigned long long key) {
    BestChoice choice;
    if (key == kNoCourierKey) {
        return choice;
    }
    const std::int32_t etaMs = static_cast<std::int32_t>(key >> 32);
    if (etaMs >= kUnreachableMs) {
        return choice;  // el mejor es "inalcanzable": nadie puede hacer el pedido
    }
    choice.etaMs = etaMs;
    choice.courierId = static_cast<std::int32_t>(key & 0xFFFFFFFFULL);
    return choice;
}

std::vector<BestChoice> bestCourierSingleThread(const std::vector<std::int32_t>& etaMs,
                                                const std::vector<std::int32_t>& courierIds, std::size_t orderCount) {
    validateBestInput(etaMs, courierIds, orderCount);
    std::vector<BestChoice> best(orderCount);
    for (std::size_t order = 0; order < orderCount; ++order) {
        unsigned long long bestKey = kNoCourierKey;
        for (std::size_t column = 0; column < courierIds.size(); ++column) {
            bestKey = minKey(bestKey, packCourierKey(etaMs[order * courierIds.size() + column], courierIds[column]));
        }
        best[order] = decodeBestKey(bestKey);
    }
    return best;
}

std::vector<BestChoice> bestCourierMultiThread(const std::vector<std::int32_t>& etaMs,
                                               const std::vector<std::int32_t>& courierIds, std::size_t orderCount,
                                               ThreadPool& pool) {
    validateBestInput(etaMs, courierIds, orderCount);
    const std::size_t columnCount = courierIds.size();
    const std::size_t stripeCount = pool.threads();
    // partialBestKey[stripe * orderCount + order]: cada hilo escribe solo las celdas de su franja.
    std::vector<unsigned long long> partialBestKey(stripeCount * orderCount, kNoCourierKey);
    pool.parallelFor(stripeCount, [&](std::size_t firstStripe, std::size_t endStripe) {
        for (std::size_t stripe = firstStripe; stripe < endStripe; ++stripe) {
            const Range columns = chunkRange(columnCount, stripeCount, stripe);
            for (std::size_t order = 0; order < orderCount; ++order) {
                unsigned long long stripeBestKey = kNoCourierKey;
                for (std::size_t column = columns.begin; column < columns.end; ++column) {
                    stripeBestKey =
                        minKey(stripeBestKey, packCourierKey(etaMs[order * columnCount + column], courierIds[column]));
                }
                partialBestKey[stripe * orderCount + order] = stripeBestKey;
            }
        }
    });
    // Combinar los parciales en el hilo que llamo (despues de la barrera de parallelFor).
    std::vector<BestChoice> best(orderCount);
    for (std::size_t order = 0; order < orderCount; ++order) {
        unsigned long long bestKey = kNoCourierKey;
        for (std::size_t stripe = 0; stripe < stripeCount; ++stripe) {
            bestKey = minKey(bestKey, partialBestKey[stripe * orderCount + order]);
        }
        best[order] = decodeBestKey(bestKey);
    }
    return best;
}

}  // namespace accel
