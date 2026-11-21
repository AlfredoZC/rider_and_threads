// Aplicacion 08: elegir el mejor courier de cada pedido (argmin por fila de la matriz de ETA). La clave de
// 64 bits (eta << 32 | id REAL) hace que el minimo sea el de menor ETA y, a igual ETA, el de menor id.
// 1 hilo == N hilos == una referencia independiente. La equivalencia GPU == CPU (arbol convergente y
// divergente) la prueba el bench (columna correct) con el build CUDA.
#include "accel/BestCourier.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "FALLO " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

namespace {

using accel::BestChoice;
using accel::kNoBestCourier;
using accel::kUnreachableMs;
using Ids = std::vector<std::int32_t>;
using Matrix = std::vector<std::int32_t>;

// Referencia independiente: bucles con comparaciones explicitas, sin claves.
std::vector<BestChoice> referenceBest(const Matrix& eta, const Ids& ids, std::size_t orderCount) {
    std::vector<BestChoice> best(orderCount);
    for (std::size_t order = 0; order < orderCount; ++order) {
        for (std::size_t column = 0; column < ids.size(); ++column) {
            const std::int32_t value = eta[order * ids.size() + column];
            if (value >= kUnreachableMs) continue;
            if (best[order].courierId == kNoBestCourier || value < best[order].etaMs ||
                (value == best[order].etaMs && ids[column] < best[order].courierId)) {
                best[order] = BestChoice{ids[column], value};
            }
        }
    }
    return best;
}

bool same(const std::vector<BestChoice>& left, const std::vector<BestChoice>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t order = 0; order < left.size(); ++order) {
        if (left[order].courierId != right[order].courierId || left[order].etaMs != right[order].etaMs) return false;
    }
    return true;
}

void testExample() {
    // Los 2 pedidos del material: columnas = couriers 1, 3, 4, 7. A: 9 9 11 9 (empate triple 1, 3, 7);
    // B: 6 6 9 5 (gana el 7).
    const Ids ids = {1, 3, 4, 7};
    const Matrix eta = {9, 9, 11, 9, 6, 6, 9, 5};
    const auto best = accel::bestCourierSingleThread(eta, ids, 2);
    CHECK(best[0].courierId == 1 && best[0].etaMs == 9);  // empate triple: gana el menor id
    CHECK(best[1].courierId == 7 && best[1].etaMs == 5);
    for (const unsigned threads : {1u, 2u, 3u, 4u, 8u}) {
        accel::ThreadPool pool(threads);
        CHECK(same(accel::bestCourierMultiThread(eta, ids, 2, pool), best));
    }
}

void testIdsNotPositions() {
    // Columnas desordenadas 7 4 1 3: el desempate es por id REAL (gana el 1, que esta en la posicion 2).
    const Ids ids = {7, 4, 1, 3};
    const Matrix eta = {5, 5, 5, 8,   // empate de 7, 4 y 1: gana el 1
                        9, 2, 2, 2};  // empate de 4, 1 y 3: gana el 1
    const auto best = accel::bestCourierSingleThread(eta, ids, 2);
    CHECK(best[0].courierId == 1 && best[0].etaMs == 5);
    CHECK(best[1].courierId == 1 && best[1].etaMs == 2);
    accel::ThreadPool pool(3);
    CHECK(same(accel::bestCourierMultiThread(eta, ids, 2, pool), best));
}

void testUnreachableAndEdges() {
    accel::ThreadPool pool(4);
    // Todos inalcanzables: no hay mejor.
    const Ids ids = {4, 2, 9};
    const Matrix allUnreachable = {kUnreachableMs, kUnreachableMs, kUnreachableMs};
    CHECK(accel::bestCourierSingleThread(allUnreachable, ids, 1)[0].courierId == kNoBestCourier);
    CHECK(accel::bestCourierMultiThread(allUnreachable, ids, 1, pool)[0].courierId == kNoBestCourier);
    // Uno solo alcanzable gana aunque tenga la ETA mas alta y el id mas alto.
    const Matrix oneReachable = {kUnreachableMs, kUnreachableMs, 1000000};
    CHECK(accel::bestCourierSingleThread(oneReachable, ids, 1)[0].courierId == 9);
    // Una sola columna.
    const auto single = accel::bestCourierMultiThread(Matrix{42, 17}, Ids{5}, 2, pool);
    CHECK(single[0].courierId == 5 && single[0].etaMs == 42 && single[1].etaMs == 17);
    // ETA 0 con el id 0 es una clave valida (no se confunde con "sin dato").
    const auto zero = accel::bestCourierSingleThread(Matrix{0, 3}, Ids{0, 1}, 1);
    CHECK(zero[0].courierId == 0 && zero[0].etaMs == 0);
    // Sin pedidos, sin couriers.
    CHECK(accel::bestCourierSingleThread(Matrix{}, ids, 0).empty());
    CHECK(accel::bestCourierMultiThread(Matrix{}, ids, 0, pool).empty());
    CHECK(accel::bestCourierSingleThread(Matrix{}, Ids{}, 3)[1].courierId == kNoBestCourier);
    CHECK(accel::bestCourierMultiThread(Matrix{}, Ids{}, 3, pool)[2].courierId == kNoBestCourier);
    // Tamanos que no coinciden.
    bool threw = false;
    try {
        (void)accel::bestCourierSingleThread(Matrix{1, 2, 3}, Ids{1, 2}, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    // La clave: ETA en los bits altos, id en los bajos; el menor id gana el empate y la ETA manda sobre el id.
    CHECK(accel::packCourierKey(5, 9) < accel::packCourierKey(6, 1));
    CHECK(accel::packCourierKey(5, 2) < accel::packCourierKey(5, 3));
    CHECK(accel::decodeBestKey(accel::kNoCourierKey).courierId == kNoBestCourier);
}

void testRandomMatrices() {
    std::mt19937 generator(3);
    for (const auto& shape : {std::pair<std::size_t, std::size_t>{1, 1}, {10, 50}, {30, 150}, {7, 333}, {100, 500}, {3, 2001}}) {
        Ids ids;
        for (std::size_t column = 0; column < shape.second; ++column) {
            ids.push_back(static_cast<std::int32_t>(1 + (column * 7919) % 100003));  // ids no ordenados
        }
        Matrix eta;
        for (std::size_t order = 0; order < shape.first; ++order) {
            for (std::size_t column = 0; column < shape.second; ++column) {
                // Valores chicos: muchos empates. Algunos inalcanzables. En algunas filas, todos iguales.
                eta.push_back(order % 4 == 3 ? 7 : (generator() % 10 == 0 ? kUnreachableMs : static_cast<std::int32_t>(generator() % 20)));
            }
        }
        const auto reference = referenceBest(eta, ids, shape.first);
        CHECK(same(accel::bestCourierSingleThread(eta, ids, shape.first), reference));
        for (const unsigned threads : {2u, 4u, 7u, 16u}) {
            accel::ThreadPool pool(threads);
            CHECK(same(accel::bestCourierMultiThread(eta, ids, shape.first, pool), reference));
        }
    }
}

}  // namespace

int main() {
    testExample();
    testIdsNotPositions();
    testUnreachableAndEdges();
    testRandomMatrices();
    std::cout << "TODO OK\n";
    return 0;
}
