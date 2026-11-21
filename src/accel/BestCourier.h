#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accel/CandidatePredicate.h"  // ACCEL_HOST_DEVICE
#include "accel/ThreadPool.h"
#include "accel/TravelTimeTable.h"  // kUnreachableMs

namespace accel {

// Aplicacion 08: elegir el mejor courier de cada pedido = argmin de cada fila de la matriz de ETA.
//
// La clave de 64 bits junta la ETA (32 bits altos) y el id REAL del courier (32 bajos, no la posicion en la
// fila): el minimo de las claves es el de menor ETA y, a igual ETA, el de menor id, el mismo desempate del
// Dispatcher. Con una sola reduccion de minimo se obtiene la ETA y el courier a la vez.
ACCEL_HOST_DEVICE inline unsigned long long packCourierKey(std::int32_t etaMs, std::int32_t courierId) {
    const std::uint32_t etaBits = static_cast<std::uint32_t>(etaMs < 0 ? 0 : etaMs);  // la ETA nunca es negativa
    return (static_cast<unsigned long long>(etaBits) << 32) | static_cast<std::uint32_t>(courierId);
}
ACCEL_HOST_DEVICE inline unsigned long long minKey(unsigned long long first, unsigned long long second) {
    return first < second ? first : second;
}
// Identidad del minimo: mayor que cualquier clave real; un hilo sin datos no gana nunca.
inline constexpr unsigned long long kNoCourierKey = ~0ULL;

// El mejor de un pedido. courierId == kNoBestCourier si no hay ninguno (fila vacia o todos inalcanzables).
inline constexpr std::int32_t kNoBestCourier = -1;
struct BestChoice {
    std::int32_t courierId = kNoBestCourier;
    std::int32_t etaMs = kUnreachableMs;
};
BestChoice decodeBestKey(unsigned long long key);

// `etaMs` es la matriz orderCount x courierIds.size() por filas (la de EtaBatch); courierIds[c] es el id REAL
// del courier de la columna c. Lanzan std::invalid_argument si los tamanos no coinciden. Un pedido sin
// couriers alcanzables da kNoBestCourier.
std::vector<BestChoice> bestCourierSingleThread(const std::vector<std::int32_t>& etaMs,
                                                const std::vector<std::int32_t>& courierIds, std::size_t orderCount);
// Cada hilo reduce su franja de columnas a un minimo parcial por pedido; despues se combinan.
std::vector<BestChoice> bestCourierMultiThread(const std::vector<std::int32_t>& etaMs,
                                               const std::vector<std::int32_t>& courierIds, std::size_t orderCount,
                                               ThreadPool& pool);

// Memoria de la GPU para el argmin. Este encabezado no incluye CUDA: lo implementan accel/cuda/BestCourier.cu y
// accel/cpu/BestCourierCpu.cpp (sin CUDA: todo devuelve "compilado sin CUDA"). Cada funcion devuelve "" o un
// mensaje; ninguna lanza.
class BestCourierDevice {
public:
    BestCourierDevice() = default;
    ~BestCourierDevice();
    BestCourierDevice(const BestCourierDevice&) = delete;
    BestCourierDevice& operator=(const BestCourierDevice&) = delete;

    // Copia una matriz de ETA propia (el bench). En el programa la matriz ya esta en el device (la dejo la ETA en
    // lote) y se pasa su puntero a run().
    std::string uploadEta(const std::int32_t* etaMs, std::size_t orderCount, std::size_t courierCount, double* elapsedMs);
    std::string uploadCourierIds(const std::int32_t* courierIds, std::size_t courierCount, double* elapsedMs);
    // Un bloque por pedido. externalEta == nullptr usa la matriz de uploadEta. divergent = arbol que empareja
    // vecinos (hilos activos desparramados); si no, el arbol convergente (los activos son siempre los primeros).
    std::string run(const void* externalEta, std::size_t orderCount, std::size_t courierCount, bool divergent,
                    double* elapsedMs);
    // Trae una clave por pedido (espera al kernel).
    std::string download(std::vector<unsigned long long>& keys, double* elapsedMs);
    void release();

private:
    void* etaPointer_ = nullptr;
    std::size_t etaCapacity_ = 0;
    void* idsPointer_ = nullptr;
    std::size_t idsCapacity_ = 0;
    void* keysPointer_ = nullptr;
    std::size_t keysCapacity_ = 0;
    std::size_t orderCount_ = 0;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

}  // namespace accel
