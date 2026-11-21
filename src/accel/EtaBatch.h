#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "accel/BestCourier.h"
#include "accel/EtaFormula.h"
#include "accel/GpuWorker.h"
#include "accel/ThreadPool.h"
#include "accel/TravelTimeTable.h"

namespace accel {

// Aplicacion 04: ETA en lote, una matriz pedidos x couriers (fila = pedido, columna = courier).

// Las columnas de la matriz como estructura de arreglos (SoA): un arreglo por campo.
struct CourierColumns {
    std::vector<std::int32_t> node;      // interseccion donde esta
    std::vector<std::int32_t> freeAtMs;  // cuando queda libre, relativo a ahora
    std::size_t size() const { return node.size(); }
};

// El mismo courier como arreglo de estructuras (AoS), como lo guardaria un std::vector<Courier>. Los campos
// que la ETA no usa viajan igual por el bus de memoria: sirve para medir el costo de ese diseno.
struct CourierRecord {
    std::int32_t node;
    std::int32_t freeAtMs;
    std::int32_t bagCount;
    std::int32_t bagCapacity;
    std::int32_t state;
    std::int32_t deliveredCount;
    std::int32_t breakdownCount;
    std::int32_t restUntilMs;
};
static_assert(sizeof(CourierRecord) == 32, "un CourierRecord ocupa 32 bytes");

// Los pedidos del lote, tambien SoA.
struct OrderColumns {
    std::vector<std::int32_t> restaurantNode;
    std::vector<std::int32_t> customerNode;
    std::vector<std::int32_t> readyAtMs;  // cuando esta lista la comida, relativo a ahora
    std::size_t size() const { return restaurantNode.size(); }
};

// Columnas compactas: copia (gather) los campos de los couriers `courierIds` (indices de `fleet`), en ese orden.
// Lanza std::out_of_range si algun indice no existe.
CourierColumns gatherColumns(const CourierColumns& fleet, const std::vector<std::int32_t>& courierIds);
// La misma flota como registros AoS (para el bench): los campos que la ETA no usa se rellenan con datos fijos.
std::vector<CourierRecord> toRecords(const CourierColumns& columns);

// Matriz de ETA (orders.size() filas x couriers.size() columnas, por filas). Lanzan std::invalid_argument si
// las columnas no tienen el mismo tamano o si algun nodo esta fuera de la tabla.
std::vector<std::int32_t> computeEtaSingleThread(const CourierColumns& couriers, const OrderColumns& orders,
                                                 const TravelTimeTable& table);
// Las columnas se reparten en franjas; cada hilo llena las suyas en todas las filas.
std::vector<std::int32_t> computeEtaMultiThread(const CourierColumns& couriers, const OrderColumns& orders,
                                                const TravelTimeTable& table, ThreadPool& pool);

// Memoria de la GPU para la ETA en lote (la tabla se sube una vez y queda). Este encabezado no incluye CUDA:
// lo implementan accel/cuda/EtaBatch.cu (con -DUSE_CUDA=ON) y accel/cpu/EtaBatchCpu.cpp (sin CUDA: todo devuelve
// "compilado sin CUDA"). Cada funcion devuelve "" o un mensaje; ninguna lanza.
class EtaDevice {
public:
    EtaDevice() = default;
    ~EtaDevice();
    EtaDevice(const EtaDevice&) = delete;
    EtaDevice& operator=(const EtaDevice&) = delete;

    std::string uploadTable(const std::int32_t* cells, std::size_t nodeCount, double* elapsedMs);
    // Copian las columnas de los couriers (SoA o AoS) y los pedidos; reservan lo que haga falta.
    std::string uploadCouriersSoA(const CourierColumns& couriers, double* elapsedMs);
    std::string uploadCouriersAoS(const std::vector<CourierRecord>& couriers, double* elapsedMs);
    std::string uploadOrders(const OrderColumns& orders, double* elapsedMs);
    // Un hilo por par (pedido, courier), el courier en el eje x. aos = lee los registros de 32 bytes.
    std::string runEta(bool aos, double* elapsedMs);
    std::string download(std::vector<std::int32_t>& eta, double* elapsedMs);
    // Puntero a la matriz de ETA en el device (valido hasta la proxima runEta que la agrande): lo usa el argmin.
    const void* etaPointer() const { return etaPointer_; }
    void release();

private:
    std::string ensure(void** pointer, std::size_t& capacity, std::size_t count, std::size_t elementBytes,
                       const char* what);
    void* tablePointer_ = nullptr;
    std::size_t tableNodeCount_ = 0;
    void* soaNodePointer_ = nullptr;
    void* soaFreePointer_ = nullptr;
    void* aosPointer_ = nullptr;
    void* restaurantPointer_ = nullptr;
    void* customerPointer_ = nullptr;
    void* readyPointer_ = nullptr;
    void* etaPointer_ = nullptr;
    std::size_t soaCapacity_ = 0;
    std::size_t aosCapacity_ = 0;
    std::size_t orderCapacity_ = 0;
    std::size_t etaCapacity_ = 0;
    std::size_t courierCount_ = 0;
    std::size_t orderCount_ = 0;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

// La fila de ETA de un pedido y el mejor courier de esa fila (aplicacion 08), con kNoBestCourier si no hay ninguno.
struct EtaWithBest {
    std::vector<std::int32_t> eta;
    std::int32_t bestCourierId = kNoBestCourier;
};

// Servicio del Dispatcher: calcula la matriz de ETA en la GPU (GpuWorker, tabla residente en el device) con un
// plazo en tiempo REAL y, si no hay GPU, falla o vence, en CPU con la misma formula. Lo llama un solo hilo.
class EtaBatch {
public:
    explicit EtaBatch(GpuWorker* gpu);
    ~EtaBatch();
    EtaBatch(const EtaBatch&) = delete;
    EtaBatch& operator=(const EtaBatch&) = delete;

    std::vector<std::int32_t> compute(const std::shared_ptr<const TravelTimeTable>& table, CourierColumns couriers,
                                      OrderColumns orders, std::int64_t timeoutMs);

    // Igual que compute() para UN pedido, y ademas el argmin de la fila (id real del courier, menor id en empate):
    // en la GPU va en la MISMA tarea que la ETA; el respaldo CPU es identico. courierIds[c] es el id de la columna c.
    EtaWithBest computeWithBest(const std::shared_ptr<const TravelTimeTable>& table, CourierColumns couriers,
                                OrderColumns orders, std::vector<std::int32_t> courierIds, std::int64_t timeoutMs);

    std::size_t gpuBatches() const { return gpuBatches_; }
    std::size_t gpuBests() const { return gpuBests_; }
    std::size_t cpuBests() const { return cpuBests_; }
    std::size_t cpuBatches() const { return cpuBatches_; }

private:
    EtaWithBest run(const std::shared_ptr<const TravelTimeTable>& table, CourierColumns couriers, OrderColumns orders,
                    const std::vector<std::int32_t>* courierIds, std::int64_t timeoutMs);
    struct DeviceState;
    GpuWorker* gpu_;
    std::shared_ptr<DeviceState> device_;
    std::size_t gpuBatches_ = 0;
    std::size_t cpuBatches_ = 0;
    std::size_t gpuBests_ = 0;
    std::size_t cpuBests_ = 0;
};

}  // namespace accel
