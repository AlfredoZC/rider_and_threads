#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "accel/CandidatePredicate.h"
#include "accel/GpuWorker.h"
#include "accel/ThreadPool.h"
#include "accel/TravelTimeTable.h"

namespace accel {

// Aplicacion 10: filtrar la flota antes de cotizar un pedido (filter / compactacion).
//
// La flota va como estructura de arreglos: un arreglo por campo y el courier es el indice.
struct FleetColumns {
    std::vector<std::int32_t> node;       // interseccion donde esta (indice en la tabla)
    std::vector<std::uint8_t> available;  // 1 = puede tomar un pedido (no averiado, no en descanso, con lugar)
    std::size_t size() const { return node.size(); }
};

// Candidatos en orden creciente de indice. Lanzan std::invalid_argument si las columnas no tienen el
// mismo tamano, o si el restaurante o el nodo de un courier disponible esta fuera de la tabla.
std::vector<std::int32_t> filterCandidatesSingleThread(const FleetColumns& fleet, const TravelTimeTable& table,
                                                       std::int32_t restaurantNode, std::int32_t maxTravelMs);
// Cada franja de la flota se filtra en su PROPIA lista y despues se concatenan en orden de franja:
// sin candados y con el mismo resultado, en el mismo orden, que 1 hilo.
std::vector<std::int32_t> filterCandidatesMultiThread(const FleetColumns& fleet, const TravelTimeTable& table,
                                                      std::int32_t restaurantNode, std::int32_t maxTravelMs,
                                                      ThreadPool& pool);

// Memoria de la GPU para el filtro: la tabla (se sube una vez y queda) y la flota. Este encabezado no
// incluye CUDA: lo implementan accel/cuda/CandidateFilter.cu (con -DUSE_CUDA=ON) y
// accel/cpu/CandidateFilterCpu.cpp (sin CUDA: todo devuelve "compilado sin CUDA"). Cada funcion devuelve ""
// o un mensaje; ninguna lanza. Los errores de ejecucion dejan el error CUDA pendiente (ver ApspDevice).
class CandidateFilterDevice {
public:
    CandidateFilterDevice() = default;
    ~CandidateFilterDevice();
    CandidateFilterDevice(const CandidateFilterDevice&) = delete;
    CandidateFilterDevice& operator=(const CandidateFilterDevice&) = delete;

    // Reserva y copia la tabla (nodeCount^2 enteros). `elapsedMs` (opcional): tiempo con cudaEvent.
    std::string uploadTable(const std::int32_t* cells, std::size_t nodeCount, double* elapsedMs);
    // Reserva lugar para `courierCount` couriers (flota + lista de candidatos + contador).
    std::string reserveFleet(std::size_t courierCount);
    // Copia las columnas de la flota (reserva si hace falta).
    std::string uploadFleet(const FleetColumns& fleet, double* elapsedMs);
    // Pone el contador en 0 y lanza el kernel. naive = un atomico global por candidato; si no, lista
    // privada por bloque en memoria compartida y un atomico global por bloque. `elapsedMs`: cudaEvent.
    std::string runFilter(bool naive, std::int32_t restaurantNode, std::int32_t maxTravelMs, double* elapsedMs);
    // Trae cuantos hay y despues solo esos (espera al kernel). El ORDEN no esta fijado: el conjunto si.
    std::string download(std::vector<std::int32_t>& candidates, double* elapsedMs);
    void release();

    std::size_t tableNodeCount() const { return tableNodeCount_; }

private:
    void releaseFleet();
    void* tablePointer_ = nullptr;
    std::size_t tableNodeCount_ = 0;
    void* nodePointer_ = nullptr;
    void* availablePointer_ = nullptr;
    void* candidatesPointer_ = nullptr;
    void* countPointer_ = nullptr;
    std::size_t fleetCapacity_ = 0;
    std::size_t fleetSize_ = 0;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

// Servicio del Dispatcher: filtra en la GPU (el GpuWorker, con la tabla ya residente en el device) con un
// plazo en tiempo REAL y, si no hay GPU, falla o vence, en CPU. Lo llama un solo hilo (el Dispatcher).
class CandidateFilter {
public:
    explicit CandidateFilter(GpuWorker* gpu);
    ~CandidateFilter();
    CandidateFilter(const CandidateFilter&) = delete;
    CandidateFilter& operator=(const CandidateFilter&) = delete;

    // Candidatos (indices en orden creciente). `timeoutMs`: plazo real de la GPU; si vence, se filtra en CPU.
    std::vector<std::int32_t> filter(const std::shared_ptr<const TravelTimeTable>& table, FleetColumns fleet,
                                     std::int32_t restaurantNode, std::int32_t maxTravelMs, std::int64_t timeoutMs);

    // Cuantas veces se uso cada camino (para las pruebas).
    std::size_t gpuFilters() const { return gpuFilters_; }
    std::size_t cpuFilters() const { return cpuFilters_; }

private:
    struct DeviceState;
    GpuWorker* gpu_;
    std::shared_ptr<DeviceState> device_;
    std::size_t gpuFilters_ = 0;
    std::size_t cpuFilters_ = 0;
};

}  // namespace accel
