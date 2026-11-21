#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accel/GpuWorker.h"
#include "accel/ThreadPool.h"

namespace accel {

// Percentiles p50, p90 y p99 del tiempo de entrega. Se ordenan los tiempos con radix sort LSD de 1 bit
// por pasada (del bit menos significativo al mas significativo, solo los bits del maximo): cada pasada
// separa de forma ESTABLE los que tienen ese bit en 0 (primero) de los que lo tienen en 1; la posicion de
// cada clave sale de un scan exclusivo de "tiene el bit en 0". Despues se lee el percentil de rango mas
// cercano en el arreglo ordenado.

// Cuantos bits hay que mirar: los del numero mas grande (si el maximo es 0, ninguno).
int significantBits(std::uint32_t maxKey);

// Posicion (desde 0) del percentil `percent` de `count` datos ordenados: ceil(p * N / 100) - 1, en
// enteros. Requiere count > 0.
std::size_t nearestRankIndex(unsigned percent, std::size_t count);

struct DeliveryPercentiles {
    std::uint32_t p50Ms = 0;
    std::uint32_t p90Ms = 0;
    std::uint32_t p99Ms = 0;
};

// Percentiles de un arreglo YA ordenado y no vacio.
DeliveryPercentiles percentilesFromSorted(const std::vector<std::uint32_t>& sortedMs);

// UNA pasada del radix sort: reubica las claves de forma estable, ceros del `bit` primero.
std::vector<std::uint32_t> radixSortPassCpu(const std::vector<std::uint32_t>& keys, int bit);

// CPU, 1 hilo: todas las pasadas necesarias.
std::vector<std::uint32_t> radixSortCpu1(std::vector<std::uint32_t> keys);

// CPU, N hilos. En cada pasada: (1) cada franja cuenta sus ceros; (2) un scan de esos N numeros dice donde
// arranca cada franja; (3) cada franja reubica sus claves con sus propios cursores (posiciones que solo
// ella calcula: nadie escribe la misma casilla).
std::vector<std::uint32_t> radixSortCpuN(std::vector<std::uint32_t> keys, ThreadPool& pool);

// Memoria del dispositivo para el radix sort (tres buffers de claves + marcas + scan + auxiliar + dos
// cudaEvent). Este encabezado no incluye nada de CUDA: lo implementan accel/cuda/DeliveryPercentiles.cu
// (usa accel/cuda/Scan.cu, el mismo scan que los pedidos en curso) y, sin CUDA,
// accel/cpu/DeliveryPercentilesCpu.cpp (todo devuelve "compilado sin CUDA"). Cada funcion devuelve ""
// si salio bien o un mensaje de error; ninguna lanza ni aborta. `elapsedMs` (opcional) recibe el tiempo
// medido con cudaEvent.
class DeviceRadixSort {
public:
    DeviceRadixSort() = default;
    ~DeviceRadixSort();
    DeviceRadixSort(const DeviceRadixSort&) = delete;
    DeviceRadixSort& operator=(const DeviceRadixSort&) = delete;

    // Reserva espacio para `keyCapacity` claves (libera antes lo que hubiera).
    std::string allocate(std::size_t keyCapacity);
    void release();

    // Copia las claves al dispositivo (count <= keyCapacity) y calcula cuantos bits hay que ordenar.
    std::string upload(const std::uint32_t* hostKeys, std::size_t count, double* elapsedMs);
    // bitCount pasadas de marcar + scan + reubicar; la primera lee las claves originales y las demas alternan dos buffers.
    std::string run(double* elapsedMs);
    // Trae las claves ordenadas (count enteros).
    std::string download(std::uint32_t* hostSortedKeys, double* elapsedMs);

private:
    // [0] claves originales (no se tocan: cada corrida parte de ellas), [1] y [2] se alternan entre pasadas.
    std::uint32_t* deviceKeys_[3] = {nullptr, nullptr, nullptr};
    std::int32_t* deviceIsZero_ = nullptr;
    std::int32_t* deviceZerosUpTo_ = nullptr;
    std::int32_t* deviceScratch_ = nullptr;
    std::size_t keyCapacity_ = 0;
    std::size_t uploadedCount_ = 0;
    int bitCount_ = 0;
    int sortedBuffer_ = 0;  // cual de los buffers de claves quedo ordenado despues de run
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

// Todo el camino en GPU: reservar, subir, ordenar, traer. Devuelve "" o un mensaje de error.
// Pensada para correr DENTRO de una tarea del GpuWorker.
std::string radixSortGpu(const std::vector<std::uint32_t>& keys, std::vector<std::uint32_t>& sortedKeys);

struct DeliveryPercentilesSummary {
    DeliveryPercentiles percentiles;
    std::size_t count = 0;
    bool usedGpu = false;  // false: se uso el respaldo CPU
};

// Percentiles de los tiempos de entrega (ms, int64 como los da el OrderBook; los negativos se recortan a 0
// y los que no caben en uint32 al maximo). Lo pide al GpuWorker si hay uno y esta sano, y espera como mucho
// `timeout` en tiempo REAL; si no hay GPU, la tarea falla o vence, ordena en la CPU (1 hilo). Nunca lanza.
DeliveryPercentilesSummary deliveryPercentiles(const std::vector<std::int64_t>& deliveryTimesMs, GpuWorker* worker,
                                               std::chrono::milliseconds timeout = std::chrono::milliseconds(2000));

// Linea para stderr: "[estadisticas] percentiles de entrega: p50 17.0 min, p90 31.0 min, p99 47.0 min (backend=cpu)".
std::string formatDeliveryPercentiles(const DeliveryPercentiles& percentiles, std::size_t count,
                                      const std::string& backendName);

}  // namespace accel
