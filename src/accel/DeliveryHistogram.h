#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accel/GpuWorker.h"
#include "accel/HostDevice.h"
#include "accel/ThreadPool.h"

namespace accel {

// Histograma de tiempos de entrega por minuto: una barra por minuto (0..59) y una ultima barra
// (la 60) que junta todo lo que tardo 60 minutos o mas. La suma de las barras es siempre la
// cantidad de entregas, sea cual sea el dato.
constexpr int kDeliveryHistogramBins = 61;
constexpr std::int32_t kMillisecondsPerMinute = 60 * 1000;

// Barra de una entrega: su minuto con division entera (14 min 59,999 s cae en la 14). Un tiempo
// negativo va a la primera barra y uno que se pasa de la ultima, a la ultima.
ACCEL_HOST_DEVICE inline int deliveryMinuteBin(std::int32_t deliveryTimeMs) {
    int minute = deliveryTimeMs / kMillisecondsPerMinute;
    if (minute < 0) {
        minute = 0;
    }
    if (minute > kDeliveryHistogramBins - 1) {
        minute = kDeliveryHistogramBins - 1;
    }
    return minute;
}

// CPU, 1 hilo: recorre las entregas una vez y suma 1 en la barra de cada una.
std::vector<unsigned> deliveryTimeHistogramCpu1(const std::vector<std::int32_t>& deliveryTimesMs);

// CPU, N hilos: cada trozo (ThreadPool::parallelFor) cuenta en un histograma PROPIO y al final
// los suma al total bajo un mutex (un candado por trozo, no uno por entrega).
std::vector<unsigned> deliveryTimeHistogramCpuN(const std::vector<std::int32_t>& deliveryTimesMs, ThreadPool& pool);

// Que kernel de GPU usar.
//  privatized     cada bloque cuenta en su copia del histograma en memoria compartida y al final
//                 suma sus barras al global (un atomico global por barra no vacia y por bloque).
//  globalAtomics  version ingenua: un atomico global por entrega; todas las entregas del mismo
//                 minuto compiten por la misma direccion (sirve para medir la contencion).
enum class HistogramKernel { privatized, globalAtomics };

// Memoria del dispositivo para el histograma (tiempos + barras + dos cudaEvent). Este encabezado no
// incluye nada de CUDA: lo implementan accel/cuda/DeliveryHistogram.cu y, sin CUDA,
// accel/cpu/DeliveryHistogramCpu.cpp (todo devuelve "compilado sin CUDA"). Cada funcion devuelve ""
// si salio bien o un mensaje de error; ninguna lanza ni aborta. `elapsedMs` (opcional) recibe el
// tiempo medido con cudaEvent.
class DeviceHistogram {
public:
    DeviceHistogram() = default;
    ~DeviceHistogram();
    DeviceHistogram(const DeviceHistogram&) = delete;
    DeviceHistogram& operator=(const DeviceHistogram&) = delete;

    // Reserva espacio para `timeCapacity` tiempos (libera antes lo que hubiera).
    std::string allocate(std::size_t timeCapacity);
    void release();

    // Copia los tiempos al dispositivo (count <= timeCapacity).
    std::string upload(const std::int32_t* hostTimesMs, std::size_t count, double* elapsedMs);
    // Pone las barras en cero (fuera de la medicion) y lanza el kernel sobre los tiempos subidos.
    std::string run(HistogramKernel kernel, double* elapsedMs);
    // Trae las kDeliveryHistogramBins barras al host.
    std::string download(unsigned* hostBins, double* elapsedMs);

private:
    std::int32_t* deviceTimes_ = nullptr;
    unsigned* deviceBins_ = nullptr;
    std::size_t timeCapacity_ = 0;
    std::size_t uploadedCount_ = 0;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

// Todo el camino en GPU: reservar, subir, lanzar, traer. Devuelve "" o un mensaje de error.
// Pensada para correr DENTRO de una tarea del GpuWorker.
std::string deliveryTimeHistogramGpu(const std::vector<std::int32_t>& deliveryTimesMs,
                                     std::vector<unsigned>& bins,
                                     HistogramKernel kernel = HistogramKernel::privatized);

struct DeliveryHistogramResult {
    std::vector<unsigned> bins;
    bool usedGpu = false;  // false: se uso el respaldo CPU
};

// Histograma de los tiempos de entrega (ms, int64 como los da el OrderBook; los que no caben en
// int32 se recortan a los bordes). Lo pide al GpuWorker si hay uno y esta sano, y espera como mucho
// `timeout` en tiempo REAL; si no hay GPU, la tarea falla o vence, usa la CPU (1 hilo). Nunca lanza.
DeliveryHistogramResult deliveryTimeHistogram(const std::vector<std::int64_t>& deliveryTimesMs,
                                              GpuWorker* worker,
                                              std::chrono::milliseconds timeout = std::chrono::milliseconds(2000));

// Linea para stderr: "[estadisticas] histograma de entrega (min): 14:4 22:2 60+:1 (n=7, backend=cpu)",
// solo con las barras no vacias.
std::string formatDeliveryHistogram(const std::vector<unsigned>& bins, const std::string& backendName);

}  // namespace accel
