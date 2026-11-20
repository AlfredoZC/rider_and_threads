#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "accel/GpuWorker.h"
#include "accel/ThreadPool.h"

namespace accel {

// Pedidos abiertos (creados y todavia no entregados ni rechazados) a lo largo del tiempo y su pico.
// Cada pedido aporta un evento +1 al crearse y un -1 al cerrarse; ordenados por tiempo (en un empate
// el +1 va primero, asi la cuenta nunca baja de 0), la suma acumulada despues de cada evento (scan
// inclusivo) es la cantidad de pedidos abiertos en ese momento, y su maximo es el pico.

constexpr std::int64_t kStillOpen = -1;  // closedAtMs de un pedido que seguia abierto al terminar

struct OrderLifetime {
    std::int64_t createdAtMs = 0;
    std::int64_t closedAtMs = kStillOpen;  // entregado o rechazado
};

struct OrderEvents {
    std::vector<std::int32_t> timeMs;  // cuando ocurrio cada evento (los tiempos se recortan a int32)
    std::vector<std::int32_t> change;  // +1 (pedido creado) o -1 (pedido entregado o rechazado)
};

// Eventos ordenados por tiempo, con el +1 antes que el -1 en los empates.
OrderEvents buildOrderEvents(const std::vector<OrderLifetime>& orders);

struct OpenOrdersResult {
    std::vector<std::int32_t> openOrders;  // pedidos abiertos justo despues de cada evento
    std::int32_t peakOpenOrders = 0;       // el pico (0 si nunca hubo pedidos abiertos)
    std::size_t peakEvent = 0;             // primer evento en que se alcanza el pico
};

// CPU, 1 hilo: suma acumulada y, en la misma pasada, el maximo.
OpenOrdersResult openOrdersCpu1(const std::vector<std::int32_t>& change);

// CPU, N hilos: pasada 1 (cada franja suma sus cambios), un paso chico que suma los totales de las
// franjas anteriores y pasada 2 (cada franja hace su scan desde su offset y anota su pico). Cada hilo
// escribe solo en su franja y en su casilla de los vectores por franja: no hay carreras.
OpenOrdersResult openOrdersCpuN(const std::vector<std::int32_t>& change, ThreadPool& pool);

// Memoria del dispositivo para el scan y el pico (cambios + pedidos abiertos + auxiliar + clave del
// pico + dos cudaEvent). Este encabezado no incluye nada de CUDA: lo implementan
// accel/cuda/OpenOrders.cu y, sin CUDA, accel/cpu/OpenOrdersCpu.cpp (todo devuelve "compilado sin
// CUDA"). Cada funcion devuelve "" si salio bien o un mensaje de error; ninguna lanza ni aborta.
// `elapsedMs` (opcional) recibe el tiempo medido con cudaEvent.
class DeviceOpenOrders {
public:
    DeviceOpenOrders() = default;
    ~DeviceOpenOrders();
    DeviceOpenOrders(const DeviceOpenOrders&) = delete;
    DeviceOpenOrders& operator=(const DeviceOpenOrders&) = delete;

    // Reserva espacio para `eventCapacity` eventos (libera antes lo que hubiera).
    std::string allocate(std::size_t eventCapacity);
    void release();

    std::string upload(const std::int32_t* hostChange, std::size_t count, double* elapsedMs);
    // Scan inclusivo jerarquico (Kogge-Stone por bloque de 512, doble buffer) + reduccion max del pico.
    std::string run(double* elapsedMs);
    // Trae los pedidos abiertos (count enteros) y la clave de 64 bits del pico.
    std::string download(std::int32_t* hostOpenOrders, unsigned long long* hostPeakKey, double* elapsedMs);

private:
    std::int32_t* deviceChange_ = nullptr;
    std::int32_t* deviceOpenOrders_ = nullptr;
    std::int32_t* deviceScratch_ = nullptr;
    unsigned long long* devicePeakKey_ = nullptr;
    std::size_t eventCapacity_ = 0;
    std::size_t uploadedCount_ = 0;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

// Clave de 64 bits del pico: pedidos abiertos (>= 0) en los 32 bits altos y (0xFFFFFFFF - evento) en
// los bajos. La clave maxima es el pico mas alto y, entre empates, el evento mas temprano.
inline unsigned long long peakKey(std::int32_t openOrders, std::uint32_t event) {
    const std::uint64_t nonNegative = openOrders > 0 ? static_cast<std::uint64_t>(openOrders) : 0;
    return (nonNegative << 32) | (0xFFFFFFFFu - event);
}

// Todo el camino en GPU: reservar, subir, lanzar, traer. Devuelve "" o un mensaje de error.
// Pensada para correr DENTRO de una tarea del GpuWorker.
std::string openOrdersGpu(const std::vector<std::int32_t>& change, OpenOrdersResult& result);

// Convierte la clave del pico traida de la GPU al resultado (count > 0).
void applyPeakKey(unsigned long long key, OpenOrdersResult& result);

struct OpenOrdersSummary {
    std::int32_t peakOpenOrders = 0;
    std::int32_t peakTimeMs = 0;  // tiempo de simulacion del evento del pico
    std::int32_t openAtEnd = 0;   // abiertos despues del ultimo evento
    bool usedGpu = false;         // false: se uso el respaldo CPU
};

// Pico y abiertos al final de los pedidos dados. Lo pide al GpuWorker si hay uno y esta sano, y
// espera como mucho `timeout` en tiempo REAL; si no hay GPU, la tarea falla o vence, usa la CPU
// (1 hilo). Nunca lanza.
OpenOrdersSummary openOrdersSummary(const std::vector<OrderLifetime>& orders, GpuWorker* worker,
                                    std::chrono::milliseconds timeout = std::chrono::milliseconds(2000));

// Linea para stderr: "[estadisticas] pedidos en curso: pico 4 en t=180.0s, abiertos al final 2 (backend=cpu)".
std::string formatOpenOrders(std::int32_t peakOpenOrders, std::int32_t peakTimeMs, std::int32_t openAtEnd,
                             const std::string& backendName);

}  // namespace accel
