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

// Linea de tiempo global de entregas: la lista de cada courier ya esta ordenada por hora de entrega, asi
// que en vez de reordenar todo se FUSIONAN las listas de a pares, en un arbol de ceil(log2 L) rondas. Para
// fusionar en paralelo se reparte la SALIDA (no la entrada): cada hilo produce un tramo de posiciones y
// calcula con co-rank (busqueda binaria) cuantos elementos salen de cada lista. La fusion es estable (en
// un empate va primero el de la lista de la izquierda). Con la linea de tiempo ordenada, las entregas de
// cada franja (hora simulada) salen de dos busquedas binarias, sin atomicos.
//
// Todas las listas viven una detras de otra en un solo arreglo; la lista k ocupa
// [listStart[k], listStart[k+1]). Una lista puede estar vacia (courier sin entregas).

struct DeliveryRecord {
    std::int32_t deliveredAtMs = 0;  // la clave por la que se ordena
    std::int32_t orderId = 0;
};

constexpr std::int32_t kMillisecondsPerHour = 3600 * 1000;

ACCEL_HOST_DEVICE inline int minInt(int left, int right) { return left < right ? left : right; }
ACCEL_HOST_DEVICE inline int maxInt(int left, int right) { return left > right ? left : right; }

// Fusion secuencial estable: si empatan, primero el de `first` (por eso "<=").
ACCEL_HOST_DEVICE inline void mergeSequential(const DeliveryRecord* first, int firstCount, const DeliveryRecord* second,
                                              int secondCount, DeliveryRecord* merged) {
    int firstPosition = 0;
    int secondPosition = 0;
    int mergedPosition = 0;
    while (firstPosition < firstCount && secondPosition < secondCount) {
        if (first[firstPosition].deliveredAtMs <= second[secondPosition].deliveredAtMs) {
            merged[mergedPosition++] = first[firstPosition++];
        } else {
            merged[mergedPosition++] = second[secondPosition++];
        }
    }
    while (firstPosition < firstCount) merged[mergedPosition++] = first[firstPosition++];
    while (secondPosition < secondCount) merged[mergedPosition++] = second[secondPosition++];
}

// Co-rank: de las primeras `outputCount` posiciones de la fusion, cuantas vienen de `first` (el resto viene
// de `second`). Busqueda binaria con dos condiciones: first[i-1] <= second[j] y second[j-1] < first[i]
// (estricta: en un empate gana `first`).
ACCEL_HOST_DEVICE inline int coRank(int outputCount, const DeliveryRecord* first, int firstCount,
                                    const DeliveryRecord* second, int secondCount) {
    int takenFromFirst = minInt(outputCount, firstCount);  // empezar tomando todo lo posible de first
    int takenFromSecond = outputCount - takenFromFirst;
    int firstLowerBound = maxInt(0, outputCount - secondCount);  // first nunca aporta menos que esto
    int secondLowerBound = maxInt(0, outputCount - firstCount);
    for (;;) {
        const bool tooManyFromFirst = takenFromFirst > 0 && takenFromSecond < secondCount &&
                                      first[takenFromFirst - 1].deliveredAtMs > second[takenFromSecond].deliveredAtMs;
        const bool tooManyFromSecond = takenFromSecond > 0 && takenFromFirst < firstCount &&
                                       second[takenFromSecond - 1].deliveredAtMs >= first[takenFromFirst].deliveredAtMs;
        if (tooManyFromFirst) {  // devolver la mitad del margen a second
            const int step = (takenFromFirst - firstLowerBound + 1) / 2;
            secondLowerBound = takenFromSecond;
            takenFromFirst -= step;
            takenFromSecond += step;
        } else if (tooManyFromSecond) {  // ">=": en un empate gana first
            const int step = (takenFromSecond - secondLowerBound + 1) / 2;
            firstLowerBound = takenFromFirst;
            takenFromFirst += step;
            takenFromSecond -= step;
        } else {
            return takenFromFirst;
        }
    }
}

// El par que contiene una posicion de la salida: el PRIMERO cuyo final esta mas alla de ella (asi se saltean
// los pares vacios de couriers sin entregas; buscar "el ultimo par que empieza antes" no terminaria con ellos).
ACCEL_HOST_DEVICE inline int pairContaining(int position, const int* listStart, int listCount) {
    int low = 0;
    int high = (listCount + 1) / 2 - 1;
    while (low < high) {
        const int middle = (low + high) / 2;
        if (listStart[minInt(2 * middle + 2, listCount)] > position) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low;
}

// Produce las posiciones [outputBegin, outputEnd) de la salida de una ronda. Si el tramo cruza de un par al
// siguiente, lo hace en pedazos (uno por par). Con un numero impar de listas la ultima se copia tal cual.
ACCEL_HOST_DEVICE inline void mergeRoundRange(const DeliveryRecord* roundInput, DeliveryRecord* roundOutput,
                                              const int* listStart, int listCount, int outputBegin, int outputEnd) {
    int position = outputBegin;
    while (position < outputEnd) {
        const int pair = pairContaining(position, listStart, listCount);
        const int pairBegin = listStart[2 * pair];
        const int secondBegin = listStart[minInt(2 * pair + 1, listCount)];
        const int pairEnd = listStart[minInt(2 * pair + 2, listCount)];
        const int pieceEnd = minInt(outputEnd, pairEnd);
        const DeliveryRecord* first = roundInput + pairBegin;
        const DeliveryRecord* second = roundInput + secondBegin;
        const int firstCount = secondBegin - pairBegin;
        const int secondCount = pairEnd - secondBegin;
        // co-rank en los dos extremos del pedazo: desde donde y hasta donde leer en cada lista
        const int firstFrom = coRank(position - pairBegin, first, firstCount, second, secondCount);
        const int firstTo = coRank(pieceEnd - pairBegin, first, firstCount, second, secondCount);
        const int secondFrom = (position - pairBegin) - firstFrom;
        const int secondTo = (pieceEnd - pairBegin) - firstTo;
        mergeSequential(first + firstFrom, firstTo - firstFrom, second + secondFrom, secondTo - secondFrom,
                        roundOutput + position);
        position = pieceEnd;
    }
}

// Primera posicion de la linea de tiempo con deliveredAtMs >= timeMs (busqueda binaria).
ACCEL_HOST_DEVICE inline int firstAtOrAfter(const DeliveryRecord* timeline, int count, long long timeMs) {
    int low = 0;
    int high = count;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (timeline[middle].deliveredAtMs < timeMs) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

// Comienzos de lista de la ronda siguiente: cada par pasa a ser una sola lista.
std::vector<int> listStartAfterRound(const std::vector<int>& listStart);

// CPU, 1 hilo: arbol de fusiones de a pares, cada una secuencial.
std::vector<DeliveryRecord> mergeCourierListsCpu1(std::vector<DeliveryRecord> records, std::vector<int> listStart);

// CPU, N hilos: en cada ronda se reparte la SALIDA en franjas iguales (ThreadPool::parallelFor) y cada hilo usa
// co-rank para saber que leer; aunque en la ultima ronda quede un solo par trabajan todos. La vuelta de
// parallelFor es la barrera entre rondas.
std::vector<DeliveryRecord> mergeCourierListsCpuN(std::vector<DeliveryRecord> records, std::vector<int> listStart,
                                                  ThreadPool& pool);

// Entregas en cada franja de `bucketMs` (una hora simulada en el programa), contando en una pasada (la
// referencia) y con dos busquedas binarias por franja (lo que hace la GPU). Requiere la linea ordenada
// y tiempos >= 0; vacia si no hay entregas.
std::vector<int> deliveriesPerBucketCpu(const std::vector<DeliveryRecord>& timeline, int bucketMs);
std::vector<int> deliveriesPerBucketBySearch(const std::vector<DeliveryRecord>& timeline, int bucketMs);

// Lo que la GPU necesita saber de antemano (se calcula en la CPU, son pocos enteros): los comienzos de lista de
// TODAS las rondas uno detras de otro, cuantas listas tiene cada ronda y cuantas franjas hay.
struct TimelinePlan {
    std::vector<int> allListStarts;
    std::vector<int> listCountPerRound;
    int bucketMs = kMillisecondsPerHour;
    int bucketCount = 0;
};
TimelinePlan makeTimelinePlan(const std::vector<DeliveryRecord>& records, const std::vector<int>& listStart, int bucketMs);

struct DeliveryTimelineResult {
    std::vector<DeliveryRecord> timeline;  // todas las entregas, ordenadas por hora de entrega
    std::vector<int> deliveriesPerBucket;  // entregas en cada franja
};

// Memoria del dispositivo (tres buffers de registros + comienzos de lista + cuentas + dos cudaEvent). Este
// encabezado no incluye nada de CUDA: lo implementan accel/cuda/DeliveryTimeline.cu y, sin CUDA,
// accel/cpu/DeliveryTimelineCpu.cpp (todo devuelve "compilado sin CUDA"). Cada funcion devuelve "" si
// salio bien o un mensaje de error; ninguna lanza ni aborta. `elapsedMs` (opcional) recibe el tiempo medido
// con cudaEvent.
class DeviceTimeline {
public:
    DeviceTimeline() = default;
    ~DeviceTimeline();
    DeviceTimeline(const DeviceTimeline&) = delete;
    DeviceTimeline& operator=(const DeviceTimeline&) = delete;

    // Reserva para `recordCapacity` registros y para el plan dado (libera antes lo que hubiera).
    std::string allocate(std::size_t recordCapacity, const TimelinePlan& plan);
    void release();

    // Copia los registros (count <= recordCapacity) y el plan (el mismo con que se reservo, o uno que quepa).
    std::string upload(const DeliveryRecord* hostRecords, std::size_t count, const TimelinePlan& plan, double* elapsedMs);
    // Un lanzamiento por ronda (el siguiente espera al anterior) y un hilo por franja para las cuentas.
    std::string run(double* elapsedMs);
    // Trae la linea de tiempo (count registros) y las cuentas por franja (plan.bucketCount enteros).
    std::string download(DeliveryRecord* hostTimeline, int* hostPerBucket, double* elapsedMs);

private:
    // [0] registros originales (no se tocan: cada corrida parte de ellos); [1] y [2] se alternan entre rondas.
    DeliveryRecord* deviceRecords_[3] = {nullptr, nullptr, nullptr};
    int* deviceListStarts_ = nullptr;
    int* devicePerBucket_ = nullptr;
    std::size_t recordCapacity_ = 0;
    std::size_t listStartCapacity_ = 0;
    std::size_t bucketCapacity_ = 0;
    std::size_t uploadedCount_ = 0;
    std::vector<int> listCountPerRound_;
    int bucketMs_ = kMillisecondsPerHour;
    int bucketCount_ = 0;
    DeliveryRecord* sortedBuffer_ = nullptr;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

// Todo el camino en GPU: reservar, subir, fusionar, contar por hora y traer. Devuelve "" o un error.
// Pensada para correr DENTRO de una tarea del GpuWorker.
std::string deliveryTimelineGpu(const std::vector<DeliveryRecord>& records, const std::vector<int>& listStart,
                                int bucketMs, DeliveryTimelineResult& result);

struct DeliveryTimelineSummary {
    std::vector<int> deliveriesPerHour;
    std::size_t total = 0;
    bool usedGpu = false;  // false: se uso el respaldo CPU
};

// Entregas por hora simulada a partir de las listas de cada courier. Lo pide al GpuWorker si hay uno y esta
// sano, y espera como mucho `timeout` en tiempo REAL; si no hay GPU, la tarea falla o vence, fusiona en la
// CPU (1 hilo). Nunca lanza.
DeliveryTimelineSummary deliveryTimelineSummary(const std::vector<DeliveryRecord>& records,
                                                const std::vector<int>& listStart, GpuWorker* worker,
                                                std::chrono::milliseconds timeout = std::chrono::milliseconds(2000));

// Linea para stderr: "[estadisticas] entregas por hora simulada: h0 4, h1 3 (backend=cpu)".
std::string formatDeliveriesPerHour(const std::vector<int>& deliveriesPerHour, const std::string& backendName);

}  // namespace accel
