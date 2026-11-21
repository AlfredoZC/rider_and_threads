#pragma once

#include <cstddef>
#include <string>

namespace accel {

// Envoltorios de CUDA para las aplicaciones de calibracion del modo bench. Este encabezado no
// incluye nada de CUDA: lo implementan accel/cuda/Calib.cu (con -DUSE_CUDA=ON) y
// accel/cpu/CalibCpu.cpp (sin CUDA, donde todo devuelve "compilado sin CUDA"). Los .cu solo
// tienen kernels y envoltorios; el estado y la concurrencia del host viven en los .cpp.
// Todas las funciones devuelven "" si salieron bien o un mensaje de error; ninguna lanza ni aborta.

// Memoria del dispositivo con dos cudaEvent propios (creados en allocate, fuera de la region medida).
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Reserva `bytes` (libera antes lo que hubiera). Sin memoria: devuelve un mensaje claro.
    std::string allocate(std::size_t bytes);
    void release();
    std::size_t bytes() const { return bytes_; }

    // Copias sincronas; `ms` recibe el tiempo medido con cudaEvent. bytes <= bytes().
    std::string upload(const void* host, std::size_t bytes, double* elapsedMs);
    std::string download(void* host, std::size_t bytes, double* elapsedMs);
    // Copia dispositivo -> dispositivo con un kernel (grid-stride, 16 B por hilo); mide con cudaEvent.
    std::string copyTo(DeviceBuffer& destination, double* elapsedMs);
    // Pone todos los bytes en cero (sin medir).
    std::string clear();

private:
    friend std::string launchCounterKernel(DeviceBuffer&, std::size_t, bool, double*);
    void* devicePointer_ = nullptr;
    std::size_t bytes_ = 0;
    void* startEvent_ = nullptr;
    void* stopEvent_ = nullptr;
};

std::string deviceSynchronize();

// Memoria libre y total del dispositivo (para acotar los tamanos del barrido).
std::string gpuMemory(std::size_t* freeBytes, std::size_t* totalBytes);

// Lanza `count` veces un kernel de un hilo que suma 1 al contador de 8 bytes de `counter`. Con
// syncEach espera (cudaDeviceSynchronize) despues de cada lanzamiento; sin el, solo al final.
// `eventMs` recibe el tiempo total entre dos cudaEvent. Despues de cada lanzamiento se revisa
// cudaGetLastError. Los lanzamientos van en el mismo stream, asi que el contador nunca compite.
std::string launchCounterKernel(DeviceBuffer& counter, std::size_t count, bool syncEach, double* eventMs);

}  // namespace accel
