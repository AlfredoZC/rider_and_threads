#pragma once

#include <stdexcept>
#include <string>

namespace accel {

// La GPU no se puede usar (worker detenido o "no saludable"). Quien llama usa el respaldo CPU.
class GpuUnavailableError : public std::runtime_error {
public:
    explicit GpuUnavailableError(const std::string& what) : std::runtime_error(what) {}
};

// La tarea tenia plazo y ya habia vencido al sacarla de la cola: no se ejecuto.
class GpuTaskExpiredError : public std::runtime_error {
public:
    explicit GpuTaskExpiredError(const std::string& what) : std::runtime_error(what) {}
};

// Una tarea la lanza cuando el error CUDA no es recuperable (pegajoso, contexto perdido):
// el worker queda "no saludable" para siempre y los envios siguientes fallan rapido.
class GpuFatalError : public std::runtime_error {
public:
    explicit GpuFatalError(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace accel
