#pragma once

#include <functional>
#include <memory>
#include <string>

#include "accel/GpuContext.h"
#include "accel/GpuInfo.h"
#include "accel/GpuWorker.h"

namespace accel {

// --backend cpu|gpu|auto
enum class BackendChoice { automatic, cpu, gpu };

// Interpreta el valor de --backend; `ok` = false si no es ninguno de los tres.
BackendChoice parseBackendChoice(const std::string& text, bool& ok);

struct BackendSelection {
    std::unique_ptr<GpuWorker> worker;  // nullptr = se usa la CPU
    std::string line;                   // linea para stderr: backend elegido y por que
    bool failed = false;                // true = pidieron `gpu` y no se pudo: terminar con error
};

// Decide el backend. `cpu` no crea el worker ni toca CUDA. `auto` crea el worker y usa la GPU si
// el sondeo (que corre en el hilo del worker) la da por utilizable; si no, respaldo CPU con el
// motivo. `gpu` exige GPU: sin ella `failed`.
BackendSelection selectBackend(BackendChoice choice,
                               std::function<GpuInfo()> probe = initGpuContext);

}  // namespace accel
