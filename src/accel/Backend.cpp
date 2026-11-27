#include "accel/Backend.h"

#include <stdexcept>

namespace accel {

BackendChoice parseBackendChoice(const std::string& text, bool& ok) {
    ok = true;
    if (text == "auto") {
        return BackendChoice::automatic;
    }
    if (text == "cpu") {
        return BackendChoice::cpu;
    }
    if (text == "gpu") {
        return BackendChoice::gpu;
    }
    ok = false;
    return BackendChoice::automatic;
}

BackendSelection selectBackend(BackendChoice choice, std::function<GpuInfo()> probe) {
    BackendSelection selection;
    if (choice == BackendChoice::cpu) {
        selection.line = "Backend: CPU (--backend cpu: la GPU no se toca)";
        return selection;
    }

    // El sondeo corre en el hilo del worker: ahi nace el contexto CUDA. Se escribe `info` antes
    // de que el constructor devuelva (espera con un future), asi que leerla despues es seguro.
    auto info = std::make_shared<GpuInfo>();
    selection.worker = std::make_unique<GpuWorker>([probe = std::move(probe), info] {
        *info = probe();
        if (!info->available) {
            throw std::runtime_error(info->message);
        }
    });

    if (selection.worker->healthy()) {
        selection.line = describe(*info) + ", hilo dueno de la GPU";
        return selection;
    }
    GpuInfo unusable;
    unusable.message = selection.worker->unhealthyReason();
    selection.worker.reset();  // stop + join
    if (choice == BackendChoice::gpu) {
        selection.failed = true;
        selection.line = "Error: --backend gpu pero la GPU no esta disponible: " + unusable.message;
    } else {
        selection.line = describe(unusable);
    }
    return selection;
}

}  // namespace accel
