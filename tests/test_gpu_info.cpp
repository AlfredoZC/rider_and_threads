#include <cassert>
#include <iostream>
#include <string>

#include "accel/GpuInfo.h"

// Build sin CUDA (el que compila tests/run.sh): el backend GPU debe reportarse
// como no disponible, con un motivo legible, y sin abortar el proceso.
int main() {
    const accel::GpuInfo info = accel::gpuInfo();

    assert(!info.available);
    assert(info.name.empty());
    assert(info.computeMajor == 0 && info.computeMinor == 0);
    assert(info.totalMemoryBytes == 0);
    assert(info.message.find("sin CUDA") != std::string::npos);

    // La linea de arranque nunca queda vacia y menciona el motivo.
    const std::string line = accel::describe(info);
    assert(!line.empty());
    assert(line.find("CPU") != std::string::npos);

    // Idempotente: llamarla de nuevo da lo mismo.
    assert(accel::gpuInfo().message == info.message);

    std::cout << "test_gpu_info OK: " << line << '\n';
    return 0;
}
