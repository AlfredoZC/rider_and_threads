#pragma once

#include "accel/GpuInfo.h"

namespace accel {

// Se llama UNA vez, en el hilo dueño de la GPU (GpuWorker), antes de cualquier tarea.
// Crea el contexto (calentamiento) y decide si la GPU es utilizable; nunca lanza ni aborta:
// si no lo es, `available` queda en false con el motivo en `message`. Hay dos versiones:
//   accel/cuda/GpuContext.cu     cudaSetDeviceFlags(BlockingSync), cudaSetDevice(0), cudaFree(0)
//                                y la sonda computeMode != Prohibited, cc >= 7.5
//   accel/cpu/GpuContextCpu.cpp  "compilado sin CUDA"
GpuInfo initGpuContext();

// Para los envoltorios de kernels (dentro de una tarea del worker): lanza GpuFatalError si hay
// un error CUDA pegajoso pendiente. Sin CUDA no hace nada.
void throwIfGpuError(const char* what);

}  // namespace accel
