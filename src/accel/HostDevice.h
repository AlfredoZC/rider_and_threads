#pragma once

// Funciones que compilan igual para la CPU y para la GPU: bajo nvcc llevan __host__ __device__.
#ifdef __CUDACC__
#define ACCEL_HOST_DEVICE __host__ __device__
#else
#define ACCEL_HOST_DEVICE
#endif
