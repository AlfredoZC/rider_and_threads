# ADR 0001: compilación opcional con CUDA

- **Estado:** aceptada (2026-11-20)
- **Contexto:** el enunciado de la segunda evaluación (`GPU additions and the benchmarking report`,
  "Ground rules") exige: *"It must still build without CUDA. Put the GPU code behind a CMake option
  such as `-DUSE_CUDA=ON`, off by default. Without it, `cmake .. && make` builds the CPU version"*.
  Cada feature GPU necesita además una versión CPU equivalente.

## Decisión

- Opción de CMake `USE_CUDA`, **apagada por defecto**. Sin ella el build es el de siempre (C++17 +
  OpenCV) y `cmake .. && make` produce la versión CPU.
- El código acelerado vive en `src/accel/`:
  - `src/accel/*.cpp`: código común a los dos builds.
  - `src/accel/cuda/*.cu`: implementación GPU, solo con `USE_CUDA=ON`.
  - `src/accel/cpu/*.cpp`: implementación de respaldo, solo sin CUDA.
- Con `USE_CUDA=ON`: CMake >= 3.18, CUDA Toolkit >= 11.8, arquitecturas `75;89` por defecto
  (T4 de Colab y RTX 4050 de la laptop de desarrollo), `-lineinfo` para Nsight Compute.
- `USE_CUDA` no se combina con TSan ni ASan: los sanitizers se corren sobre el build CPU. Por eso los
  `.cu` contienen solo kernels y sus envoltorios; la concurrencia y el estado compartido del host
  van en `.cpp`, donde TSan los ve.
- Al arrancar, el programa escribe en stderr una línea con el backend detectado (nunca en stdout,
  que lee `tools/validate_log.py`). Sin GPU o sin driver informa "no disponible" y sigue en CPU, sin
  abortar.

## Alternativas descartadas

- **Compilar los mismos `.cu` como C++ con `ec2/common/cuda_compat.h` en el build normal.** Una sola
  fuente por kernel, pero la emulación crea un `std::thread` por hilo CUDA y su barrera espera
  girando: viola "nobody spins" y no escala al tamaño de la ciudad. El enunciado además prohíbe
  cronometrarla. Queda solo como herramienta de prueba de correctitud.
- **Archivos `.cu` y `.cpp` mezclados en la misma carpeta con `#ifdef USE_CUDA`.** Menos carpetas,
  pero cada archivo tendría dos implementaciones entrelazadas y los tests sin CUDA podrían compilar
  código host de CUDA por error.
