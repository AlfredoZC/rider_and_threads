# ADR 0004: hilo dueño de la GPU

- **Estado:** aceptada (2026-11-20)
- **Contexto:** el enunciado de la segunda evaluación pide: *"One thread owns the GPU. Other threads
  submit work to it through a queue and receive the results through futures, so couriers keep riding
  and the window keeps refreshing while a kernel runs. quoteTimeoutMs still applies to estimates
  computed on the GPU."* El brief limita la concurrencia de CPU a la biblioteca estándar de C++17 y
  exige que nadie gire ni duerma para esperar a otro hilo, que todo hilo se una y que no haya
  punteros crudos dueños.

## Decisión

- `accel::GpuWorker` (`src/accel/GpuWorker.{h,cpp}`, solo C++ estándar): un `std::thread` con una
  cola FIFO (`mutex` + `condition_variable`). `submit(fn)` devuelve un `std::future<R>`; la espera del
  hilo es bloqueante. Los `.cu` no tienen concurrencia (ADR 0001): solo kernels y envoltorios.
- La inicialización corre **en el propio hilo** (`accel::initGpuContext`, `.cu` con respaldo en
  `src/accel/cpu/`): `cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)` (la sincronización duerme el
  hilo en vez de girar), `cudaSetDevice(0)`, `cudaFree(0)` (crea el contexto: calentamiento) y una
  sonda cacheada (`computeMode != Prohibited`, cc >= 7.5). El constructor espera el resultado con un
  `future`.
- **Salud:** si la inicialización falla, o una tarea lanza `GpuFatalError` (error CUDA pegajoso;
  `throwIfGpuError` lo traduce), el worker queda "no saludable" para siempre: las tareas en cola se
  rechazan y los `submit` siguientes lanzan `GpuUnavailableError` al instante, y el llamador usa la
  CPU. Otras excepciones van solo al future de su tarea. Sin `exit`/`abort`.
- **Apagado:** `stop()` rechaza envíos nuevos, deja terminar la tarea en curso, resuelve con
  excepción todo lo encolado (nunca un future sin resolver) y une el hilo; es idempotente y el
  destructor la llama. `main` lo apaga después de los hilos que lo usan (dispatcher, fleet).
- **Plazo:** `quoteTimeoutMs` es tiempo **real** ("longest REAL time the dispatcher waits"), no
  simulado (un `SimClock` a escala 60 lo habría vencido 60 veces antes). `accel::deadlineAfterMs`
  calcula UNA vez por ronda un plazo absoluto de `steady_clock` (`now + clamp(ms, 0, 24 h)`, el patrón
  del Dispatcher) y `accel::waitUntilDeadline(future, deadline)` espera hasta él; los candidatos
  comparten el plazo (nunca N × timeout). Si es tarde, el llamador usa la CPU y descarta el future (no
  bloquea). `submit(fn, deadline)` admite además un plazo por tarea: si al sacarla de la cola ya
  venció, no se ejecuta y su future lanza `GpuTaskExpiredError`, para que la cola no acumule trabajo
  que nadie espera. Contrato para quien envía (documentado en `GpuWorker.h`): capturar por valor o
  `shared_ptr` (la tarea sobrevive al future descartado), no esperar futures del mismo worker y no
  llamar `stop()`/destruir el worker desde una tarea.
- **Alcance de esta entrega:** es la infraestructura (worker, `--backend`, `SimContext::gpu`, helpers
  de plazo). Ningún hilo de la simulación envía trabajo todavía; dispatcher y fleet lo harán en las
  features siguientes, con la CPU como respaldo.
- **Interruptor:** `--backend cpu|gpu|auto` (por defecto `auto`). `cpu` no crea el worker ni toca CUDA
  (permite probar el respaldo dentro del build CUDA; en WSL ocultar la GPU con
  `CUDA_VISIBLE_DEVICES=""` aborta en libcuda). `gpu` exige GPU y termina con error si no hay.
- **Acceso desde la simulación:** `SimContext::gpu`, un `accel::GpuWorker*` **no dueño** y nulo en
  CPU. Puntero y no referencia porque es opcional; lo posee `main`, que lo crea antes de los hilos y
  lo apaga después de ellos, así que nunca queda colgando.

## Alternativas descartadas

- **Un hilo (o stream) por aplicación/repartidor que llame CUDA por su cuenta.** Cada hilo con
  contexto propio multiplica memoria y creación de contexto, y las llamadas compiten sin orden; el
  enunciado pide explícitamente un único dueño.
- **Llamar CUDA directamente desde el dispatcher.** Es lo más simple, pero el dispatcher se bloquea
  durante el kernel y con él la asignación de pedidos; no se puede aplicar `quoteTimeoutMs` sin
  abandonar un kernel a medias, y un error CUDA tumbaría al dispatcher.
- **Varios streams / varios workers.** Permitiría solapar kernels, pero la carga actual (una
  estimación por ronda) no lo justifica y exige sincronizar memoria de dispositivo compartida. Se
  puede añadir después sin cambiar `submit`.
- **Esperar con sondeo (`future.wait_for` en bucle, `cudaStreamQuery`, sleeps).** Viola "nobody
  spins"; se usa `BlockingSync` y esperas bloqueantes.
