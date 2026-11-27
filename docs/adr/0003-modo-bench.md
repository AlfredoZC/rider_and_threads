# ADR 0003: modo bench (`--bench`, `--bench-once`)

- **Estado:** aceptada (2026-11-20)
- **Contexto:** el enunciado de la segunda evaluación (`GPU additions and the benchmarking report`,
  secciones 2.1 a 2.4) pide un modo `--headless --bench` sin ventana que produzca, para cada
  aplicación, un CSV con `application, version, size, rep, ms, correct`, con el protocolo de medición
  de 2.2 (calentamiento, ≥ 10 repeticiones, versiones intercaladas, misma entrada y semilla, chequeo de
  corrección en cada corrida, solo Release) y `--bench-once` para `ncu` y `compute-sanitizer`.
  El brief solo permite la biblioteca estándar de C++17 para la concurrencia de CPU
  (`std::thread`, `std::async`, mutex, `condition_variable`, atómicos).

## Decisión

- **CLI.**
  `delivery_sim <config> --headless --bench <results.csv> [--apps a,b] [--reps N] [--threads 1,2,4,...] [--sizes s1,s2] [--seed S] [--force]`
  y `delivery_sim <config> --headless --bench-once --app X --size N [--version V] [--threads N] [--seed S]`.
  `--bench` no corre la simulación ni abre ventana. Los números se leen con `std::from_chars`, sin
  signos ni basura final y con límites (`--reps` 1..100000, hilos 1..1024, listas ≤ 32 elementos,
  semilla 32 bits). Los flags desconocidos o mezclados entre modos se rechazan. No se sobrescribe un
  CSV existente sin `--force` (apertura exclusiva `fopen("wx")`, sin ventana de carrera).
  `--sizes` es una adición menor a la CLI aprobada: sin ella, `tools/check_bench.sh` tardaría minutos.
- **Qué mide el CSV.** Una fila por corrida, con `flush` por fila:
  `application,version,size,rep,ms,correct,threads,seed,h2d_ms,kernel_ms,d2h_ms`. `version` es
  `cpu1`, `cpuN`, `gpu_kernel` o `gpu_e2e`; `correct` es `yes`/`no`; `threads` es 1 para `cpu1` y las
  versiones GPU y N para `cpuN`. Las columnas de tiempo que no aplican quedan vacías. El CSV solo
  recibe identificadores generados por el código (nombres de aplicación y de versión), nunca texto del
  config; aun así `csvEscape` cita comas, comillas y saltos de línea (RFC 4180).
- **Reloj de cada columna.** `ms` de `cpu1`, `cpuN` y `gpu_e2e` es `std::steady_clock` (en `gpu_e2e`
  con `cudaDeviceSynchronize()` antes de parar); `ms` de `gpu_kernel` y las columnas `h2d_ms`,
  `kernel_ms`, `d2h_ms` son `cudaEvent`. Las copias de `gpu_e2e` no incluyen `cudaMalloc` porque el
  buffer se reserva una vez por tamaño, fuera de la región medida.
- **Protocolo.** Por (aplicación, tamaño): se prepara la entrada con la semilla; una corrida de
  calentamiento de cada versión sin registrar; luego `--reps` repeticiones (por defecto 10) con el
  orden de versiones **rotado a la izquierda una posición por repetición**, de modo que cada versión
  ocupa cada posición y el calor acumulado no favorece siempre a la misma. `cpuN` aparece una vez por
  cada valor del barrido de hilos (por defecto 1, 2, 4, 6, 8, 12, 16 recortado a
  `hardware_concurrency`). En cada corrida el resultado se compara con la referencia (la salida de
  `cpu1`) y se escribe `correct`; si alguna fila da `no`, o una corrida no se pudo ejecutar, el
  programa termina con código ≠ 0 **después** de escribir el CSV.
- **Solo se mide un build Release.** `--bench` se niega (código 2, "Never time a Debug or sanitizer
  build") si `CMAKE_BUILD_TYPE` no es `Release` o si hay TSan/ASan; el build por defecto del proyecto
  (`RelWithDebInfo`) tampoco se mide. CMake genera `BuildInfoGenerated.h` con el tipo, los sanitizers,
  los flags y el commit; los sanitizers también se detectan con `__SANITIZE_*__`. `--bench-once` corre
  en cualquier build (es para perfilar) pero avisa que sus tiempos no valen.
- **`<csv>.env.json`.** Modelo de CPU (`/proc/cpuinfo`), `hardware_concurrency`, memoria, SO, GPU
  (nombre, compute capability, SMs, memoria, reloj, ancho del bus y **ancho de banda pico teórico** =
  2 × reloj de memoria × bus / 8), versiones de CUDA runtime y driver, compilador, tipo de build,
  flags, commit, semilla, repeticiones y barrido de hilos. Sin rutas ni nombres de usuario.
- **Registro de aplicaciones.** Interfaz `BenchApp` (nombre, tamaños, `prepare` con semilla, `run`
  de una versión con N hilos que devuelve tiempos y corrección, `describe` para la métrica
  derivada) y `BENCH_REGISTER_APP` en el archivo de cada aplicación: agregar una aplicación es crear
  un archivo en `src/bench/` (y, si usa la GPU, sus kernels en `src/accel/cuda/` con la versión de
  respaldo en `src/accel/cpu/`); no hay lista central que editar.
- **Pool de CPU propio** (`src/accel/ThreadPool.*`): `std::thread` + `mutex` + `condition_variable`,
  los hilos esperan bloqueados (sin giro ni sleep), el hilo que llama ejecuta el trozo 0, las
  excepciones del cuerpo se relanzan en quien llama y el destructor une a todos. Los pools se crean
  una vez por número de hilos, antes de la región medida.
- **Subgrafos de la ciudad real** (`src/bench/CitySubgraph.*`): `extractRegion(config, nodo, n)` hace
  BFS determinista desde un nodo de Equipetrol (el `startNode` del config), tratando las calles como
  no dirigidas para crecer y visitando vecinos en el orden de `config.nodes`, hasta reunir exactamente
  `n` nodos (25, 100, 400, 1000, 2000, 3741 por defecto). Conserva, en su orden y sentido, todas las
  calles con ambos extremos dentro. Las regiones son anidadas. Si `n` supera lo alcanzable, error. Las
  aplicaciones de grafo del segundo parcial lo usarán; aquí solo se implementa y se prueba.
- **Contenido inicial: calibración de overheads** (no suma puntos del §1; alimenta §2.5 y el
  roofline). `calib_copy`: `cpu1` = `memcpy`, `cpuN` = `memcpy` particionado, `gpu_e2e` = ida y vuelta
  H2D + D2H con memoria paginable (`h2d_ms`, `d2h_ms` con `cudaEvent`) y `gpu_kernel` = copia
  dispositivo→dispositivo con un kernel (ancho de banda de la VRAM para el roofline); 6 tamaños de 4 KB
  a 256 MB, acotados por la memoria libre de la GPU; en cada corrida el destino se compara byte a byte
  con el origen. `calib_launch`: N ∈ {1, 10, 100, 1000, 10000} lanzamientos de un kernel de un hilo que
  suma 1 a un contador (`gpu_kernel`: seguidos y una sola espera; `gpu_e2e`: cada uno seguido de
  `cudaDeviceSynchronize`); la referencia de CPU es N llamadas a una función trivial (`cpu1`) y N
  despachos del pool (`cpuN`, la latencia de despertar hilos). Correcto = el contador vale N.
- **Sin CUDA** las versiones GPU se omiten con un aviso y el bench corre las de CPU.
- **Seguridad de la implementación.** Tamaños e índices en `size_t` y los repartos del pool sin
  multiplicaciones que desborden; cada llamada CUDA y cada lanzamiento (`cudaGetLastError`) se revisa y
  la falta de memoria devuelve un mensaje, sin abortar; sin `std::exit`/`abort` en `src/`; ningún sleep
  ni espera activa; todo hilo se une; los `.cu` solo contienen kernels y envoltorios (el envoltorio
  `DeviceBuffer` guarda el puntero del dispositivo y dos eventos, estado de un solo hilo: el que es
  dueño de la GPU).

## Alternativas descartadas

- **Rejilla sintética como grafo de prueba.** Escala fácil, pero sin calles de un sentido ni la
  irregularidad de la ciudad real; los subgrafos de OSM miden lo que el simulador realmente usa y las
  regiones anidadas hacen comparables los tamaños.
- **Un esqueleto vacío, sin aplicaciones.** Habría dejado el modo bench sin nada que ejecutar ni
  probar de punta a punta (CSV, rotación, corrección, `ncu`); la calibración es la aplicación más
  chica que ejercita todo el camino y además da los techos de copia y de latencia que necesita el
  informe.
- **Lanzar un `std::thread` por llamada en `cpuN`** (o `std::async` por trozo). Cuesta ~20-50 µs por
  llamada y contamina justo los tamaños chicos que se quieren medir; el pool reutilizable mide el
  cómputo y su despacho por separado.
- **Un pool que gira o duerme esperando trabajo.** Prohibido por el brief ("nobody spins or sleeps to
  wait for another thread") y distorsiona los tiempos.
- **OpenMP, TBB o los hilos de la plataforma para `cpuN`.** El brief solo permite la biblioteca estándar.
- **Un binario de bench separado de `delivery_sim`.** El enunciado pide `./delivery_sim city.json
  --headless --bench results.csv`; además comparte el mismo cargador de config y de ciudad.
- **Medir cualquier build y solo avisar.** El enunciado dice "Never time a Debug or sanitizer build";
  un aviso se ignora y un CSV con tiempos de Debug llega al informe, así que el bench se niega.
- **CSV con una columna por versión (formato ancho).** El enunciado fija una fila por corrida; con una
  fila por corrida el CSV agrupa con cualquier herramienta y conserva todas las repeticiones.
- **Sobrescribir el CSV sin avisar.** Se perderían corridas costosas de la máquina con GPU.
- **`cudaMalloc` dentro de la región medida de `gpu_e2e`.** Su costo (milisegundos, muy variable)
  taparía el de las copias que se quiere calibrar; el enunciado lo pide solo si ocurre en cada llamada
  de la aplicación real, y esa decisión es de cada aplicación, no de la calibración.
- **Memoria fijada (pinned) en `calib_copy`.** Da más ancho de banda H2D/D2H, pero las aplicaciones
  del proyecto usan memoria paginable de `std::vector`; se calibra lo que se usará. Queda como mejora si
  una aplicación pasa a memoria fijada.

## Consecuencias

- `cmake .. && make` (build por defecto `RelWithDebInfo`) compila el modo bench pero no lo deja medir:
  hay que configurar `-DCMAKE_BUILD_TYPE=Release`.
- `tests/run.sh` compila ahora también `src/bench/*.cpp`; `tools/check_bench.sh` valida el CSV y el
  `.env.json` de un build Release.
