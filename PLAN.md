# Plan de trabajo: Equipetrol Delivery (simulación concurrente)

> Documento de tutoría. Tú escribes el código; este plan dice **qué** construir,
> **en qué orden**, **qué concepto del curso** usa cada pieza y **cómo verificar**
> cada paso antes de avanzar. Cada fase termina con un *checkpoint* que debes
> pasar antes de pasar a la siguiente.

---

## 0. Las reglas duras del ingeniero (checklist permanente)

Si rompes cualquiera de estas, se pierden puntos o directamente no se revisa (DP.0).

| # | Regla | Consecuencia práctica en el código |
|---|-------|-------------------------------------|
| R1 | **C++17**, nada de C++20 | Nada de `std::jthread`, `std::stop_token`, `std::counting_semaphore`, `std::latch`, `std::barrier`. Todo eso lo construyes con `mutex` + `condition_variable`. |
| R2 | Solo **stdlib + OpenCV (solo ventana) + 1 librería JSON** | Nada de `pthread_create`, OpenMP, TBB, Boost, Qt. OpenCV **no** se usa para hilos (`cv::parallel_for_` prohibido). JSON: `nlohmann/json.hpp` **copiado dentro del repo** (`third_party/`). |
| R3 | `mkdir build && cd build && cmake .. && make` y **nada más** | Nada de `-DOPENCV_DIR=...`, ni `FetchContent` que dependa de red. El target se llama **`delivery_sim`**. |
| R4 | **Cero warnings** con `-Wall -Wextra` | Headers de terceros como `SYSTEM` includes. Compila siempre con `-Werror` mientras desarrollas. |
| R5 | Opción `-DENABLE_TSAN=ON` | Mismo target, con `-fsanitize=thread -g -O1`. |
| R6 | **Prohibido dormir para coordinar** y **prohibido spinning** | `sleep_for` solo para (a) tiempo simulado y (b) ritmo de frames. Toda espera por otro hilo = `condition_variable::wait(lock, predicado)`. |
| R7 | Idle = **~0% CPU** | Couriers sin trabajo bloqueados en un `cv.wait`, nunca en un `while` con sleep. |
| R8 | Todo hilo termina con `join()` | Nada de `detach()`. Nunca. |
| R9 | Solo locks con alcance (RAII) | `lock_guard`, `unique_lock`, `scoped_lock`. Nunca `m.lock()` / `m.unlock()` a mano. |
| R10 | Cero punteros crudos que posean memoria | `shared_ptr` / `unique_ptr`. Punteros crudos solo como "observadores" si hace falta (mejor evitarlos). |
| R11 | Log = 1 JSON válido por línea, sin mezclar; resumen == log | Un único monitor `EventLog` escribe líneas completas y actualiza los contadores **en la misma sección crítica**. |
| R12 | Parada en ≤ 5 s reales, exit 0 | Todas las esperas deben poder ser despertadas por la señal de stop. |
| R13 | Mapa de **OpenStreetMap**, Segundo→Cuarto Anillo, con atribución visible | "© OpenStreetMap contributors" en la ventana **y** en el README. |
| R14 | Mínimo 25 intersecciones, 40 segmentos, 6 restaurantes repartidos | Los couriers se mueven **solo** por segmentos; respetar `oneWay`. |
| R15 | Trabajo individual | Tú escribes el código. Yo reviso, explico y te hago preguntas. |

---

## 1. Entorno (decisión ya tomada)

Windows/MinGW **no sirve**: ThreadSanitizer no funciona ahí y el revisor usa Linux/Mac.
Trabajaremos en **WSL2 Ubuntu** (ya lo tienes, y WSLg abre ventanas: `DISPLAY=:0`).

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev valgrind clang gdb
```

Edita el código desde VS Code con la extensión *WSL* (abre la carpeta como
`\\wsl$\Ubuntu\...` o trabaja desde `/mnt/c/...`, aunque compilar desde `/mnt/c` es más lento).

---

## 2. Arquitectura propuesta

### 2.1 Hilos (cada uno con un destino deliberado: `join`)

| Hilo | Cantidad | En qué hace loop | Cómo espera (sin CPU) |
|------|----------|------------------|------------------------|
| **main / render** | 1 | Dibuja un frame (~30 fps), revisa flag de señal y fin de `durationS` | `cv::waitKey(33)` = pacing de frames (permitido) |
| **OrderGenerator** | 1 | Espera un intervalo aleatorio (tiempo simulado), crea 1..`burstMax` órdenes | `cv.wait_until(deadline, stop)` en `SimClock` |
| **Dispatcher** | 1 | Toma orden nueva del `OrderBook`, pide cotizaciones, asigna | `cv.wait_until(próximo vencimiento, hayOrden‖stop)` |
| **QuoteWorker** (pool) | `hardware_concurrency()` acotado | Toma tareas de cotización, calcula ETA, cumple la `promise` | `cv.wait(hayTarea‖stop)` |
| **Courier** | `fleet.couriers` | Máquina de estados: idle → a restaurante → recoger → entregar | idle: `cv.wait(tieneTrabajo‖stop)`; viajando: `cv.wait_until(finSegmento, stop‖avería)` |

### 2.2 Objetos compartidos (monitores: dato privado + mutex + métodos)

| Monitor | Protege | Quién lo toca |
|---------|---------|---------------|
| `EventLog` | archivo de log + contadores del resumen + `t` | todos |
| `OrderBook` | cola acotada (`maxPending`) de órdenes esperando courier | Generator (push), Dispatcher (take), Couriers (avisar que hay capacidad) |
| `Restaurant` | `pickupSlots`, cola FIFO de couriers esperando | Couriers |
| `Courier` | estado, bolsa, ruta, segmento actual (para dibujar) | su propio hilo, Dispatcher (`tryAssign`), QuoteWorkers (snapshot), Render (snapshot), otro Courier (handover) |
| `QuotePool` | cola de tareas de cotización | Dispatcher, QuoteWorkers |
| `SimClock` | inmutable tras arrancar (start + timeScale) + flag de stop | todos (solo lectura) |
| `CityGraph` | **inmutable** tras cargar → no necesita lock | todos (solo lectura) |

### 2.3 Canales entre hilos (esto va al esquema del README)

1. Generator → `OrderBook` → Dispatcher (productor/consumidor, cv).
2. Dispatcher → `QuotePool` → QuoteWorker → **`std::promise<Quote>` / `std::future<Quote>`** → Dispatcher (con `wait_until` = `quoteTimeoutMs`, excepciones cruzan con `set_exception`).
3. Dispatcher → `Courier::tryAssign(order&&)` → despierta al courier (cv del courier).
4. Courier ↔ `Restaurant` (cola FIFO con slots, cv).
5. Courier A → Courier B en avería (**`std::scoped_lock(a.m, b.m)`**, dos locks sin deadlock).
6. Courier → `OrderBook::notifyCapacity()` → Dispatcher (hay courier libre, reintenta).
7. Todos → `EventLog`.
8. Render ← snapshots de `Courier` y órdenes activas.
9. main → todos: `stop()` en cada monitor (set flag bajo lock + `notify_all`).

### 2.4 Trucos clave de diseño (entiéndelos antes de programar)

- **Movimiento suave sin gastar CPU**: el courier *no* se despierta cada frame. Publica
  "estoy en el segmento S, empecé en `t0`, termino en `t1`" y duerme hasta `t1`
  (`wait_until`, interrumpible). El **render interpola** la posición con la hora simulada
  actual. Resultado: puntos que se mueven fluido y courier con 0% CPU.
- **Tiempo simulado**: `simNow = (steady_clock::now() - start) * timeScale`. Toda espera
  "de simulación" se convierte a un `time_point` real y se hace con
  `cv.wait_until(lock, realDeadline, [&]{ return stop || evento; })`. Así el stop
  despierta a cualquiera al instante (regla R12).
- **`std::async` y el destructor que bloquea**: el `future` de `std::async` bloquea en su
  destructor. Si lo usas para cotizaciones y una llega tarde, el dispatcher se congela.
  Por eso las cotizaciones van a un **pool propio con `promise`** (la tardía llena un
  `future` que nadie lee, y nadie se bloquea). `std::async(std::launch::async, ...)` lo
  usamos donde sí queremos esperar el resultado: **precálculo de caminos mínimos al
  arrancar**, una tarea por restaurante, en paralelo.
- **Log == resumen**: `EventLog::emit(evento)` toma el lock, calcula `t`, escribe la línea
  completa y actualiza contadores. Todo junto → imposible que discrepen, y `t` es
  monótono en el archivo.
- **FIFO en restaurantes sin semáforos**: `deque<courierId>` + `inUse`; un courier entra
  cuando es el `front()` y `inUse < slots`. `pickupStarted` se loguea **dentro** del lock
  del restaurante para que el orden del log refleje el orden real.
- **Señales**: el handler solo hace `g_signal = 1;` (`volatile std::sig_atomic_t`).
  El render loop lo mira cada frame (ya existe ese loop, no es spinning).
- **OpenCV en el hilo main**: `imshow`/`waitKey` siempre desde main.

### 2.5 Estructura de carpetas

```
CMakeLists.txt
README.md
config/equipetrol.json
config/tests/        (small, burst, slots1, bag1, breakdown1, unreachable, invalid_*.json)
data/equipetrol.png
third_party/nlohmann/json.hpp
src/  main.cpp  Config.*  CityGraph.*  SimClock.h  Order.h  EventLog.*
      OrderBook.*  Restaurant.*  Courier.*  QuotePool.*  Dispatcher.*
      OrderGenerator.*  Renderer.*
tools/ validate_log.py   osm_extract.py (opcional)
experiments/  (configs, logs, summaries crudos)
analysis/     (salidas completas de TSAN, ASan, Valgrind)
```

---

## 3. Fases (en este orden; cada una con checkpoint)

> Orden pensado para que **siempre** tengas algo que compila y corre, y para hacer
> el apagado (DP.9) temprano: si lo dejas al final, todo lo demás se rompe al añadirlo.

### Fase 0 — Esqueleto y build (DP.0)
- `CMakeLists.txt`: C++17, `-Wall -Wextra`, `find_package(OpenCV REQUIRED)`, `Threads`,
  `option(ENABLE_TSAN ...)`, `json.hpp` como include `SYSTEM`.
- `main.cpp` que abre una ventana negra y cierra con ESC.
- ✅ **Checkpoint**: `mkdir build && cd build && cmake .. && make` → cero warnings;
  lo mismo con `-DENABLE_TSAN=ON`; la ventana abre en WSLg.

### Fase 1 — Mapa real y configuración (DP.1)
- Exportar PNG desde openstreetmap.org (Share/Export → Image, límites personalizados)
  cubriendo 2do–4to anillo con Av. San Martín visible. Anotar N/S/W/E.
- Tomar ≥ 30 intersecciones reales (Overpass Turbo o clic manual), ≥ 45 segmentos,
  ≥ 6 restaurantes repartidos. Marcar `oneWay` donde corresponda.
- `Config` + validación: argumento faltante, archivo ilegible, JSON roto, propiedad
  ausente, nodo inexistente → mensaje en `stderr`, `return 1`, sin simular.
- `lat/lon → pixel` lineal con los bounds.
- ✅ **Checkpoint**: dibujas nodos y segmentos encima del PNG y **caen sobre las calles**.
  Los `invalid_*.json` salen con código ≠ 0.

### Fase 2 — Piezas base (single-thread primero)
- `CityGraph`: longitudes con haversine, Dijkstra respetando `oneWay`, detección de inalcanzable.
- `SimClock`, `Order` (**move-only**: copia `= delete`, move `= default`), `EventLog`.
- ✅ **Checkpoint**: pequeño test que imprime una ruta; `EventLog` escribe líneas JSON válidas.

### Fase 3 — Couriers que se mueven + apagado (DP.3, DP.8, DP.9)
- Hilos `Courier` que, sin órdenes aún, recorren rutas aleatorias por el grafo.
- `Renderer` con interpolación, colores por estado, atribución, panel de contadores.
- Señales + `durationS` + secuencia de stop + `join` de todos + resumen.
- ✅ **Checkpoint**: puntos moviéndose suave por las calles; Ctrl-C → sale en < 5 s con 0;
  `Simulation has started...` aparece. TSAN limpio.

### Fase 4 — Órdenes y OrderBook (DP.2)
- `OrderGenerator` con `std::mt19937(seed)`, intervalos exponenciales, ráfagas.
- `OrderBook` acotado; `queueFull` y `unreachable`.
- ✅ **Checkpoint**: misma seed → mismas órdenes creadas; nunca más de `maxPending` esperando.

### Fase 5 — Despacho simple + restaurantes (DP.5)
- Dispatcher **secuencial** primero (cotiza uno tras otro). *Guárdalo: es el "antes" del experimento.*
- `Courier::tryAssign`, ciclo recoger/entregar, `Restaurant` FIFO con slots.
- `noCourier` con `acceptTimeoutMs`.
- ✅ **Checkpoint**: órdenes entregadas de punta a punta; `validate_log.py` pasa.

### Fase 6 — Cotizaciones concurrentes (DP.4)
- `QuotePool` (tamaño según `hardware_concurrency()`), `promise/future`,
  `wait_until(quoteTimeoutMs)`, `set_exception` en una cotización que falla.
- ✅ **Checkpoint**: inyectas una excepción aleatoria en cotizaciones → el programa sigue;
  una cotización lenta se ignora; la ventana nunca se congela.

### Fase 7 — Averías y traspaso atómico (DP.6)
- Avería aleatoria durante un viaje; `scoped_lock` sobre los dos couriers; `orderReassigned`.
- Caso sin receptor disponible (todos averiados/llenos) resuelto sin bloquear.
- ✅ **Checkpoint**: `breakdownProbability: 1` corre y termina; nunca una orden en dos bolsas.

### Fase 8 — Validador y escenarios de prueba (DP.7)
- `tools/validate_log.py`: todas las reglas de las secciones 4, 6 y 7.
- Configs de la sección 11 (pequeña, grande 64 couriers, ráfagas, slots=1, bag=1, p=1,
  inalcanzables, inválidas). Script que corre cada una N veces.
- ✅ **Checkpoint**: 20 corridas por escenario, todas válidas.

### Fase 9 — Herramientas (DP.11)
- TSAN (build con `ENABLE_TSAN`), ASan+LSan (o Valgrind Memcheck), opcional Helgrind.
- Guardar salidas **completas** en `analysis/`. Documentar versión, comando, hallazgo, fix, re-run.
- ✅ **Checkpoint**: cero reportes propios; lo que venga de OpenCV/GTK, listado y explicado.

### Fase 10 — Experimentos (DP.10)
Ideas (una variable a la vez, todo lo demás igual, misma seed):
- Cotización secuencial vs concurrente → latencia de asignación, avg delivery. **(cambio de diseño con antes/después)**
- Tamaño del pool: 1, 2, 4, `hardware_concurrency()`.
- Nº de couriers vs tiempo medio de entrega y rechazos.
- `pickupSlots` 1 vs 3 → tiempo en cola del restaurante.
- ✅ **Checkpoint**: tabla comparativa + datos crudos en `experiments/`.

### Fase 11 — README final (DP.12)
- Build/run, dependencias con versiones, atribución OSM.
- Esquema de hilos y canales (sección 2.3 de este plan, actualizado a tu código real).
- Mapa de conceptos (sección 4 de este plan, con archivo:línea reales).
- Argumento escrito de "por qué no hay data race" por cada objeto compartido.

---

## 4. Borrador del mapa de conceptos (dónde vivirá cada concepto)

| Carpeta | Concepto | Dónde lo usaremos |
|---------|----------|-------------------|
| 01 | Creación explícita y destino deliberado de cada hilo | `main.cpp`: crea y hace `join` de todos los hilos en la secuencia de stop |
| 01 | Decisión según paralelismo del hardware | Tamaño de `QuotePool` = `clamp(hardware_concurrency()-1, 1, N)` |
| 02 | Datos al hilo por valor / referencia / move | id del courier por valor; `CityGraph` inmutable por `std::cref`; `Order` por move |
| 02 | Órdenes viajando por move, en un solo lugar | `Order` move-only: Generator → OrderBook → Courier → (handover) → otro Courier |
| 03 | Funciones miembro en hilos; contenedores de hilos | `std::thread(&Courier::run, courierPtr)`; `vector<std::thread>` de couriers y workers |
| 03 | Propiedad compartida, sin punteros crudos dueños | `shared_ptr<Courier>`, `shared_ptr<Restaurant>`, `shared_ptr<OrderBook>`, `shared_ptr<EventLog>` |
| 04 | Tarea que devuelve resultado por future con launch policy | `std::async(std::launch::async, ...)`: Dijkstra desde cada restaurante al arrancar |
| 04 | promise/future como canal de un disparo | Cada cotización: el worker cumple una `promise<Quote>` |
| 04 | Espera con límite y excepción entre hilos | `future.wait_until(quoteDeadline)`; `promise.set_exception` → el dispatcher la atrapa en `get()` |
| 05 | Argumento escrito de ausencia de races por objeto | Sección del README, un párrafo por monitor |
| 05 | Tipo seguro de copiar/mover por sus miembros especiales | `Order` (move-only, regla de cinco) y `CourierSnapshot` (copiable, solo valores) |
| 06 | Exclusión mutua solo con locks de alcance, secciones cortas | todos los monitores; `unique_lock` liberado antes de dibujar/calcular |
| 06 | Una operación con dos locks a la vez y por qué no hay deadlock | Handover: `std::scoped_lock lk(a->m_, b->m_)` (algoritmo de evasión de deadlock) |
| 07 | Monitores | `OrderBook`, `Restaurant`, `Courier`, `EventLog`, `QuotePool` |
| 07 | Espera por condición, robusta a despertares espurios y al stop | todos los `cv.wait(lk, [&]{ return stop_ || ...; })` |

---

## 5. Cómo vamos a trabajar (tutoría)

1. Por cada fase te explico el concepto y te doy la **interfaz** (firmas, responsabilidades),
   no la implementación.
2. Tú escribes el código y me dices "listo, revisa".
3. Yo reviso buscando: races, esperas incorrectas, violaciones de R1–R15, warnings.
   Te señalo el problema y **por qué**, tú lo corriges.
4. Corremos el checkpoint juntos (build, TSAN, validador).
5. Te hago 2–3 preguntas tipo defensa oral sobre lo que escribiste (el ingeniero te las hará).
