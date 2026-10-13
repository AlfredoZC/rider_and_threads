# Plan de 7 días: commits pequeños, un PR por commit

> Objetivo: terminar el proyecto en una semana **escribiéndolo tú**, con la IA como tutora, en
> commits pequeños que muestren avance progresivo. Primero **concurrencia** (días 1 a 4) y después
> **paralelismo** (días 5 y 6). El día 7 es para el informe y la defensa.

---

## 1. El ciclo de cada commit (repítelo ~30 veces)

```
1. ENTENDER  -> Le pides a la IA que te explique el concepto del commit (no el código).
                Respondes las "preguntas de control" de ese commit sin mirar.
2. RAMA      -> git switch master && git pull
                git switch -c <rama-del-commit>
3. ESCRIBIR  -> Escribes el código tú. Si te trabas, pides una PISTA, no la solución.
                Tamaño: un commit = una idea = idealmente menos de ~80 líneas.
4. PROBAR    -> Compilas (cero warnings) y haces la prueba del commit.
5. COMMIT    -> git add <archivos> && git commit -m "<mensaje>"
                git push -u origin <rama>
6. PR        -> En GitHub: Pull Request de <rama> hacia master, con la plantilla de abajo.
7. MERGE     -> Mergeas el PR y vuelves a master: git switch master && git pull
```

### Plantilla de Pull Request (cópiala en cada PR)

```markdown
## Qué cambia
<una o dos líneas>

## Por qué
<qué problema resuelve o qué requisito del brief cumple (cita la sección: §4, §6, DP.3...)>

## Concepto del curso
<carpeta 01..07 y concepto: "07 – condition_variable con predicado", etc.>

## Cómo lo probé
<comando que corriste y qué viste>

## Qué aprendí / qué mejoraría
<una línea>
```

### Reglas para usar la IA (para que realmente aprendas)
1. **Pide explicación antes que código**: "Explícame qué hace `cv.wait(lk, pred)` y por qué necesita `unique_lock`".
2. Si te da código, **tipéalo tú**, no lo pegues. Al tipearlo vas entendiendo cada línea.
3. Antes de cada commit, **explícale a la IA con tus palabras** qué hace tu código y pídele que te corrija.
4. Si no puedes responder las preguntas de control, **no hagas el commit todavía**.

### Paso 0 (hoy mismo, 15 minutos)
Tu trabajo de las fases 0 y 1 está en la rama `setup`, pero no en `master`.
1. En GitHub: PR de `setup` hacia `master`, título *"Fase 0-1: build, mapa OSM y configuración"*. Merge.
2. `git switch master && git pull`

---

## 2. Calendario

| Día | Tema | Commits | Puntos de la rúbrica que destraba |
|-----|------|---------|-----------------------------------|
| 1 | Piezas base (sin hilos todavía) | 6 | base de DP.7 |
| 2 | **Concurrencia I**: hilos de courier, ventana, apagado | 5 | DP.3, DP.8, DP.9 |
| 3 | **Concurrencia II**: productor/consumidor de órdenes | 5 | DP.2, DP.7 |
| 4 | **Concurrencia III**: restaurantes, despacho, entrega, averías | 6 | DP.5, DP.6, DP.3 |
| 5 | **Paralelismo I**: tareas, futures, pool de hilos | 5 | DP.4 |
| 6 | **Paralelismo II** + herramientas + experimentos | 5 | DP.10, DP.11, "No data races" |
| 7 | Informe y defensa | 3 | DP.12 |

**Para aprobar** necesitas ≥ 70 puntos **y** cero data races. Si vas atrasado, el orden de prioridad es:
días 2, 3, 4 (concurrencia, lo que más puntos da) > día 6 (TSAN) > día 5 > el resto.

---

## Día 1: piezas base (todavía sin hilos)

Meta del día: el grafo calcula rutas, existe el reloj simulado, la orden no se puede copiar y el log escribe
JSON. Todo se prueba desde `main` sin hilos.

### 1.1 `feat/sim-clock`: reloj simulado
- **Escribes:** `src/SimClock.h` con `nowMs()` y `toReal(simMs)` (unas 20 líneas).
- **Concepto:** tiempo simulado = tiempo real × `timeScale`; `steady_clock` vs `system_clock`.
- **Prueba:** en `main`, imprime `nowMs()` antes y después de 1 segundo; con `timeScale` 60 debe dar ~60000.
- **Preguntas de control:** ¿por qué `steady_clock`? ¿Para qué sirve `toReal()`? (Pista: `wait_until`.)
- **Commit:** `Agregar reloj de tiempo simulado`

### 1.2 `feat/city-graph`: grafo y distancias
- **Escribes:** `src/CityGraph.h/.cpp`: constructor que arma la lista de adyacencia desde la config (dos aristas
  si la calle no es `oneWay`) y la distancia haversine.
- **Prueba:** imprime cuántas aristas tiene `small.json` (4 calles, una de un sentido → 7 aristas).
- **Preguntas:** ¿qué es una lista de adyacencia? ¿Por qué una calle de doble sentido son dos aristas?
- **Commit:** `Agregar grafo de calles con distancias haversine`

### 1.3 `feat/dijkstra`: ruta más corta
- **Escribes:** `shortestPath(from, to)` con `std::priority_queue` + `std::greater<>`, y la reconstrucción con `prev`.
- **Prueba:** con `small.json`, `b→a = [b, a]` y `a→b = [a, d, c, b]` (no se sube contra el sentido).
- **Preguntas:** ¿por qué se ignora una entrada vieja de la cola? ¿Qué devuelves si no hay camino?
- **Commit:** `Agregar Dijkstra respetando calles de un sentido`

### 1.4 `feat/order`: orden move-only
- **Escribes:** `src/Order.h`: struct con copia `= delete` y move `= default noexcept`.
- **Concepto (05):** regla de cinco; "una orden existe en un solo lugar".
- **Prueba:** escribe `Order b = a;` y verifica que **no compila**; luego `Order b = std::move(a);` sí. (Borra la prueba.)
- **Preguntas:** ¿por qué el move es `noexcept`? ¿En qué estado queda `a` después del move?
- **Commit:** `Agregar Order como tipo solo-movible`

### 1.5 `feat/event-log`: el primer monitor
- **Escribes:** `src/EventLog.h/.cpp` con `std::mutex`, `lock_guard`, `orderCreated`, `orderRejected` y `stopping()`.
  La `t` se calcula **dentro** del lock.
- **Concepto (06/07):** monitor; sección crítica; por qué el log y el contador van en el mismo lock.
- **Prueba:** escribe 3 eventos y abre el archivo: 3 líneas JSON válidas.
- **Preguntas:** ¿qué pasaría si dos hilos escribieran sin el mutex? ¿Por qué `t` dentro del lock?
- **Commit:** `Agregar EventLog como monitor con mutex`

### 1.6 `feat/summary`: resumen final
- **Escribes:** `summary()` (copia bajo lock) y `toJson()`; en `main`, imprimir el resumen como última línea.
- **Prueba:** `created = delivered + rejected + pending` en la salida.
- **Commit:** `Agregar resumen final consistente con el log`

---

## Día 2: Concurrencia I (los couriers son hilos)

Meta del día: N motos se mueven a la vez por las calles, sin gastar CPU, y el programa se apaga limpio.

### 2.1 `feat/courier-thread`: un hilo por courier
- **Escribes:** `src/Courier.h/.cpp` con `run()` que, por ahora, solo espera `stop_` con `cv.wait(lk, pred)`.
  En `main`: `std::vector<std::thread>`, `emplace_back(&Courier::run, courier)` y `join()` al final.
- **Conceptos:** 01 (creación y `join`), 03 (función miembro en hilo, `vector<thread>`, `shared_ptr`), 07 (cv).
- **Prueba:** 8 hilos creados; al cerrar la ventana todos terminan y el programa sale con 0.
- **Preguntas:** ¿qué pasa si destruyes un `std::thread` sin `join`? ¿Por qué pasar el `shared_ptr` por valor?
- **Commit:** `Crear un hilo por courier con apagado por condition_variable`

### 2.2 `feat/courier-ride`: moverse tramo por tramo
- **Escribes:** `ride(path)`: por cada tramo publica `from/to/t0/t1` bajo lock y duerme con
  `cv.wait_until(toReal(t1), pred: stop_)`. Temporalmente, el courier va a destinos aleatorios.
- **Concepto:** `wait_until` interrumpible (tiempo simulado) vs `sleep_for` (prohibido para coordinar).
- **Prueba:** imprime en `stderr` cada nodo al que llega un courier.
- **Preguntas:** ¿por qué el mutex queda libre mientras el courier duerme? ¿Qué pasa con Ctrl-C a mitad de un tramo?
- **Commit:** `Mover couriers por el grafo con esperas interrumpibles`

### 2.3 `feat/render-couriers`: dibujar puntos que se mueven
- **Escribes:** `snapshot()` (copia bajo lock) en `Courier`; en `main` o `Renderer`, interpolar
  `pos = A + (B − A)·frac` y dibujar un círculo por courier.
- **Concepto (05):** snapshot copiable = sin data race al dibujar.
- **Prueba:** los puntos se mueven **suave** sobre las calles.
- **Commit:** `Dibujar couriers interpolando su posición`

### 2.4 `feat/signals-shutdown`: SIGINT y SIGTERM
- **Escribes:** `std::atomic<bool> g_stopRequested`, handler con `std::signal`, el loop que lo revisa y la
  secuencia de apagado (`stop()` a todos + `join()`).
- **Prueba:** `./delivery_sim ... & sleep 5; kill -INT $!; wait $!; echo $?` → 0, en menos de 5 s.
- **Preguntas:** ¿por qué `atomic` y no `bool`? ¿Por qué el handler no llama a `stop()` directamente?
- **Commit:** `Apagado ordenado con SIGINT/SIGTERM`

### 2.5 `feat/duration-start-line`: fin por duración y línea de inicio
- **Escribes:** terminar cuando `nowMs() >= durationS*1000`; imprimir `Simulation has started...` con `std::endl`
  **después** de crear los hilos.
- **Prueba:** con `durationS: 240` y `timeScale: 60` termina solo a los ~4 s.
- **Commit:** `Terminar por durationS e imprimir la línea de inicio`

---

## Día 3: Concurrencia II (productor/consumidor de órdenes)

Meta del día: las órdenes se crean continuamente, en ráfagas, con la seed, y el libro tiene un límite.

### 3.1 `feat/order-book`: cola acotada (monitor)
- **Escribes:** `src/OrderBook.h/.cpp` con `tryPush(Order)`, que devuelve la orden si está lleno, y `stop()`.
- **Concepto (07):** buffer acotado; devolver la orden en lugar de perderla.
- **Prueba:** con `maxPending: 3`, el cuarto `tryPush` te devuelve la orden.
- **Commit:** `Agregar OrderBook como cola acotada`

### 3.2 `feat/order-generator`: hilo productor
- **Escribes:** `src/OrderGenerator.h/.cpp` con su propio hilo, `std::mt19937(seed)`, intervalo exponencial y
  `wait_until` interrumpible.
- **Prueba:** aparecen líneas `orderCreated` en el log; misma seed → mismas órdenes.
- **Preguntas:** ¿por qué el generador de números es local al hilo? ¿Por qué `orderCreated` va antes de `tryPush`?
- **Commit:** `Generar órdenes en un hilo propio con seed reproducible`

### 3.3 `feat/bursts-queuefull`: ráfagas y `queueFull`
- **Escribes:** ráfagas de 1 a `burstMax` y el rechazo `queueFull` cuando `tryPush` devuelve la orden.
- **Prueba:** config con `burstMax: 40` y `maxPending: 15` → aparecen rechazos `queueFull`.
- **Commit:** `Ráfagas de órdenes y rechazo queueFull`

### 3.4 `feat/unreachable`: destinos inalcanzables
- **Escribes:** rechazo `unreachable` usando `distanceMeters` (por ahora sin precálculo; el precálculo en paralelo llega el día 5).
- **Prueba:** config con un nodo aislado → aparecen rechazos `unreachable`.
- **Commit:** `Rechazar órdenes con destino inalcanzable`

### 3.5 `feat/validate-log`: validador del log
- **Escribes:** `tools/validate_log.py` (Python, no es parte concurrente: aquí la IA puede ayudarte más).
- **Prueba:** valida un log real y luego uno que rompas a mano: tiene que detectar el error.
- **Commit:** `Agregar validador del log y del resumen`

---

## Día 4: Concurrencia III (despacho, restaurantes, entregas, averías)

Meta del día: las órdenes se entregan de punta a punta, respetando slots, bolsa y orden de llegada.

### 4.1 `feat/courier-offer`: la bolsa del courier
- **Escribes:** `Courier::offer(Order)`, que acepta si hay espacio (bajo lock, registra `orderAssigned`) o te
  devuelve la orden; `run()` despierta cuando tiene trabajo. Se elimina el paseo aleatorio.
- **Concepto:** volver a verificar bajo lock (check-then-act, 07/ex.40).
- **Commit:** `Couriers aceptan órdenes respetando bagCapacity`

### 4.2 `feat/dispatcher-simple`: despacho secuencial
- **Escribes:** hilo `Dispatcher` que toma del `OrderBook` (`cv.wait` con predicado) y ofrece al courier más cercano.
- **Prueba:** aparecen `orderAssigned`. **Guarda esta versión**: es el "antes" del experimento del día 6.
- **Commit:** `Dispatcher secuencial: asigna al courier más cercano`

### 4.3 `feat/restaurant-slots`: ventanillas FIFO
- **Escribes:** `src/Restaurant.h/.cpp` con `deque` de llegada, `inUse` y `cv.wait(pred: soy el primero && hay slot)`;
  `pickupStarted` se registra dentro del lock.
- **Concepto (07):** monitor FIFO sin semáforos (C++17); `notify_all`.
- **Prueba:** con `pickupSlots: 1`, nunca hay dos couriers recogiendo a la vez.
- **Commit:** `Restaurantes con slots y atención en orden de llegada`

### 4.4 `feat/pickup-deliver`: ciclo completo
- **Escribes:** recoger (`assigned → carried`, `orderPickedUp`) y entregar (`orderDelivered`, `notifyCapacity`).
- **Prueba:** el validador da OK con entregas reales y los colores azul/verde/gris en la ventana.
- **Commit:** `Ciclo completo: recoger y entregar órdenes`

### 4.5 `feat/no-courier`: `noCourier`
- **Escribes:** `putBack` + contador de generación en `OrderBook`; rechazo cuando pasa `acceptTimeoutMs`.
- **Concepto:** evitar el despertar perdido (el aviso queda en el estado, no solo en el notify).
- **Commit:** `Rechazo noCourier y reintento al liberarse capacidad`

### 4.6 `feat/breakdowns`: averías y traspaso atómico
- **Escribes:** avería aleatoria en un viaje; `transferBag(from, to)` con `std::scoped_lock(from.m_, to.m_)`.
- **Concepto (06):** dos locks a la vez sin deadlock.
- **Prueba:** `breakdownProbability: 1` corre y termina sin colgarse; validador OK.
- **Commit:** `Averías con traspaso todo-o-nada usando scoped_lock`

---

## Día 5: Paralelismo I (repartir trabajo entre núcleos)

> Concurrencia es **estructurar** el programa en tareas independientes; paralelismo es **ejecutarlas a la vez
> en varios núcleos** para terminar antes. Hoy usamos `std::async`, `promise/future` y un pool de hilos para que
> el cálculo pesado (Dijkstra) corra en paralelo.

### 5.1 `feat/parallel-precompute`: precálculo en paralelo con `std::async`
- **Escribes:** `precomputeRoutingTables()`: una tarea `std::async(std::launch::async, ...)` por restaurante; el
  rechazo `unreachable` pasa a usar la tabla (O(1)).
- **Concepto (04):** task + future + launch policy; por qué `launch::async` y no la política por defecto.
- **Prueba:** mide el tiempo del precálculo en serie vs en paralelo y anótalo (sirve para el día 6).
- **Commit:** `Precalcular distancias en paralelo con std::async`

### 5.2 `feat/quote-pool`: pool de hilos de cotización
- **Escribes:** `src/QuotePool.h/.cpp`: `vector<std::thread>` de workers que toman tareas de una cola con cv.
  El tamaño es `hardware_concurrency() - 1` (manejando que devuelva 0).
- **Concepto (01/03/07):** decisión según el hardware; pool = productor/consumidor de tareas.
- **Commit:** `Pool de hilos para cotizaciones según hardware_concurrency`

### 5.3 `feat/quote-promise`: cotizaciones con `promise/future`
- **Escribes:** cada tarea lleva un `std::promise<Quote>`; el dispatcher guarda los `future` y elige el menor ETA.
- **Preguntas:** ¿por qué aquí **no** `std::async`? (Pista: el destructor de su future bloquea.)
- **Commit:** `Cotizaciones en paralelo con promise/future`

### 5.4 `feat/quote-timeout`: límite de tiempo
- **Escribes:** `future.wait_until(deadline)` con **un solo** deadline (`quoteTimeoutMs`); las tardías se ignoran.
- **Prueba:** con un retraso artificial en la cotización, la ventana no se congela.
- **Commit:** `Ignorar cotizaciones que superan quoteTimeoutMs`

### 5.5 `feat/quote-exceptions`: errores entre hilos
- **Escribes:** `set_exception` en el worker y `try/catch` alrededor de `get()` en el dispatcher.
- **Prueba:** con un fallo inyectado (por ejemplo, la variable de entorno `DELIVERY_SIM_QUOTE_FAULT=0.3`) el programa sigue.
- **Commit:** `Propagar errores de cotización como excepciones`

---

## Día 6: Paralelismo II (medir), herramientas y experimentos

### 6.1 `feat/tsan-clean`: ThreadSanitizer en cero
- Corre TSAN en todos los escenarios. Lo que reporte dentro de Qt/GLib/X11/GDAL (la ventana de OpenCV) se
  suprime con `__tsan_default_suppressions`; lo que sea **tuyo** se arregla.
- Guarda la salida completa en `analysis/`.
- **Commit:** `ThreadSanitizer sin reportes; supresiones para bibliotecas de terceros`

### 6.2 `feat/asan`: memoria
- ASan/LSan o Valgrind sobre una corrida completa con señal. Salida en `analysis/`.
- **Commit:** `Análisis de memoria con AddressSanitizer`

### 6.3 `exp/sequential-vs-parallel`: experimento con cambio de diseño (obligatorio)
- Mismo config y misma seed: dispatcher **secuencial** (la versión del día 4) vs **paralelo** (día 5), con un retraso
  artificial por cotización. Mide la latencia de asignación (`t_assigned − t_created`) y los `noCourier`.
- **Commit:** `Experimento: cotización secuencial vs paralela`

### 6.4 `exp/pool-size`: escalado (Amdahl)
- Tamaño del pool 1, 2, 4 y `hardware_concurrency()`; ¿dónde deja de mejorar y por qué?
- **Commit:** `Experimento: tamaño del pool y ley de Amdahl`

### 6.5 `test/scenarios`: escenarios del revisor
- `burst`, `slots1`, `bag1`, `breakdown1`, `unreachable`, `idle` (CPU) + script que corre cada uno N veces con
  señales aleatorias.
- **Commit:** `Escenarios de prueba y corrida repetida con señales`

---

## Día 7: informe y defensa

### 7.1 `docs/schematic`: esquema de hilos y canales y jerarquía de locks (actualizados al código real)
### 7.2 `docs/concept-map`: mapa de conceptos con `archivo:línea` reales + argumento de no-data-race por objeto
### 7.3 `docs/experiments-tools`: tablas de experimentos y de herramientas en el README

**Defensa:** responde en voz alta la autoevaluación de la masterclass (parte 12) y explica cada fila de tu mapa
de conceptos.

---

## 3. Si te atrasas

| Si al final del día… | Haz esto |
|----------------------|----------|
| 2 no se mueven los couriers | Día 3 por la mañana termina 2.2–2.4; recorta 3.5 (validador) a una versión mínima |
| 4 no hay entregas | Prioriza 4.1–4.4; las averías (4.6) pueden ir el día 5 en la mañana |
| 5 el pool no funciona | Deja el dispatcher secuencial y haz 5.1 (async); el experimento 6.3 compara async vs serie |
| 6 TSAN reporta algo tuyo | Todo lo demás espera: sin "No data races" no se aprueba |
