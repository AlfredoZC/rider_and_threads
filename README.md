# Equipetrol Delivery: simulación concurrente de despacho

Simulación concurrente en **C++17** de una empresa de delivery en **Zona Equipetrol, Santa Cruz de la Sierra**
(entre el Segundo y el Cuarto Anillo). Motocicletas, un flujo continuo de órdenes, restaurantes con
ventanillas limitadas y un único libro de órdenes, todo simulado en tiempo acelerado y dibujado en vivo
sobre una imagen real de OpenStreetMap.

> Mapa y datos de calles: **© OpenStreetMap contributors**. Los datos de OpenStreetMap están disponibles
> bajo la [Open Database License (ODbL)](https://www.openstreetmap.org/copyright).

> **Estado del documento:** en construcción. Las secciones marcadas con `PENDIENTE` se completan a medida
> que avanza la implementación. El esquema y el mapa de conceptos deben coincidir con el código final.

---

## Contenido

1. [Dependencias](#1-dependencias)
2. [Compilación y ejecución](#2-compilación-y-ejecución)
3. [Entrada y salida](#3-entrada-y-salida)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [El mapa](#5-el-mapa)
6. [Decisiones de diseño](#6-decisiones-de-diseño)
7. [Esquema del programa: hilos y canales](#7-esquema-del-programa-hilos-y-canales)
8. [Jerarquía de locks y ausencia de deadlock](#8-jerarquía-de-locks-y-ausencia-de-deadlock)
9. [Argumento de ausencia de data races](#9-argumento-de-ausencia-de-data-races)
10. [Mapa de conceptos](#10-mapa-de-conceptos)
11. [Experimentos](#11-experimentos)
12. [Análisis con herramientas](#12-análisis-con-herramientas)
13. [Limitaciones conocidas](#13-limitaciones-conocidas)

---

## 1. Dependencias

Versiones usadas durante el desarrollo (Ubuntu 24.04.1 LTS sobre WSL2):

| Dependencia | Versión usada | Mínimo exigido | Uso |
|-------------|---------------|----------------|-----|
| g++ | 13.3.0 | 7 | Compilador C++17 |
| CMake | 3.28.3 | 3.11 | Generación del build |
| GNU Make | 4.3 | 4.1 | Build |
| OpenCV | 4.6.0 | 4.1 | **Solo** la ventana (dibujo e imagen de fondo) |
| nlohmann/json | 3.11.3 | — | Lectura de la configuración y escritura del log. Incluida en `third_party/` |
| Valgrind | 3.22.0 | — | Análisis de memoria (DP.11) |

No se usa ninguna otra librería. Toda la concurrencia está construida con la biblioteca estándar de C++17
(`std::thread`, `std::mutex`, `std::condition_variable`, `std::promise`/`std::future`, `std::async`, `std::atomic`).

Instalación en Ubuntu:

```bash
sudo apt install -y build-essential cmake pkg-config libopencv-dev valgrind
```

`nlohmann/json` no hay que instalarla: viene en `third_party/nlohmann/json.hpp`.

## 2. Compilación y ejecución

```bash
mkdir build && cd build
cmake .. && make
./delivery_sim ../config/equipetrol.json
./delivery_sim ../config/equipetrol.json --log run1.log
```

| Argumento | Significado |
|-----------|-------------|
| `<config.json>` (obligatorio) | Ruta al archivo de configuración |
| `--log <path>` (opcional) | Escribe el log de eventos en `<path>` en vez de `events.log` |

Builds instrumentados:

```bash
mkdir build-tsan && cd build-tsan && cmake -DENABLE_TSAN=ON .. && make   # ThreadSanitizer
mkdir build-asan && cd build-asan && cmake -DENABLE_ASAN=ON .. && make   # AddressSanitizer + UBSan
```

> **Nota sobre ThreadSanitizer en kernels recientes:** en kernels con alta aleatorización de memoria
> (por ejemplo WSL2 con kernel 6.6), el binario instrumentado puede abortar al iniciar con
> `FATAL: ThreadSanitizer: unexpected memory mapping` o con un segmentation fault, sin llegar a ejecutar
> el programa. No es un defecto del programa. Se resuelve de dos formas:
>
> ```bash
> setarch "$(uname -m)" -R ./delivery_sim ../config/equipetrol.json   # desactiva ASLR solo para esta ejecución
> sudo sysctl vm.mmap_rnd_bits=28                                     # o: reduce la aleatorización del sistema
> ```

La simulación termina al cumplirse `simulation.durationS` segundos simulados, o al recibir `SIGINT` (Ctrl-C)
o `SIGTERM`. En ambos casos imprime el resumen y sale con código 0.

## 3. Entrada y salida

- **Entrada:** archivo JSON con el formato de la sección 5 del enunciado (`map`, `nodes`, `streets`,
  `restaurants`, `fleet`, `orders`, `dispatch`, `incidents`, `simulation`). Una configuración inválida produce
  un mensaje en `stderr` y un código de salida distinto de cero, sin iniciar la simulación.
- **stdout:** `Simulation has started...` cuando la simulación está lista y corriendo; al final, una única línea
  JSON con el resumen (`created`, `delivered`, `rejected`, `pending`, `reassigned`, `breakdowns`,
  `avgDeliveryMs`, `maxDeliveryMs`, `deliveredPerCourier`).
- **Log de eventos:** un objeto JSON por línea (`orderCreated`, `orderAssigned`, `pickupStarted`,
  `orderPickedUp`, `orderDelivered`, `orderRejected`, `courierBreakdown`, `orderReassigned`,
  `simulationStopping`).
- **Ventana:** mapa de OpenStreetMap con restaurantes, destinos de entrega activos, cada moto como un punto
  que se mueve por las calles (color según estado), panel de contadores y atribución.

## 4. Estructura del repositorio

```
CMakeLists.txt            build (target delivery_sim, opciones ENABLE_TSAN / ENABLE_ASAN)
config/equipetrol.json    configuración oficial
config/tests/             escenarios de prueba (válidos e inválidos)
data/equipetrol.png       imagen de OpenStreetMap usada de fondo
src/                      código fuente C++17
third_party/nlohmann/     librería JSON (un solo header)
tools/                    scripts de apoyo: validación del log, corrida de escenarios, extracción OSM
experiments/              datos crudos de los experimentos (configs, logs, resúmenes)
analysis/                 salida completa sin editar de las herramientas de análisis
```

## 5. El mapa

PENDIENTE (Fase 1):

- Fuente de la imagen y procedimiento de exportación desde OpenStreetMap.
- Bounding box: `north = ?`, `south = ?`, `west = ?`, `east = ?`.
- Cómo se extrajeron intersecciones y segmentos (consulta Overpass o procedimiento manual), manejo de calles
  curvas (nodos intermedios de forma) y de calles de un solo sentido.
- Tamaño del grafo: `? nodos`, `? segmentos`, `? restaurantes`, y cómo quedan repartidos entre los anillos.
- Captura de la ventana con el grafo superpuesto a la imagen.

## 6. Decisiones de diseño

PENDIENTE. Decisiones que el enunciado deja abiertas y que se documentan aquí:

- Generación de órdenes: distribución de los intervalos y regla de las ráfagas.
- Elección del destino de entrega y definición de `unreachable`.
- Estimación del tiempo de entrega (ETA) usada por el dispatcher y número de candidatos cotizados.
- Qué órdenes se transfieren en una avería (asignadas y/o recogidas), qué pasa si ningún courier tiene espacio
  y si la avería es permanente o se repara.
- Tiempos simulados fijos (entrega en el restaurante, entrega al cliente, reparación).
- Qué es reproducible con una misma `seed` y qué depende del planificador del sistema operativo.

## 7. Esquema del programa: hilos y canales

> Diseño planificado. PENDIENTE: actualizar con los nombres reales del código al terminar.

### Hilos de ejecución

| Hilo | Cantidad | Sobre qué hace loop | Cómo espera |
|------|----------|---------------------|-------------|
| main / render | 1 | Dibuja un frame, revisa la señal y el fin de `durationS` | `cv::waitKey` (ritmo de frames) |
| OrderGenerator | 1 | Espera un intervalo aleatorio simulado y crea 1..`burstMax` órdenes | `condition_variable::wait_until` interrumpible |
| Dispatcher | 1 | Toma una orden del libro, pide cotizaciones en paralelo y la asigna | `condition_variable` del `OrderBook` |
| QuoteWorker | según `hardware_concurrency()` | Calcula una cotización y cumple su `promise` | `condition_variable` del `QuotePool` |
| Courier | `fleet.couriers` | Idle → restaurante → recoger → entregar (con averías) | su propia `condition_variable` / la del restaurante |

### Canales entre hilos

| # | Canal | Mecanismo |
|---|-------|-----------|
| 1 | OrderGenerator → OrderBook → Dispatcher | Monitor acotado productor/consumidor |
| 2 | Dispatcher → QuotePool → Dispatcher | Cola de tareas + `std::promise`/`std::future` con `wait_until` |
| 3 | Dispatcher → Courier | Método del monitor `Courier::offer` + notificación |
| 4 | Courier ↔ Restaurant | Monitor con cola FIFO y slots |
| 5 | Courier averiado → Courier receptor | `std::scoped_lock` sobre ambos couriers |
| 6 | Courier → OrderBook / Fleet | Aviso de capacidad liberada (contador de generación + notificación) |
| 7 | Todos → EventLog | Monitor (líneas completas + contadores) |
| 8 | Render / QuoteWorkers ← Courier | `snapshot()` que devuelve una copia |
| 9 | main → todos | `stop()` en cada monitor; señal vía `std::atomic<bool>` |

PENDIENTE: diagrama.

## 8. Jerarquía de locks y ausencia de deadlock

> Diseño planificado. PENDIENTE: confirmar contra el código final.

| Nivel | Mutex | Puede tomar a continuación |
|-------|-------|----------------------------|
| 1 | `Courier` (uno, o dos a la vez con `std::scoped_lock`) | `EventLog`, `Fleet` |
| 2 | `OrderBook`, `Restaurant`, `QuotePool`, `Fleet` | `EventLog` |
| 3 | `EventLog` | ninguno |

PENDIENTE: argumento completo.

## 9. Argumento de ausencia de data races

PENDIENTE: un párrafo por objeto compartido (`CityGraph`, `SimClock`, `EventLog`, `OrderBook`, `Restaurant`,
`Courier`, `Fleet`, `QuotePool`, flag de señal).

## 10. Mapa de conceptos

PENDIENTE: completar la columna *Dónde* con `archivo:línea` reales y la justificación de cada fila.

| Carpeta | Concepto | Dónde se usa | Por qué es el adecuado ahí y qué fallaría sin él |
|---------|----------|--------------|--------------------------------------------------|
| 01 | Creación explícita de hilos y destino deliberado (join) para cada uno | | |
| 01 | Una decisión que depende del paralelismo disponible del hardware | | |
| 02 | Datos pasados a un hilo por valor, por referencia y por move | | |
| 02 | Órdenes que viajan por el sistema por move, existiendo en un solo lugar | | |
| 03 | Funciones miembro de objetos compartidos ejecutándose en hilos; contenedores de hilos | | |
| 03 | Propiedad compartida que garantiza el tiempo de vida; sin punteros crudos dueños | | |
| 04 | Tareas que devuelven un resultado por un future, con launch policy explícita | | |
| 04 | Un par promise/future usado como canal de un solo disparo entre dos hilos | | |
| 04 | Una espera con límite de tiempo y una excepción que cruza de un hilo a otro | | |
| 05 | Argumento escrito, por objeto compartido, de por qué no hay data race | [Sección 9](#9-argumento-de-ausencia-de-data-races) | |
| 05 | Un tipo seguro de copiar o mover porque sus funciones miembro especiales son correctas | | |
| 06 | Exclusión mutua solo con locks de alcance; secciones críticas cortas | | |
| 06 | Una operación que necesita dos locks a la vez y por qué no puede hacer deadlock | | |
| 07 | Monitores: datos compartidos inaccesibles sin su lock | | |
| 07 | Hilos que se bloquean hasta que se cumple una condición, robustos a despertares espurios y a un despertar final de apagado | | |

## 11. Experimentos

PENDIENTE (Fase 10). Para cada experimento: qué se cambió, qué se mantuvo igual, qué se midió y qué se concluye.
Los datos crudos de cada corrida están en `experiments/`.

| Experimento | Variante | Corridas | created | delivered | rejected | avgDeliveryMs | Otra métrica |
|-------------|----------|----------|---------|-----------|----------|---------------|--------------|
| | | | | | | | |

**Cambio de diseño motivado por un experimento:** PENDIENTE (antes/después con números).

## 12. Análisis con herramientas

PENDIENTE (Fase 9). Cada herramienta observó una corrida completa de `config/equipetrol.json` con ráfagas,
averías y apagado por señal. La salida completa y sin editar está en `analysis/`.

| Herramienta | Versión | Comando exacto | Qué reportó | Qué se cambió | Resultado al re-correr |
|-------------|---------|----------------|-------------|---------------|------------------------|
| ThreadSanitizer | | | | | |
| AddressSanitizer / LeakSanitizer | | | | | |
| Valgrind Memcheck | | | | | |

Reportes que persisten en la versión final y su explicación: PENDIENTE.

## 13. Limitaciones conocidas

PENDIENTE.

---

Mapa y datos de calles © OpenStreetMap contributors, disponibles bajo la ODbL.
