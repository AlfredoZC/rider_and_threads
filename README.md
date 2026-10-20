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

### Imagen y datos

Tanto la imagen (`data/equipetrol.png`, 965 × 1371 px) como los datos de calles (`data/equipetrol.osm`) se
exportaron de [openstreetmap.org](https://www.openstreetmap.org) con **la misma caja geográfica**:

| north | south | west | east |
|-------|-------|------|------|
| −17.7535 | −17.7765 | −63.2040 | −63.1870 |

La caja (~1.8 km × 2.5 km) encierra Av. San Martín de punta a punta con ~200–300 m de margen. Con el norte
arriba, una latitud/longitud se convierte a píxel con una proyección lineal sobre esos bounds (en 2.5 km el error
frente a la proyección Mercator de OSM es menor que un píxel).

### Dónde están los anillos

En esta zona, OpenStreetMap rotula los anillos con nombres de avenida:

| Anillo | Nombre en OSM | Dónde se ve |
|--------|---------------|-------------|
| Segundo Anillo | Avenida Cristóbal de Mendoza | Abajo a la derecha, donde termina Av. San Martín |
| Tercer Anillo | Avenida Noel Kempff Mercado / Tercer Anillo Externo, con sus rotondas | La curva que cruza el centro de la imagen |
| Cuarto Anillo | Avenida Antonio Vaca Díez | Arriba a la izquierda, donde empieza Av. San Martín |
| — | Avenida San Martín (en su tramo norte corre junto a Av. Leigue Castedo) | Diagonal de noroeste a sureste |

Av. San Martín une el Cuarto Anillo, en (−17.7571, −63.2011), con el Segundo, en (−17.7724, −63.1903), a ~3.5 km
y ~1.5 km de la Plaza 24 de Septiembre respectivamente.

### Red de calles

`config/equipetrol.json` se genera con `tools/osm_extract.py` a partir de `data/equipetrol.osm`:

```bash
python3 tools/osm_extract.py data/equipetrol.osm config/equipetrol.json --all-streets --preview docs/graph_preview.png
```

| Elemento | Cantidad | Mínimo exigido |
|----------|----------|----------------|
| Intersecciones y nodos de forma | 668 | 25 |
| Segmentos de calle | 1027 (259 de un solo sentido) | 40 |
| Restaurantes | 6 | 6 |

Decisiones del modelado:

- **Calles incluidas:** avenidas (`trunk`, `primary`, `secondary`, `tertiary`), sus enlaces y retornos
  (`*_link`) y todas las calles residenciales. Las pasarelas peatonales, los senderos y las vías de servicio
  no se incluyen.
- **Curvas:** entre dos cruces se conservan los puntos de forma necesarios (Douglas-Peucker, tolerancia 6 m),
  para que ningún tramo atraviese una manzana. El script mide cuánto se aleja cada tramo de la calle real y
  avisa si supera 20 m. Los pocos casos (≤ 27 m) son cruces de avenidas de doble calzada, donde el nodo queda
  entre ambas calzadas.
- **Sentidos:** las calles residenciales y los enlaces respetan el `oneway` de OSM. Las avenidas de doble calzada
  (dos vías de un solo sentido en OSM) se modelan como calles de doble sentido.
- **Conectividad:** se conserva la mayor componente fuertemente conexa: desde cualquier nodo se puede llegar a
  cualquier otro respetando los sentidos.
- **Restaurantes:** son restaurantes reales de OSM, elegidos de forma que queden lo más separados posible entre
  sí: Burger King (entre el 3er y el 4to anillo), Puerto Milanesa, Fazenda y Pits Burguer (Tercer Anillo),
  La Palette e I'Milenza (entre el 2do y el 3er anillo).

![Grafo sobre el mapa](docs/graph_preview.png)

## 6. Decisiones de diseño

PENDIENTE. Decisiones que el enunciado deja abiertas y que se documentan aquí:

- Generación de órdenes: distribución de los intervalos y regla de las ráfagas.
- Elección del destino de entrega y definición de `unreachable`.
- Estimación del tiempo de entrega (ETA) usada por el dispatcher y número de candidatos cotizados.
- Qué órdenes se transfieren en una avería (asignadas y/o recogidas), qué pasa si ningún courier tiene espacio
  y si la avería es permanente o se repara.
- Tiempos simulados fijos (entrega en el restaurante, entrega al cliente, reparación).
- Qué es reproducible con una misma `seed` y qué depende del planificador del sistema operativo.

## 6b. La ventana

![La simulación en marcha](docs/ventana.png)

Cada punto es un courier, dibujado en su posición real sobre la calle que recorre, y
esa posición avanza de forma continua cuadro a cuadro. Los colores distinguen los
estados: gris libre, amarillo yendo al restaurante, **naranja esperando turno en la
puerta**, verde entregando, rojo averiado. Las líneas violetas unen a cada courier con
la puerta del cliente al que está yendo.

El naranja no es decorativo: ver varios juntos en un restaurante es la exclusión mutua
de los `pickupSlots` ocurriendo a la vista.

La capa del grafo de calles está oculta por defecto, porque con 1027 calles y sus
flechas tapa el mapa. La tecla `G` la muestra cuando hace falta inspeccionar la red.

## 7. Esquema del programa: hilos y canales

Estado: lo marcado **[hecho]** está implementado y probado; **[plan]** es diseño todavía no
escrito. Se actualiza en cada avance.

### Flujo de una orden, de punta a punta

```
GENERADOR   [hecho]   espera un intervalo aleatorio (wait_for, 0% CPU)
                      crea la orden y llama a OrderBook.admit()
                              |
                              |  si pending_ == maxPending -> orderRejected(queueFull)
                              v
            [plan]    despierta al dispatcher
                              |
DISPATCHER  [plan]    espera en su cv (0% CPU) <-------------------+
                      toma las ordenes sin asignar
                      pide una estimacion POR COURIER, en paralelo
                        - un std::future por candidato
                        - plazo de quoteTimeoutMs REALES
                        - las que llegan tarde se descartan
                        - si una lanza, la excepcion llega en get() y no tumba nada
                      elige el mejor tiempo estimado
                      OrderBook.assign(orden, courier)   <- candado: microsegundos
                      courier.offer(orden)               -> despierta al courier
                              |
COURIER     [hecho]   espera en su cv (0% CPU) <-------------------+
            [plan]    calcula su ruta con SU PROPIO Router
                      viaja tramo a tramo                <- SIN candados, segundos
                      llega al restaurante -> pickupStarted
                      espera a que la comida este lista si llego antes
                      OrderBook.pickedUp()               <- candado: microsegundos
                      viaja al cliente                   <- SIN candados
                      OrderBook.delivered()              <- candado: microsegundos
                      vuelve a esperar en su cv
```

El patrón que se repite: **los candados aparecen en instantes puntuales y duran
microsegundos; el trabajo largo ocurre siempre afuera**.

### Por qué la orden no se "reserva" sosteniendo un candado

Un viaje largo de este mapa son 213 s simulados, o 3,5 s reales con `timeScale: 60`.
Sostener un mutex durante todo ese tiempo dejaría al renderer sin poder llamar a
`snapshot()` (lo necesita 30 veces por segundo) y congelaría la ventana.

La exclusividad que dura en el tiempo **se modela con estado, no con un candado**: la orden
está reservada porque su estado dice `assigned` y `assignedCourier` apunta a su courier.
Ningún otro la puede tomar porque `canTransition(assigned, assigned)` devuelve `false`, y esa
comprobación ocurre dentro del candado del `OrderBook`, que dura microsegundos.

> **Candado = operación corta e indivisible. Estado = propiedad que dura en el tiempo.**

### Hilos de ejecución

| Hilo | Cantidad | Sobre qué hace loop | Cómo espera | Estado |
|------|----------|---------------------|-------------|--------|
| main / render | 1 | Dibuja un frame y revisa teclado y señal | `cv::waitKey` (ritmo de frames) | [hecho] parcial |
| `OrderGenerator` | 1 | Espera un intervalo exponencial y crea 1..`burstMax` órdenes | `cv.wait_for` con predicado, interrumpible | [hecho] |
| `Dispatcher` | 1 | Toma una orden, pide cotizaciones en paralelo y la asigna | `condition_variable` | [plan] |
| Tareas de cotización | una por candidato | Calcula `estimateDeliveryMs` y cumple su `future` | no espera: termina | [hecho] la función, [plan] el lanzamiento |
| `CourierActor` | `fleet.couriers` (8) | Idle → restaurante → recoger → entregar | su propia `condition_variable` | [hecho] el ciclo de vida, [plan] el reparto |

### Canales entre hilos

| # | Canal | Mecanismo | Estado |
|---|-------|-----------|--------|
| 1 | `OrderGenerator` → `OrderBook` | monitor acotado; `admit()` verifica y agrega bajo un mismo candado | [hecho] |
| 2 | `OrderBook` → `Dispatcher` | notificación al admitir una orden | [plan] |
| 3 | `Dispatcher` → tareas de cotización | `std::future` con plazo de `quoteTimeoutMs` reales | [plan] |
| 4 | `Dispatcher` → `CourierActor` | `offer()` sobre el monitor del courier + `notify_one` | [plan] |
| 5 | `CourierActor` ↔ `Restaurant` | monitor con cola FIFO y `pickupSlots` | [plan] |
| 6 | Courier averiado → courier receptor | `std::scoped_lock` sobre los dos couriers | [hecho] |
| 7 | Todos → `EventLog` | monitor; una línea JSON completa por evento | [hecho] |
| 8 | Renderer / dispatcher ← `CourierActor` | `snapshot()` devuelve una copia tomada bajo candado | [hecho] |
| 9 | main → todos | `requestStop()` en cada monitor y `join()` de todos los hilos | [hecho] |

## 8. Jerarquía de locks y ausencia de deadlock

Un interbloqueo necesita que dos hilos tomen dos candados en **orden distinto**. La defensa es
fijar un orden total y no violarlo nunca.

| Nivel | Mutex | Puede tomar a continuación | Estado |
|-------|-------|----------------------------|--------|
| 1 | `CourierActor` (uno, o **dos a la vez** con `std::scoped_lock`) | `OrderBook`, `EventLog` | [hecho] |
| 2 | `OrderBook`, `Restaurant` | `EventLog` | [hecho] el libro |
| 3 | `EventLog` | ninguno (hoja) | [hecho] |

Dos propiedades lo sostienen hoy:

- **`EventLog::write()` no llama a nada que pueda tomar otro candado.** Es hoja por
  construcción, así que registrar un evento teniendo tomado el candado del `OrderBook` o el de
  un courier no puede bloquear a nadie.
- **El traspaso por avería toma dos candados de courier a la vez, y por eso usa
  `std::scoped_lock`.** Tomarlos uno por uno sería el interbloqueo de manual: si A se
  avería y elige a B mientras B se avería y elige a A, cada uno tendría el suyo
  esperando el del otro para siempre. `scoped_lock` los toma con un algoritmo que
  evita ese ciclo sin importar el orden en que se le pasen, y es lo que hace que las
  averías simultáneas no puedan bloquear el sistema (§4). *Verificado: 300 averías
  simultáneas cruzadas entre dos couriers, sin un solo bloqueo.*
- **`OrderGenerator` suelta su propio candado antes de llamar a `admit()`.** Nunca sostiene dos
  a la vez, de modo que no puede participar de un ciclo.

## 9. Argumento de ausencia de data races

Un objeto por párrafo, con la evidencia que lo respalda.

**`Config`** — se carga y valida por completo antes de crear ningún hilo y después nadie la
modifica. Un dato que nadie escribe no puede tener data races, así que se comparte por
referencia sin ninguna sincronización.

**`CityGraph`** — mismo caso. Se construye una vez y no existe ningún método que lo modifique:
todos son `const`. Lo leen simultáneamente los couriers y las tareas de cotización sin candado.

**`SimClock`** — no guarda la hora en ninguna variable. Solo conserva el instante de arranque y
la escala, fijados en el constructor, y cada consulta le pregunta la hora al sistema operativo;
`steady_clock::now()` es thread-safe por norma. *Verificado: 4 hilos leyendo `nowMs()` en
paralelo, monótono en los cuatro.*

**`Router`** — es el único objeto con estado de trabajo mutable, y por eso **no se comparte**:
cada hilo construye el suyo. La firma `shortestPath()` se deja deliberadamente no-`const` para
que el compilador frene cualquier intento de compartirlo como `const&`. *Verificado: 1600
estimaciones calculadas por 8 hilos, cada uno con su Router, dan exactamente los mismos números
que calculadas de a una.*

**`EventLog`** — monitor. El `ofstream` y el mutex son privados y la única forma de escribir es
`write()`. El formateo del JSON ocurre fuera del candado; dentro queda solo leer el reloj y
volcar la línea, y la marca de tiempo se toma adentro para que el orden de las líneas sea el
orden de las `t`. *Verificado: 8 hilos escribiendo 500 eventos cada uno, las 4000 líneas parsean
como JSON, ninguna perdida, `t` nunca retrocede. ThreadSanitizer: 0 reportes, y el control
negativo con el `lock_guard` comentado sí produce un race (`analysis/tsan_event_log.txt`).*

**`OrderBook`** — monitor. No expone un `pendingCount()` público a propósito: si lo hiciera, el
llamador consultaría y agregaría en dos pasos y entre ellos se soltaría el candado. `admit()`
verifica y agrega bajo un mismo candado. *Verificado: 1600 intentos desde 16 hilos contra un
`maxPending` de 50 admiten exactamente 50.*

**`CourierActor`** — monitor con `condition_variable`. La bandera de parada se escribe con el
candado tomado, que es lo que impide el aviso perdido; `wait()` lleva predicado, que es lo que
cubre los despertares espurios. *Verificado: 8 couriers ociosos 400 ms consumieron 0,151 ms de
CPU (girando habrían consumido unos 3200), terminan en 3 ms, y 300 ciclos de arrancar y parar
no colgaron nunca. ThreadSanitizer: 0 reportes (`analysis/tsan_courier_actor.txt`).*

**`OrderGenerator`** — es dueño exclusivo de su `OrderFactory` y su `mt19937`, así que esos dos
no llevan sincronización: un dato con un único dueño no se protege, se posee. Lo único
compartido es su bandera de parada y su contador, bajo su propio candado.

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

Cada experimento varía **una sola** magnitud, mantiene el resto igual —incluida la semilla,
que hace la corrida reproducible— y mide sobre el resumen final, que está verificado contra
el log por `tools/validate_log.py`. Los datos crudos están en `analysis/`.

### Experimento 1: cuándo tomar el lugar de retiro

**Qué se cambió.** El courier pedía un lugar en el restaurante al llegar y después esperaba
ahí a que la comida estuviera lista. Ahora espera la comida primero y pide el lugar después.

**Qué se mantuvo igual.** Semilla 42, 90 minutos simulados, `timeScale: 300`, sin averías.

| configuración por defecto (prep 1–5 min, 1–2 slots) | antes | después |
|---|---|---|
| entregadas | 112 | 112 |
| rechazadas | 290 | 287 |
| tiempo medio de entrega | 1198 s | 1185 s |

| recurso en disputa (prep 15–20 min, 1 slot) | antes | después |
|---|---|---|
| entregadas | 81 | **88** (+8,6 %) |
| rechazadas | 319 | 313 |
| tiempo medio de entrega | 1515 s | **1482 s** (−33 s) |

**Conclusión.** El cambio de diseño vale, pero su efecto es condicional. Con la configuración
por defecto no se nota, y el motivo es más interesante que el número: `prepTimeMs` se cuenta
desde que la orden **se crea**, y el viaje al restaurante dura cientos de segundos, así que
cuando el courier llega la comida ya está lista. El cuello de botella del sistema no es la
cocina sino el viaje. Cuando la preparación domina, retener un lugar sin usarlo sí cuesta
entregas.

Se adoptó el cambio porque el razonamiento no depende de la configuración: un recurso escaso
retenido sin usar bloquea a otros exactamente igual que uno en uso, y el enunciado permite
cualquier `prepTimeMs`.

### Experimento 2: dónde ocurre la avería

**Qué se cambió.** La avería se sorteaba una sola vez, antes de salir a repartir. Ahora
también se sortea después de retirar la comida, con la mitad de la probabilidad en cada punto
para no alterar la tasa por viaje.

**Qué se midió.** Cuántos traspasos mueven una orden ya retirada: antes, **cero por
construcción**; después, posibles. El caso es poco frecuente en corridas aleatorias porque los
couriers suelen averiarse antes de llegar al restaurante, de modo que la evidencia es el test
determinista `tests/test_breakdown.cpp`.

**Conclusión.** El traspaso que de verdad importa —mover comida que ya está en una bolsa y no
puede volver al restaurante— era inalcanzable, y con él el `std::scoped_lock` que lo protege
no se ejercitaba nunca en una corrida real.

### Experimento 3: devolver a la cola en lugar de traspasar (descartado)

**Qué se cambió.** Que las órdenes que el courier todavía no había retirado volvieran a la cola
en vez de traspasarse, dejando el todo-o-nada sólo para las que llevan comida.

**Qué se midió.** Averías concretadas en la misma corrida: de **2 a 8**. Exigir que un solo
compañero se haga cargo de *todas* las órdenes las cancelaba casi siempre.

**Conclusión: se revirtió.** El validador del log lo rechazó por dos motivos. `orderReassigned`
quedaba sin `to`, que el §6 define obligatorio como id del courier receptor; y la orden pasaba
de `assigned` a `created` sin ningún evento que lo registrara, mientras el §6 exige que todo
cambio de estado quede en el log.

Que `orderReassigned` lleve `from` y `to`, los dos ids de courier, implica que las órdenes de
un averiado van **a otro courier**: no hay evento para "vuelve a la cola" porque esa no es la
mecánica prevista. Un experimento que mejora una métrica y rompe una regla no es una mejora.

### Experimento 4: cantidad de lugares de retiro

**Qué se cambió.** `pickupSlots` de cada restaurante.

**Qué se midió.** Con un solo lugar por restaurante, la tasa de rechazo sube marcadamente: los
couriers pasan más tiempo en la fila, entregan más lento, la cola llega a `maxPending` y las
órdenes nuevas se rechazan con `queueFull`. Una corrida de 22 s con la configuración por
defecto dio `created 447 = delivered 98 + rejected 304 + pending 45`.

**Conclusión.** El sistema traslada la escasez de un recurso a la métrica de rechazo, que es el
comportamiento correcto: no pierde órdenes ni las acumula indefinidamente, las rechaza al
entrar.

## 12. Análisis con herramientas

Las salidas completas y sin editar están en `analysis/`.

| Herramienta | Versión | Qué observó | Qué reportó |
|---|---|---|---|
| ThreadSanitizer | GCC 13.3 | `EventLog` con 8 hilos × 500 eventos | 0 reportes |
| ThreadSanitizer | GCC 13.3 | `CourierActor`: 8 hilos ociosos + 300 ciclos de arranque y parada | 0 reportes |
| AddressSanitizer + LeakSanitizer | GCC 13.3 | corrida completa de 25 s con ráfagas, averías y cierre por señal | 0 errores; 320 bytes en 3 fugas, **ninguna del código propio** |
| `tools/validate_log.py` | propia | el log y el resumen de cada corrida | ver abajo |

### ThreadSanitizer

```
g++ -std=c++17 -Wall -Wextra -g -O1 -fno-omit-frame-pointer -fsanitize=thread -pthread
setarch $(uname -m) -R ./tests/bin/tsan_event_log
```

El `setarch -R` desactiva la aleatorización de direcciones. Sin eso el binario instrumentado
**segfaulta** en esta máquina, por un conflicto conocido entre el mapa de memoria de
ThreadSanitizer y el rango de aleatorización de los núcleos recientes.

**Control negativo.** Una ejecución limpia por sí sola no demuestra que el experimento tenga
capacidad de detección: podría ocurrir que la prueba no ejercite el código o que los hilos
nunca coincidan. Por eso se repitió el análisis sobre una copia con el `lock_guard` del
registro deliberadamente desactivado. En esa variante la herramienta informó una condición de
carrera y el programa abortó, lo que confirma que la ejecución limpia es significativa.

### AddressSanitizer

```
cmake -DENABLE_ASAN=ON .. && make
ASAN_OPTIONS=detect_leaks=1 ./delivery_sim config/equipetrol.json --log events.log
```

Cero errores de memoria. Las 320 fugas provienen de `libfontconfig`, llamada desde `FcInit` a
través de `libQt5Gui`, que es lo que OpenCV usa para la ventana: se asignan una vez al
inicializar el sistema de fuentes y el sistema operativo las recupera al terminar. Ninguna
traza menciona un archivo de `src/`, comprobado filtrando la salida completa por los nombres
del proyecto.

### Validador del log

`tools/validate_log.py` relee el log y el resumen y comprueba cuatro cosas que el programa no
puede verificarse a sí mismo: que cada línea lleve exactamente los campos que la tabla del §6
asigna a su evento; que cada orden recorra una secuencia de estados legal y termine en un único
estado final; que la `t` nunca decrezca; y que cada contador del resumen sea igual a la cantidad
de líneas que coinciden.

**La máquina de estados está reimplementada en Python**, así que no comparte código con la de
C++ y un agujero en una no se refleja en la otra.

Encontró tres defectos reales que ni las pruebas unitarias ni la inspección habían detectado:

1. El resumen informaba siempre `pending: 0`. `markRemainingAsPending()` deja las órdenes en un
   estado final, así que el contador de vivas bajaba a cero justo antes de que `stats()` lo
   leyera. **El mismo error estaba en el test unitario**, escrito con la idea equivocada: la
   prueba y el código se confirmaban entre sí.
2. Las órdenes rechazadas por cola llena no emitían `orderCreated`: el resumen decía 86 creadas
   y el log tenía 68.
3. El experimento 3, descrito arriba.

## 13. Limitaciones conocidas

**La preparación se cuenta desde que la orden se crea**, no desde que el courier llega al
restaurante. Es una simplificación y cambia cuál recurso resulta escaso: con los viajes de este
mapa, la comida siempre está lista antes de que alguien llegue a buscarla, así que los lugares
de retiro casi nunca son el cuello de botella. Ver el experimento 1.

**Un courier averiado no vuelve al servicio.** El enunciado no menciona reparaciones, así que
la avería se modela como una baja definitiva. En corridas largas con probabilidad alta, la flota
se agota y la tasa de entrega cae a cero.

**La estimación de tiempo ignora las órdenes que el courier ya lleva.** Un courier con la bolsa
llena da el mismo número que uno vacío, así que el dispatcher puede elegirlo creyendo que está
libre.

**El segundo tramo de la estimación se recalcula por candidato.** El trayecto del restaurante al
cliente es el mismo para todos los couriers y se computa una vez por cada uno. Con 8 candidatos
son 16 ejecuciones de Dijkstra por orden en lugar de 9.

**La representación del grafo usa un vector de vectores.** Son 668 asignaciones de memoria
dispersas; una representación contigua sería más amigable con la caché en el camino de las
estimaciones.

---

Mapa y datos de calles © OpenStreetMap contributors, disponibles bajo la ODbL.
