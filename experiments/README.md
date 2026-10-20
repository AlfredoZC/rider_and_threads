# Datos crudos de los experimentos

Cada corrida del §11 del README principal está acá con sus tres archivos:

| archivo | qué es |
|---|---|
| `<nombre>.json` | la configuración exacta con la que se corrió |
| `<nombre>.log` | el registro de eventos, una línea JSON por evento |
| `<nombre>.summary.txt` | la salida de `stdout`, cuya última línea es el resumen |

Todas se validaron con `tools/validate_log.py <nombre>.log <nombre>.summary.txt`, salvo la
del experimento 3, que es el caso en que el validador **tiene** que fallar y por eso está.

## Un detalle que hay que decir: las columnas "antes"

Tres experimentos comparan el código actual contra una versión anterior. Esas corridas no
se pueden reproducir desde el árbol entregado, porque ese código ya no existe en él. Para
generarlas se reconstruyó cada versión en un *worktree* de git sobre el commit
correspondiente, y se corrió con **el mismo archivo de configuración** que la versión nueva:
lo único que cambia entre las dos columnas es el código.

| corrida | de dónde salió el binario |
|---|---|
| `exp1_*_antes*` | commit `6847c5e`, el padre de `06381ca` (*El courier toma el lugar de retiro recién cuando la comida está lista*) |
| `exp1_*` y `exp1_*_despues*` | el árbol entregado |
| `exp6_sin_pool_*` | commit `d912b69`, donde el `QuotePool` ya existe pero el dispatcher todavía usa `std::async` por estimación |
| `exp6_con_pool_*` | el árbol entregado |
| `exp3_devolver_a_la_cola` | el árbol entregado **más** la variante descartada, reaplicada a mano sobre `CourierActor::breakDownAndHandOver` |

La variante del experimento 3 nunca se llegó a commitear: se midió, se vio que rompía el §6
y se descartó antes de entrar al repositorio. Su log está acá porque es la evidencia de por
qué se descartó, no porque sea una versión válida del programa.

## Por qué hay repeticiones

El entrelazado de los hilos no es reproducible, ni siquiera con la misma semilla (§4.5 del
README). Con una sola corrida por celda, una diferencia de tres o cuatro entregas no se
distingue del ruido. Por eso el experimento 1, cuya conclusión depende de un efecto chico,
tiene tres corridas por celda —la original más `_r2` y `_r3`— y la tabla del README informa
el rango observado en lugar de un número suelto.

Lo que **sí** es reproducible con la misma semilla son las decisiones de azar: los intervalos
entre llegadas, las ráfagas, los restaurantes, los destinos, los tiempos de preparación y los
sorteos de avería. Por eso las órdenes creadas quedan casi iguales entre corridas, y lo que
se mueve son las entregas, que dependen de quién gana cada carrera.

## Índice

### Experimento 1 — cuándo tomar el lugar de retiro

| configuración | antes | después |
|---|---|---|
| por defecto (prep 1–4 min, 1–2 lugares) | `exp1_por_defecto_antes`, `_antes_r2`, `_antes_r3` | `exp1_por_defecto`, `exp1_por_defecto_despues_r2`, `_r3` |
| recurso en disputa (prep 15–20 min, 1 lugar) | `exp1_recurso_en_disputa_antes`, `_antes_r2`, `_antes_r3` | `exp1_recurso_en_disputa`, `exp1_recurso_en_disputa_despues_r2`, `_r3` |

### Experimento 2 — dónde ocurre la avería

No tiene corridas acá. Su tabla no es numérica: antes del cambio, un traspaso de comida ya
retirada era **inalcanzable por construcción**, y después es posible. La evidencia es el test
determinista `tests/test_breakdown.cpp`, que provoca el caso cuando quiere en lugar de
esperar a que el azar lo produzca.

### Experimento 3 — devolver a la cola (descartado)

`exp3_devolver_a_la_cola`, contra `exp5_averias_flota_8` como línea de base: misma
configuración, el código entregado contra la variante descartada.

### Experimento 4 — cantidad de lugares de retiro

`exp4_un_solo_slot`, contra `exp1_por_defecto` como línea de base.

### Experimento 5 — por qué las averías casi no se concretan

Barrido de la probabilidad con 8 couriers: `exp5_barrido_p005`, `p020`, `p040`, `p060`,
`p080`. Tamaño de flota con la probabilidad fija en 0,6: `exp5_averias_flota_8` y
`exp5_averias_flota_16`.

### Experimento 6 — un pool acotado contra un hilo por estimación

`exp6_sin_pool_r1..r3` contra `exp6_con_pool_r1..r3`, las dos con
`exp6_pool_64_couriers.json`.

### Análisis con herramientas

`herramientas_corrida_completa.json` es la configuración de las corridas de
AddressSanitizer y del validador que están en `analysis/`.
