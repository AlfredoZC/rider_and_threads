# ADR 0002: escenario de ciudad para medir la GPU

- **Estado:** aceptada (2026-11-20)
- **Contexto:** el enunciado pide *"a scale where a GPU can matter"*: una configuración de ciudad
  (por ejemplo, un grafo de 3.000 a 5.000 intersecciones de OpenStreetMap, 2.000 a 5.000 couriers y
  cientos de pedidos pendientes por ronda de despacho). `config/equipetrol.json` tiene 668 nodos y
  3 couriers: a esa escala ningún kernel le gana a la CPU.

## Decisión

- `config/ciudad.json`, con el **mismo esquema** que Equipetrol (sin campos nuevos): 3.741
  intersecciones reales de Santa Cruz de la Sierra (bbox de ~4,9 × 5,4 km que contiene Equipetrol),
  150 restaurantes reales de OSM, 2.000 couriers y una tasa de pedidos que deja cientos pendientes.
- Se versiona el `.osm` **filtrado a las etiquetas que usa el extractor (`tools/osm_strip_tags.py`), sin
  metadatos de editores ni datos de contacto** (1,7 MB) junto con la consulta y el
  comando exactos (`docs/ciudad.md`), para que el JSON se pueda regenerar sin red e igual byte a byte.
- `tools/osm_extract.py` se parametriza por línea de comandos y se hace escalar (parseo incremental,
  agrupamiento por rejilla), sin cambiar su salida para Equipetrol.
- El fondo de la ventana se dibuja a partir de los vectores del propio grafo
  (`tools/render_background.py`, solo biblioteca estándar).
- Datos bajo ODbL, con atribución, declarado en `data/NOTICE.md`.

## Alternativas descartadas

- **Grafo sintético en rejilla como escenario principal.** Fácil de escalar, pero el enunciado pide una
  ciudad de OpenStreetMap y una rejilla no tiene calles de un solo sentido ni la irregularidad real.
  Un generador sintético puede servir más adelante para barrer tamaños en el modo bench.
- **Versionar solo el JSON derivado.** Más liviano, pero el resultado dependería de una descarga futura
  de OSM, que cambia con el tiempo; no sería reproducible.
- **Fondo con teselas de OpenStreetMap.** Más vistoso, pero la *Tile Usage Policy* no permite descargas
  masivas, y el fondo dependería de un servicio externo en vez de salir de los propios datos.
