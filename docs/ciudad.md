# Configuración `ciudad` (escala para GPU)

`config/ciudad.json` es la ciudad grande del segundo parcial: un grafo de ~3.7 mil intersecciones de
Santa Cruz de la Sierra (Equipetrol y alrededores), 2.000 couriers y cientos de pedidos pendientes por ronda
de despacho. Tiene el mismo esquema que `config/equipetrol.json` (no hay campos nuevos).

Datos © OpenStreetMap contributors, ODbL (ver `data/NOTICE.md`).

## Cómo se reproduce

1. Descargar el extracto (Overpass API, un solo POST; el `User-Agent` es genérico):

```bash
cat > q_ciudad.txt <<'EOF'
[out:xml][timeout:300];
(
  way["highway"~"^(trunk|primary|secondary|tertiary|trunk_link|primary_link|secondary_link|tertiary_link|residential|unclassified|living_street)$"](-17.799,-63.212,-17.750,-63.165);
  node["amenity"~"^(restaurant|fast_food)$"](-17.799,-63.212,-17.750,-63.165);
);
(._;>;);
out;
EOF
curl -A "osm-extract-student-project/1.0" --data-urlencode data@q_ciudad.txt \
     https://overpass-api.de/api/interpreter -o data/ciudad.osm
```

`out;` devuelve el cuerpo sin metadatos de editores (sin `user`, `uid`, `changeset`, `timestamp`), pero
conserva todas las etiquetas de OSM, incluidas las de contacto (`email`, `phone`...). Por eso el crudo
**no se versiona**: se reduce a lo que usa el extractor (way: `highway`, `oneway`, `name`; nodo:
`amenity` restaurant/fast_food y `name`):

```bash
python3 tools/osm_strip_tags.py crudo.osm data/ciudad.osm
```

Overpass no emite `<bounds>`, por eso el extractor recibe `--bbox`. El archivo versionado se pidió con
`osm_base` 2026-11-20T03:36:58Z.

2. Generar el grafo:

```bash
python3 tools/osm_extract.py data/ciudad.osm config/ciudad.json --image ../data/ciudad.png \
    --bbox -17.799 -63.212 -17.750 -63.165 --all-streets --restaurants 150 \
    --overlay tools/overlays/ciudad.json
```

3. Dibujar el fondo a partir de los vectores del grafo (sin teselas):

```bash
python3 tools/render_background.py config/ciudad.json data/ciudad.png
```

## Qué se versiona y por qué

Se versiona el `.osm` reducido por `tools/osm_strip_tags.py` (1.7 MB, sin metadatos de editores ni datos
de contacto): es pequeño, hace el resultado
reproducible aunque OSM cambie, y no requiere red para regenerar el JSON. El crudo sin reducir
(con `email`/`phone` de terceros) no se guarda. Todo queda muy por debajo de los 50 MB.

## Valores elegidos

| Campo | Valor | Motivo |
|-------|-------|--------|
| bbox (S, O, N, E) | −17.799, −63.212, −17.750, −63.165 (~4.9 × 5.4 km) | contiene la caja de Equipetrol y la Plaza 24 de Septiembre; da 3.741 nodos tras simplificar |
| `fleet.couriers` | 2000 | mínimo del enunciado |
| `fleet.bagCapacity`, `speedKmh` | 3, 30 | iguales a Equipetrol |
| `orders.meanIntervalMs` | 2000 | con `timeScale` 60 llega ~1 pedido cada 33 ms reales: más que lo que despacha la CPU, así la cola crece |
| `orders.burstMax` | 30 | ráfagas (1 de cada 10 llegadas) de hasta 30 pedidos |
| `orders.maxPending` | 600 | tope de la cola: cientos de pendientes sin crecer sin límite |
| `orders.seed` | 42 | semilla fija |
| `urgentProbability`, `agingMs` | 0.3, 300000 | Variante A, igual que Equipetrol |
| `kitchen.pauseEveryMs`, `pauseMs` | 900000, 240000 | Variante B, igual que Equipetrol |
| `fleet.breakAfterDeliveries`, `breakMs` | 2, 600000 | Variante C (descansos), igual que Equipetrol |
| `reinforcements.max/threshold/idleMs` | 100, 6, 300000 | Variante D: se mantienen `min(100, esperando / 6)` refuerzos activos; con `maxPending` 600 los esperando topan en 600, así que el máximo de 100 se alcanza justo con la cola llena. (Con `threshold` 300 nunca pasaba de 2.) Equipetrol la deja en 0 |
| restaurantes | 150 de 293 candidatos reales de OSM (`amenity=restaurant|fast_food` con nombre, a ≤ 60 m de un nodo del grafo) | elegidos con el criterio de "el más lejano" ya existente, así quedan repartidos por la ciudad |
| `simulation` | 3600 s a ×60 | igual que Equipetrol |

Los parámetros de despacho/variantes viven en `tools/overlays/ciudad.json`, que `osm_extract.py --overlay`
fusiona sobre el config generado.
