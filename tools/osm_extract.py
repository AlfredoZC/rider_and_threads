#!/usr/bin/env python3
"""
Genera config/equipetrol.json a partir de un export de OpenStreetMap (.osm).

Herramienta de apoyo (NO forma parte del programa C++). Solo usa la biblioteca
estándar de Python; Pillow es opcional y se usa únicamente para --preview.

Uso:
    python3 tools/osm_extract.py data/equipetrol.osm config/equipetrol.json \
        --image ../data/equipetrol.png --preview docs/graph_preview.png

Pasos:
  1. Lee los bounds y las vías (ways) del .osm.
  2. Se queda con las calles seleccionadas: avenidas principales + calles que
     cruzan Av. San Martín (lista RESIDENTIAL_NAMES).
  3. Recorta todo a los bounds de la imagen.
  4. Marca como "nodo clave" cada punto compartido por dos vías (intersección)
     y cada extremo de vía.
  5. Parte cada vía en tramos entre nodos clave y, dentro de cada tramo,
     conserva solo los puntos de forma necesarios (Douglas-Peucker) para que
     las curvas sigan la calle sin cruzar manzanas.
  6. Fusiona los nodos clave muy cercanos (las dos calzadas de una avenida
     doble vía, las rotondas) en un solo nodo.
  7. Se queda con la mayor componente fuertemente conexa: desde cualquier nodo
     se puede llegar a cualquier otro respetando los sentidos.
  8. Elige restaurantes reales de OSM repartidos por zonas (entre anillos) y a
     ambos lados de San Martín, y escribe la configuración completa.
"""
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

# --- Selección de calles -------------------------------------------------------

# Avenidas: se modelan en ambos sentidos. En OSM son doble vía (dos calzadas
# de un solo sentido, a ~20-30 m); al fusionar las calzadas en un nodo común,
# el par se representa como una calle bidireccional.
MAIN_HIGHWAYS = {"trunk", "primary", "secondary", "tertiary"}

# Enlaces y retornos entre calzadas (p. ej. los cruces de Av. La Salle sobre el
# canal). Son cortos y casi siempre de un solo sentido: se respeta oneway.
LINK_HIGHWAYS = {"trunk_link", "primary_link", "secondary_link", "tertiary_link"}

# Calles residenciales que cruzan o acompañan Av. San Martín. En estas sí se
# respeta oneway=yes de OSM (calles de un solo sentido reales).
RESIDENTIAL_NAMES = {
    "Calle Tucumán", "Calle Córdoba", "Calle La Plata", "Calle Hugo Wast",
    "Calle Nicolás Ortiz", "Calle Los Claveles", "Calle Las Dalias",
    "Calle Los Jazmines", "Calle Los Lirios", "Calle Las Azucenas",
    "Calle Las Begonias", "Calle Pablo Sanz", "Calle Doctor Fermín Peralta",
    "Calle Doctor Leonardo Nava", "Calle Doctor Alejandro Ramírez",
    "Calle Alfonsina Storni", "Calle Lugones", "Calle Sarmiento",
    "Calle Manuel Güemes", "Calle Ricardo Mujía", "Calle Claudio Peñaranda",
    "Calle Jaime Mendoza", "Calle Gregorio Reynolds", "Calle Franz Tamayo",
}

MERGE_RADIUS_M = 32.0       # nodos clave más cercanos que esto se fusionan
SIMPLIFY_TOL_M = 6.0        # tolerancia de Douglas-Peucker (en metros)
MIN_SHAPE_SPACING_M = 25.0  # no dejar puntos de forma más juntos que esto
MAX_DEVIATION_M = 20.0      # aviso si un tramo se aleja más que esto de la calle real

# Plaza 24 de Septiembre: centro de los anillos de Santa Cruz
CITY_CENTER = (-17.78366, -63.18212)


# --- Geometría ------------------------------------------------------------------

def to_xy(lat, lon, ref_lat):
    """Proyección local equirectangular en metros (suficiente para ~3 km)."""
    return (lon * 111320.0 * math.cos(math.radians(ref_lat)), lat * 110574.0)


def dist_m(a, b, ref_lat):
    ax, ay = to_xy(a[0], a[1], ref_lat)
    bx, by = to_xy(b[0], b[1], ref_lat)
    return math.hypot(ax - bx, ay - by)


def point_segment_dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def douglas_peucker(idx, xy, tol):
    """Devuelve los índices de 'idx' que hay que conservar."""
    if len(idx) <= 2:
        return list(idx)
    a, b = xy[idx[0]], xy[idx[-1]]
    worst, worst_d = None, -1.0
    for k in range(1, len(idx) - 1):
        d = point_segment_dist(xy[idx[k]], a, b)
        if d > worst_d:
            worst, worst_d = k, d
    if worst_d <= tol:
        return [idx[0], idx[-1]]
    left = douglas_peucker(idx[:worst + 1], xy, tol)
    right = douglas_peucker(idx[worst:], xy, tol)
    return left[:-1] + right


# --- Lectura del .osm ------------------------------------------------------------

def load_osm(path):
    root = ET.parse(path).getroot()
    b = root.find("bounds")
    if b is None:
        sys.exit("El .osm no tiene <bounds>")
    bounds = {"north": float(b.get("maxlat")), "south": float(b.get("minlat")),
              "west": float(b.get("minlon")), "east": float(b.get("maxlon"))}
    nodes, amenities = {}, []
    for n in root.iter("node"):
        nodes[n.get("id")] = (float(n.get("lat")), float(n.get("lon")))
        tags = {t.get("k"): t.get("v") for t in n.iter("tag")}
        if tags.get("amenity") in ("restaurant", "fast_food") and tags.get("name"):
            amenities.append((tags["name"], nodes[n.get("id")]))
    ways = []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        ways.append((tags, [nd.get("ref") for nd in w.iter("nd")]))
    return bounds, nodes, ways, amenities


def select_ways(ways, all_streets=False):
    selected = []
    for tags, refs in ways:
        hw = tags.get("highway")
        if hw in MAIN_HIGHWAYS:
            selected.append((tags, refs, False))           # avenida: bidireccional
        elif hw in LINK_HIGHWAYS or (hw in ("residential", "unclassified", "living_street") and (
                all_streets or tags.get("name") in RESIDENTIAL_NAMES)):
            ow = tags.get("oneway")
            if ow == "-1":
                refs = list(reversed(refs))
            selected.append((tags, refs, ow in ("yes", "true", "1", "-1")))
    return selected


def clip_runs(refs, nodes, bounds):
    """Partes consecutivas de la vía que quedan dentro de los bounds."""
    def inside(r):
        lat, lon = nodes[r]
        return (bounds["south"] <= lat <= bounds["north"]
                and bounds["west"] <= lon <= bounds["east"])
    runs, cur = [], []
    for r in refs:
        if r in nodes and inside(r):
            cur.append(r)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    return runs


# --- Construcción del grafo ---------------------------------------------------------

def build_graph(selected, nodes, bounds):
    ref_lat = (bounds["north"] + bounds["south"]) / 2
    runs = []                                   # (lista de refs, oneWay, nombre, es_avenida)
    usage = defaultdict(int)
    for tags, refs, one_way in selected:
        is_main = tags.get("highway") in MAIN_HIGHWAYS
        for run in clip_runs(refs, nodes, bounds):
            runs.append((run, one_way, tags.get("name", ""), is_main))
            for r in set(run):
                usage[r] += 1

    key = {r for r, c in usage.items() if c >= 2}
    for run, _, _, _ in runs:
        key.add(run[0])
        key.add(run[-1])

    # Para cada nodo clave: en qué vías está y con qué nombres de calle residencial
    runs_of, res_names_of = defaultdict(set), defaultdict(set)
    for i, (run, _, name, is_main) in enumerate(runs):
        for r in run:
            if r in key:
                runs_of[r].add(i)
                if not is_main and name:
                    res_names_of[r].add(name)

    def same_street(r, members):
        """Dos puntos de la MISMA calle nunca son el mismo cruce (p. ej. una calle en U)."""
        return any(runs_of[r] & runs_of[m] or res_names_of[r] & res_names_of[m] for m in members)

    # 6. Fusionar nodos clave cercanos (agrupamiento voraz alrededor de semillas)
    # Primero los más conectados; el id desempata para que el resultado sea
    # idéntico en cada corrida (el orden de un set de Python no es estable)
    key_list = sorted(key, key=lambda r: (-usage[r], int(r)))
    cluster_of, centers = {}, []
    for r in key_list:
        p = nodes[r]
        best, best_d = None, MERGE_RADIUS_M
        for ci, (c, members) in enumerate(centers):
            d = dist_m(p, c, ref_lat)
            if d < best_d and not same_street(r, members):
                best, best_d = ci, d
        if best is None:
            centers.append((p, [r]))
            cluster_of[r] = len(centers) - 1
        else:
            centers[best][1].append(r)
            cluster_of[r] = best
    # centro de cada grupo = promedio de sus miembros
    cluster_pos = []
    for _, members in centers:
        lat = sum(nodes[m][0] for m in members) / len(members)
        lon = sum(nodes[m][1] for m in members) / len(members)
        cluster_pos.append((lat, lon))

    # 5. Partir cada vía en tramos entre nodos clave y simplificar la forma
    vertex_pos = list(cluster_pos)              # vértices 0..k-1 = grupos
    edges = {}                                  # (a, b) -> oneWay
    xy_cache = {}

    def xy(r):
        if r not in xy_cache:
            xy_cache[r] = to_xy(nodes[r][0], nodes[r][1], ref_lat)
        return xy_cache[r]

    # Las avenidas doble vía generan dos tramos paralelos entre los mismos dos
    # cruces (una calzada por sentido). Se dibuja solo uno: la avenida ya es
    # bidireccional en el modelo.
    done_pairs = set()
    deviations = []                             # (desvío máximo en m, calle)
    for run, one_way, name, is_main in runs:
        cuts = [i for i, r in enumerate(run) if r in key]
        for s, e in zip(cuts, cuts[1:]):
            chain = run[s:e + 1]
            a, b = cluster_of[chain[0]], cluster_of[chain[-1]]
            if is_main:
                pair = frozenset((a, b))
                if pair in done_pairs:
                    continue
                done_pairs.add(pair)
            pts = {i: xy(r) for i, r in enumerate(chain)}
            keep = douglas_peucker(list(range(len(chain))), pts, SIMPLIFY_TOL_M)
            path = [a]
            last = cluster_pos[a]
            for k in keep[1:-1]:
                p = nodes[chain[k]]
                if dist_m(p, last, ref_lat) < MIN_SHAPE_SPACING_M:
                    continue
                if any(dist_m(p, cluster_pos[c], ref_lat) < MERGE_RADIUS_M / 2 for c in (a, b)):
                    continue                    # punto de forma dentro de un cruce
                vertex_pos.append(p)
                path.append(len(vertex_pos) - 1)
                last = p
            path.append(b)
            for u, v in zip(path, path[1:]):
                if u == v:
                    continue
                add_edge(edges, u, v, one_way)

            # Control de calidad: cuánto se aleja el tramo modelado de la calle real
            poly = [to_xy(vertex_pos[v][0], vertex_pos[v][1], ref_lat) for v in path]
            dev = max(min(point_segment_dist(xy(r), poly[i], poly[i + 1]) for i in range(len(poly) - 1))
                      if len(poly) > 1 else 0.0 for r in chain)
            deviations.append((dev, name or "(avenida sin nombre)", path))
    return vertex_pos, edges, deviations


def add_edge(edges, u, v, one_way):
    """Une sentidos: si existe la arista opuesta, la calle queda bidireccional."""
    if (u, v) in edges or (v, u) in edges:
        if (v, u) in edges:
            edges[(v, u)] = False
        elif not one_way:
            edges[(u, v)] = False
        return
    edges[(u, v)] = one_way


def edge_kind(edges, a, b):
    """'both' si a<->b, 'forward' si solo a->b, 'backward' si solo b->a."""
    if edges.get((a, b)) is False or edges.get((b, a)) is False:
        return "both"
    if (a, b) in edges:
        return "forward"
    if (b, a) in edges:
        return "backward"
    return None


def contract_straight(vertex_pos, edges, bounds):
    """Quita vértices de grado 2 que están en línea recta entre sus vecinos.

    Vienen de que OSM corta una misma calle en varias vías: no son cruces ni
    curvas, solo inflan el grafo. Se conservan si quitarlos desviaría la calle
    más de SIMPLIFY_TOL_M (para no cortar manzanas).
    """
    ref_lat = (bounds["north"] + bounds["south"]) / 2
    xy = [to_xy(p[0], p[1], ref_lat) for p in vertex_pos]
    changed = True
    while changed:
        changed = False
        nbrs = defaultdict(set)
        for (u, v) in edges:
            nbrs[u].add(v)
            nbrs[v].add(u)
        for v, ns in list(nbrs.items()):
            if len(ns) != 2:
                continue
            u, w = sorted(ns)
            if (u, w) in edges or (w, u) in edges:
                continue
            if point_segment_dist(xy[v], xy[u], xy[w]) > SIMPLIFY_TOL_M:
                continue
            # Solo si la calle es del mismo tipo a ambos lados:
            # u<->v<->w (doble sentido), u->v->w o w->v->u (un sentido)
            k1, k2 = edge_kind(edges, u, v), edge_kind(edges, v, w)
            if k1 != k2 or k1 is None:
                continue
            for e in ((u, v), (v, u), (v, w), (w, v)):
                edges.pop(e, None)
            if k1 == "both":
                edges[(u, w)] = False
            elif k1 == "forward":
                edges[(u, w)] = True
            else:
                edges[(w, u)] = True
            changed = True
            break
    return edges


def largest_scc(n, edges):
    """Tarjan iterativo: mayor componente fuertemente conexa."""
    adj = defaultdict(list)
    for (u, v), ow in edges.items():
        adj[u].append(v)
        if not ow:
            adj[v].append(u)
    index, low, on, stack, comps = {}, {}, set(), [], []
    counter = [0]
    for s in range(n):
        if s in index:
            continue
        work = [(s, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter[0]
                counter[0] += 1
                stack.append(v)
                on.add(v)
            recurse = False
            for j in range(i, len(adj[v])):
                w = adj[v][j]
                if w not in index:
                    work.append((v, j + 1))
                    work.append((w, 0))
                    recurse = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if recurse:
                continue
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                comps.append(comp)
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return set(max(comps, key=len))


# --- Restaurantes ----------------------------------------------------------------------

RESTAURANT_COUNT = 6
RESTAURANT_SNAP_M = 60.0    # el restaurante real debe estar a menos de esto de su nodo


def ring_zone(p, ref_lat):
    """Zona según la distancia al centro: entre 2do-3ro, cerca del 3ro, entre 3ro-4to."""
    d = dist_m(p, CITY_CENTER, ref_lat)
    return "2do-3ro" if d < 2200 else ("3ro" if d < 2800 else "3ro-4to")


def pick_restaurants(vertex_pos, keep, amenities, bounds):
    """Elige restaurantes reales de OSM lo más separados posible entre sí.

    Candidatos: restaurantes con nombre a menos de RESTAURANT_SNAP_M de un nodo
    del grafo. Selección voraz "el más lejano a los ya elegidos" (k-center),
    que reparte los restaurantes por toda la zona en vez de amontonarlos.
    """
    ref_lat = (bounds["north"] + bounds["south"]) / 2
    candidates = {}                               # nodo -> nombre (uno por nodo)
    for name, p in sorted(amenities):
        if not (bounds["south"] <= p[0] <= bounds["north"] and bounds["west"] <= p[1] <= bounds["east"]):
            continue
        v = min(keep, key=lambda k: dist_m(p, vertex_pos[k], ref_lat))
        if dist_m(p, vertex_pos[v], ref_lat) <= RESTAURANT_SNAP_M:
            candidates.setdefault(v, name)
    if len(candidates) < RESTAURANT_COUNT:
        sys.exit(f"Solo hay {len(candidates)} restaurantes cerca del grafo")

    # Empezar por el más cercano al Cuarto Anillo (el más alejado del centro)
    nodes_c = sorted(candidates)
    first = max(nodes_c, key=lambda v: dist_m(vertex_pos[v], CITY_CENTER, ref_lat))
    chosen = [first]
    while len(chosen) < RESTAURANT_COUNT:
        nxt = max((v for v in nodes_c if v not in chosen),
                  key=lambda v: min(dist_m(vertex_pos[v], vertex_pos[c], ref_lat) for c in chosen))
        chosen.append(nxt)
    chosen.sort(key=lambda v: -dist_m(vertex_pos[v], CITY_CENTER, ref_lat))   # de 4to a 2do
    return [(candidates[v], v) for v in chosen]


# --- Salida ------------------------------------------------------------------------------------

def write_preview(path, image, bounds, vertex_pos, keep, edges, restaurants, ids):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow no está instalado: se omite --preview", file=sys.stderr)
        return
    im = Image.open(image).convert("RGB")
    d = ImageDraw.Draw(im)

    def px(v):
        lat, lon = vertex_pos[v]
        return ((lon - bounds["west"]) / (bounds["east"] - bounds["west"]) * im.width,
                (bounds["north"] - lat) / (bounds["north"] - bounds["south"]) * im.height)

    for (u, v), ow in edges.items():
        if u in keep and v in keep:
            d.line([px(u), px(v)], fill=(220, 30, 30) if ow else (30, 70, 220), width=3)
            if ow:  # punta de flecha en el sentido permitido, a 2/3 del tramo
                (x1, y1), (x2, y2) = px(u), px(v)
                mx, my = (x1 + 2 * x2) / 3, (y1 + 2 * y2) / 3
                ang = math.atan2(y2 - y1, x2 - x1)
                for delta in (0.5, -0.5):
                    d.line([(mx, my), (mx - 10 * math.cos(ang + delta),
                                       my - 10 * math.sin(ang + delta))],
                           fill=(220, 30, 30), width=3)
    for v in keep:
        x, y = px(v)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255), outline=(0, 0, 0))
    for _, v in restaurants:
        x, y = px(v)
        d.rectangle([x - 8, y - 8, x + 8, y + 8], fill=(0, 0, 0))
        d.text((x + 10, y - 6), ids[v], fill=(0, 0, 0))
    im.save(path)
    print(f"Vista previa: {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("osm", help="export de OpenStreetMap (.osm)")
    ap.add_argument("out", help="config JSON a generar")
    ap.add_argument("--image", default="../data/equipetrol.png",
                    help="ruta de la imagen RELATIVA al archivo de config")
    ap.add_argument("--preview", help="PNG con el grafo dibujado sobre el mapa (requiere Pillow)")
    ap.add_argument("--all-streets", action="store_true",
                    help="incluir TODAS las calles residenciales, no solo RESIDENTIAL_NAMES")
    args = ap.parse_args()

    bounds, nodes, ways, amenities = load_osm(args.osm)
    selected = select_ways(ways, args.all_streets)
    vertex_pos, edges, deviations = build_graph(selected, nodes, bounds)
    edges = contract_straight(vertex_pos, edges, bounds)
    keep = largest_scc(len(vertex_pos), edges)
    keep &= {x for e in edges for x in e}       # sin vértices aislados

    restaurants = pick_restaurants(vertex_pos, keep, amenities, bounds)
    # Extremos reales de Av. San Martín: el courier arranca cerca de su punto medio
    sm_line = ((-17.7571245, -63.2011494), (-17.7724124, -63.1902843))

    # Ids compactos n0..nK en orden geográfico (norte -> sur), estables entre corridas
    order = sorted(keep, key=lambda v: (-vertex_pos[v][0], vertex_pos[v][1]))
    ids = {v: f"n{i}" for i, v in enumerate(order)}

    ref_lat = (bounds["north"] + bounds["south"]) / 2
    mid = ((sm_line[0][0] + sm_line[1][0]) / 2, (sm_line[0][1] + sm_line[1][1]) / 2)
    start = min(keep, key=lambda v: dist_m(vertex_pos[v], mid, ref_lat))

    streets = []
    for (u, v), ow in sorted(edges.items(), key=lambda e: (ids.get(e[0][0], ""), ids.get(e[0][1], ""))):
        if u in keep and v in keep:
            streets.append({"id": f"s{len(streets)}", "from": ids[u], "to": ids[v], "oneWay": ow})

    config = {
        "map": {
            "image": args.image,
            "attribution": "© OpenStreetMap contributors",
            "bounds": bounds,
        },
        "nodes": [{"id": ids[v], "lat": round(vertex_pos[v][0], 7), "lon": round(vertex_pos[v][1], 7)}
                  for v in order],
        "streets": streets,
        "restaurants": [
            {"id": f"r{i}", "name": name, "node": ids[v],
             "pickupSlots": 2 if i % 2 == 0 else 1,
             "prepTimeMs": [60000, 240000] if i % 2 == 0 else [120000, 300000]}
            for i, (name, v) in enumerate(restaurants)
        ],
        "fleet": {"couriers": 8, "bagCapacity": 3, "speedKmh": 30, "startNode": ids[start]},
        "orders": {"meanIntervalMs": 20000, "burstMax": 12, "maxPending": 50, "seed": 42},
        "dispatch": {"quoteTimeoutMs": 200, "acceptTimeoutMs": 600000},
        "incidents": {"breakdownProbability": 0.02},
        "simulation": {"durationS": 3600, "timeScale": 60},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")

    n_one = sum(1 for s in streets if s["oneWay"])
    print(f"nodos: {len(order)}  calles: {len(streets)} ({n_one} de un sentido)  "
          f"restaurantes: {len(restaurants)}  descartados fuera de la componente: "
          f"{len(vertex_pos) - len(keep)}", file=sys.stderr)
    for i, (name, v) in enumerate(restaurants):
        print(f"  r{i}: {name} -> {ids[v]} ({ring_zone(vertex_pos[v], ref_lat)})", file=sys.stderr)
    print(f"  startNode: {ids[start]}", file=sys.stderr)

    # Un tramo que se aleja mucho de la calle real atraviesa manzanas (defecto del brief)
    bad = sorted((d for d in deviations if d[0] > MAX_DEVIATION_M and set(d[2]) <= keep),
                 key=lambda d: -d[0])
    print(f"tramos que se alejan más de {MAX_DEVIATION_M:.0f} m de la calle real: {len(bad)}",
          file=sys.stderr)
    for dev, name, path in bad[:15]:
        print(f"  {dev:5.1f} m  {name}  {ids[path[0]]} -> {ids[path[-1]]}", file=sys.stderr)

    if args.preview:
        image_path = args.image
        import os
        image_path = os.path.normpath(os.path.join(os.path.dirname(args.out), args.image))
        write_preview(args.preview, image_path, bounds, vertex_pos, keep, edges, restaurants, ids)


if __name__ == "__main__":
    main()
