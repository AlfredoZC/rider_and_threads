#!/usr/bin/env python3
"""Genera un config JSON de ciudad (grafo de calles + restaurantes) desde un export de OpenStreetMap.

Los valores por defecto reproducen config/equipetrol.json. Ejemplos:
  python3 tools/osm_extract.py data/equipetrol.osm config/equipetrol.json --all-streets
  python3 tools/osm_extract.py data/ciudad.osm config/ciudad.json --image ../data/ciudad.png \
      --bbox S W N E --all-streets --restaurants 100 --couriers 2000

Datos (c) OpenStreetMap contributors, licencia ODbL (ver data/NOTICE.md).
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

MAIN_HIGHWAYS = {"trunk", "primary", "secondary", "tertiary"}

LINK_HIGHWAYS = {"trunk_link", "primary_link", "secondary_link", "tertiary_link"}

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

MERGE_RADIUS_M = 32.0
SIMPLIFY_TOL_M = 6.0
MIN_SHAPE_SPACING_M = 25.0
MAX_DEVIATION_M = 20.0

CITY_CENTER = (-17.78366, -63.18212)
START_LINE = ((-17.7571245, -63.2011494), (-17.7724124, -63.1902843))

def to_xy(lat, lon, ref_lat):
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

def load_osm(path, bbox=None):
    """Lee el .osm en streaming (iterparse): cada elemento se libera al procesarlo y se vacia la
    raiz, asi que la memoria solo crece con lo que se conserva (nodos y vias), no con el XML.

    bbox = (sur, oeste, norte, este) reemplaza al <bounds> del archivo; los exports de
    Overpass no traen <bounds>.
    """
    bounds = None
    nodes, amenities, ways = {}, [], []
    root = None
    for ev, el in ET.iterparse(path, events=("start", "end")):
        if ev == "start":
            if root is None:
                root = el
            continue
        if el.tag == "bounds":
            bounds = {"north": float(el.get("maxlat")), "south": float(el.get("minlat")),
                      "west": float(el.get("minlon")), "east": float(el.get("maxlon"))}
        elif el.tag == "node":
            nid = el.get("id")
            nodes[nid] = (float(el.get("lat")), float(el.get("lon")))
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            if tags.get("amenity") in ("restaurant", "fast_food") and tags.get("name"):
                amenities.append((tags["name"], nodes[nid]))
            el.clear()
            root.clear()
        elif el.tag == "way":
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            ways.append((tags, [nd.get("ref") for nd in el.findall("nd")]))
            el.clear()
            root.clear()
    if bbox is not None:
        south, west, north, east = bbox
        bounds = {"north": north, "south": south, "west": west, "east": east}
    if bounds is None:
        sys.exit("El .osm no tiene <bounds>: pasa --bbox SUR OESTE NORTE ESTE")
    return bounds, nodes, ways, amenities

def select_ways(ways, all_streets=False):
    selected = []
    for tags, refs in ways:
        hw = tags.get("highway")
        if hw in MAIN_HIGHWAYS:
            selected.append((tags, refs, False))
        elif hw in LINK_HIGHWAYS or (hw in ("residential", "unclassified", "living_street") and (
                all_streets or tags.get("name") in RESIDENTIAL_NAMES)):
            ow = tags.get("oneway")
            if ow == "-1":
                refs = list(reversed(refs))
            selected.append((tags, refs, ow in ("yes", "true", "1", "-1")))
    return selected

def clip_runs(refs, nodes, bounds):
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

def build_graph(selected, nodes, bounds, merge_radius=MERGE_RADIUS_M, simplify_tol=SIMPLIFY_TOL_M,
                min_shape_spacing=MIN_SHAPE_SPACING_M):
    ref_lat = (bounds["north"] + bounds["south"]) / 2
    runs = []
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

    runs_of, res_names_of = defaultdict(set), defaultdict(set)
    for i, (run, _, name, is_main) in enumerate(runs):
        for r in run:
            if r in key:
                runs_of[r].add(i)
                if not is_main and name:
                    res_names_of[r].add(name)

    def same_street(r, members):
        return any(runs_of[r] & runs_of[m] or res_names_of[r] & res_names_of[m] for m in members)

    key_list = sorted(key, key=lambda r: (-usage[r], int(r)))
    cluster_of, centers = {}, []
    # Rejilla espacial con celdas del tamaño del radio: cada nodo solo se compara con los
    # centros de las 9 celdas vecinas (antes, con todos los centros: O(n^2)). Mismo resultado:
    # gana el centro mas cercano y, en empate exacto, el de menor indice.
    grid = defaultdict(list)
    centers_xy = []
    for r in key_list:
        p = nodes[r]
        x, y = to_xy(p[0], p[1], ref_lat)
        cx, cy = math.floor(x / merge_radius), math.floor(y / merge_radius)
        best, best_d = None, merge_radius
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for ci in grid.get((gx, gy), ()):
                    c = centers_xy[ci]
                    d = math.hypot(x - c[0], y - c[1])
                    if (d < best_d or (best is not None and d == best_d and ci < best))                             and not same_street(r, centers[ci][1]):
                        best, best_d = ci, d
        if best is None:
            centers.append((p, [r]))
            centers_xy.append((x, y))
            grid[(cx, cy)].append(len(centers) - 1)
            cluster_of[r] = len(centers) - 1
        else:
            centers[best][1].append(r)
            cluster_of[r] = best
    cluster_pos = []
    for _, members in centers:
        lat = sum(nodes[m][0] for m in members) / len(members)
        lon = sum(nodes[m][1] for m in members) / len(members)
        cluster_pos.append((lat, lon))

    vertex_pos = list(cluster_pos)
    edges = {}
    xy_cache = {}

    def xy(r):
        if r not in xy_cache:
            xy_cache[r] = to_xy(nodes[r][0], nodes[r][1], ref_lat)
        return xy_cache[r]

    done_pairs = set()
    deviations = []
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
            keep = douglas_peucker(list(range(len(chain))), pts, simplify_tol)
            path = [a]
            last = cluster_pos[a]
            for k in keep[1:-1]:
                p = nodes[chain[k]]
                if dist_m(p, last, ref_lat) < min_shape_spacing:
                    continue
                if any(dist_m(p, cluster_pos[c], ref_lat) < merge_radius / 2 for c in (a, b)):
                    continue
                vertex_pos.append(p)
                path.append(len(vertex_pos) - 1)
                last = p
            path.append(b)
            for u, v in zip(path, path[1:]):
                if u == v:
                    continue
                add_edge(edges, u, v, one_way)

            poly = [to_xy(vertex_pos[v][0], vertex_pos[v][1], ref_lat) for v in path]
            dev = max(min(point_segment_dist(xy(r), poly[i], poly[i + 1]) for i in range(len(poly) - 1))
                      if len(poly) > 1 else 0.0 for r in chain)
            deviations.append((dev, name or "(avenida sin nombre)", path))
    return vertex_pos, edges, deviations

def add_edge(edges, u, v, one_way):
    if (u, v) in edges or (v, u) in edges:
        if (v, u) in edges:
            edges[(v, u)] = False
        elif not one_way:
            edges[(u, v)] = False
        return
    edges[(u, v)] = one_way

def edge_kind(edges, a, b):
    if edges.get((a, b)) is False or edges.get((b, a)) is False:
        return "both"
    if (a, b) in edges:
        return "forward"
    if (b, a) in edges:
        return "backward"
    return None

def contract_straight(vertex_pos, edges, bounds, simplify_tol=SIMPLIFY_TOL_M):
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
            if point_segment_dist(xy[v], xy[u], xy[w]) > simplify_tol:
                continue
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

RESTAURANT_COUNT = 6
RESTAURANT_SNAP_M = 60.0

def ring_zone(p, ref_lat, center=CITY_CENTER):
    d = dist_m(p, center, ref_lat)
    return "2do-3ro" if d < 2200 else ("3ro" if d < 2800 else "3ro-4to")

def pick_restaurants(vertex_pos, keep, amenities, bounds, count=RESTAURANT_COUNT,
                     snap_m=RESTAURANT_SNAP_M, center=CITY_CENTER):
    ref_lat = (bounds["north"] + bounds["south"]) / 2
    # rejilla de los nodos del grafo para hallar el mas cercano a cada restaurante sin
    # recorrerlos todos
    grid = defaultdict(list)
    for k in sorted(keep):
        x, y = to_xy(vertex_pos[k][0], vertex_pos[k][1], ref_lat)
        grid[(math.floor(x / snap_m), math.floor(y / snap_m))].append((k, x, y))
    candidates = {}
    for name, p in sorted(amenities):
        if not (bounds["south"] <= p[0] <= bounds["north"] and bounds["west"] <= p[1] <= bounds["east"]):
            continue
        px, py = to_xy(p[0], p[1], ref_lat)
        cx, cy = math.floor(px / snap_m), math.floor(py / snap_m)
        v, vd = None, snap_m
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for k, x, y in grid.get((gx, gy), ()):
                    d = math.hypot(px - x, py - y)
                    if d < vd or (v is not None and d == vd and k < v):
                        v, vd = k, d
        if v is not None:
            candidates.setdefault(v, name)
    if len(candidates) < count:
        sys.exit(f"Solo hay {len(candidates)} restaurantes cerca del grafo (se pidieron {count})")

    nodes_c = sorted(candidates)
    first = max(nodes_c, key=lambda v: dist_m(vertex_pos[v], center, ref_lat))
    chosen = [first]
    taken = {first}
    # distancia minima a lo ya elegido, actualizada de forma incremental (antes se recalculaba
    # contra todos los elegidos en cada paso: O(count^2 * candidatos))
    mind = {v: dist_m(vertex_pos[v], vertex_pos[first], ref_lat) for v in nodes_c}
    while len(chosen) < count:
        nxt, best = None, -1.0
        for v in nodes_c:
            if v in taken:
                continue
            if mind[v] > best:
                nxt, best = v, mind[v]
        chosen.append(nxt)
        taken.add(nxt)
        for v in nodes_c:
            mind[v] = min(mind[v], dist_m(vertex_pos[v], vertex_pos[nxt], ref_lat))
    chosen.sort(key=lambda v: -dist_m(vertex_pos[v], center, ref_lat))
    return [(candidates[v], v) for v in chosen]

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
            if ow:
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

def validate_args(ap, args):
    def finite(name, values):
        for v in values:
            if not math.isfinite(v):
                ap.error(f"{name}: el valor {v} no es un numero finito")

    def check_point(name, lat, lon):
        if not (-90.0 <= lat <= 90.0):
            ap.error(f"{name}: la latitud {lat} debe estar en [-90, 90]")
        if not (-180.0 <= lon <= 180.0):
            ap.error(f"{name}: la longitud {lon} debe estar en [-180, 180]")

    if args.bbox is not None:
        finite("--bbox", args.bbox)
        south, west, north, east = args.bbox
        check_point("--bbox (sur, oeste)", south, west)
        check_point("--bbox (norte, este)", north, east)
        if not south < north:
            ap.error("--bbox: SUR debe ser menor que NORTE")
        if not west < east:
            ap.error("--bbox: OESTE debe ser menor que ESTE")
    finite("--center", args.center)
    check_point("--center", args.center[0], args.center[1])
    finite("--start-line", args.start_line)
    check_point("--start-line (punto 1)", args.start_line[0], args.start_line[1])
    check_point("--start-line (punto 2)", args.start_line[2], args.start_line[3])
    for name, v in (("--restaurant-snap-m", args.restaurant_snap_m), ("--merge-radius-m", args.merge_radius_m),
                    ("--simplify-tol-m", args.simplify_tol_m),
                    ("--min-shape-spacing-m", args.min_shape_spacing_m),
                    ("--max-deviation-m", args.max_deviation_m)):
        finite(name, [v])
        if not v > 0:
            ap.error(f"{name} debe ser mayor que 0")
    if args.restaurants < 1:
        ap.error("--restaurants debe ser al menos 1")
    if args.couriers is not None and args.couriers < 1:
        ap.error("--couriers debe ser al menos 1")
    if os.path.isabs(args.image) or args.image[1:2] == ":" or args.image.startswith(("/", "\\")):
        ap.error("--image debe ser una ruta RELATIVA al archivo de config, no absoluta")

def merge_overlay(base, overlay):
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge_overlay(base[k], v)
        else:
            base[k] = v

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("osm", help="export de OpenStreetMap (.osm)")
    ap.add_argument("out", help="config JSON a generar")
    ap.add_argument("--image", default="../data/equipetrol.png",
                    help="ruta de la imagen RELATIVA al archivo de config")
    ap.add_argument("--preview", help="PNG con el grafo dibujado sobre el mapa (requiere Pillow)")
    ap.add_argument("--all-streets", action="store_true",
                    help="incluir TODAS las calles residenciales, no solo RESIDENTIAL_NAMES")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("SUR", "OESTE", "NORTE", "ESTE"),
                    help="recorte en grados (orden de Overpass); reemplaza al <bounds> del .osm")
    ap.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"), default=CITY_CENTER,
                    help="centro de la ciudad: ordena restaurantes y define las zonas (anillos)")
    ap.add_argument("--start-line", nargs=4, type=float, metavar=("LAT1", "LON1", "LAT2", "LON2"),
                    default=[*START_LINE[0], *START_LINE[1]],
                    help="el startNode de la flota es el nodo mas cercano al punto medio de este segmento")
    ap.add_argument("--restaurants", type=int, default=RESTAURANT_COUNT, help="cantidad de restaurantes")
    ap.add_argument("--restaurant-snap-m", type=float, default=RESTAURANT_SNAP_M,
                    help="distancia maxima (m) entre un restaurante de OSM y su nodo del grafo")
    ap.add_argument("--couriers", type=int, help="fleet.couriers (por defecto 8)")
    ap.add_argument("--merge-radius-m", type=float, default=MERGE_RADIUS_M,
                    help="radio para fusionar intersecciones cercanas")
    ap.add_argument("--simplify-tol-m", type=float, default=SIMPLIFY_TOL_M,
                    help="tolerancia de Douglas-Peucker y de la contraccion de nodos rectos")
    ap.add_argument("--min-shape-spacing-m", type=float, default=MIN_SHAPE_SPACING_M,
                    help="separacion minima entre nodos de forma en un tramo")
    ap.add_argument("--max-deviation-m", type=float, default=MAX_DEVIATION_M,
                    help="umbral para avisar de tramos que se alejan de la calle real")
    ap.add_argument("--overlay", help="JSON que se fusiona (deep merge) sobre el config generado")
    args = ap.parse_args()
    validate_args(ap, args)

    center = tuple(args.center)
    bounds, nodes, ways, amenities = load_osm(args.osm, args.bbox)
    selected = select_ways(ways, args.all_streets)
    vertex_pos, edges, deviations = build_graph(selected, nodes, bounds, args.merge_radius_m,
                                                args.simplify_tol_m, args.min_shape_spacing_m)
    edges = contract_straight(vertex_pos, edges, bounds, args.simplify_tol_m)
    keep = largest_scc(len(vertex_pos), edges)
    keep &= {x for e in edges for x in e}

    restaurants = pick_restaurants(vertex_pos, keep, amenities, bounds, args.restaurants,
                                   args.restaurant_snap_m, center)
    sm_line = ((args.start_line[0], args.start_line[1]), (args.start_line[2], args.start_line[3]))

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
    if args.overlay:
        with open(args.overlay, encoding="utf-8") as f:
            merge_overlay(config, json.load(f))
    config["map"]["attribution"] = "© OpenStreetMap contributors"  # el overlay no puede quitarla
    if args.couriers is not None:
        config["fleet"]["couriers"] = args.couriers
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")

    n_one = sum(1 for s in streets if s["oneWay"])
    print(f"nodos: {len(order)}  calles: {len(streets)} ({n_one} de un sentido)  "
          f"restaurantes: {len(restaurants)}  descartados fuera de la componente: "
          f"{len(vertex_pos) - len(keep)}", file=sys.stderr)
    for i, (name, v) in enumerate(restaurants[:20]):
        print(f"  r{i}: {name} -> {ids[v]} ({ring_zone(vertex_pos[v], ref_lat, center)})", file=sys.stderr)
    if len(restaurants) > 20:
        print(f"  ... y {len(restaurants) - 20} restaurantes mas", file=sys.stderr)
    print(f"  startNode: {ids[start]}", file=sys.stderr)

    bad = sorted((d for d in deviations if d[0] > args.max_deviation_m and set(d[2]) <= keep),
                 key=lambda d: -d[0])
    print(f"tramos que se alejan más de {args.max_deviation_m:.0f} m de la calle real: {len(bad)}",
          file=sys.stderr)
    for dev, name, path in bad[:15]:
        print(f"  {dev:5.1f} m  {name}  {ids[path[0]]} -> {ids[path[-1]]}", file=sys.stderr)

    if args.preview:
        image_path = os.path.normpath(os.path.join(os.path.dirname(args.out), args.image))
        write_preview(args.preview, image_path, bounds, vertex_pos, keep, edges, restaurants, ids)

if __name__ == "__main__":
    main()
