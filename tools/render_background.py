#!/usr/bin/env python3
"""Dibuja la imagen de fondo de una ciudad a partir de los vectores del propio config.

No descarga teselas (la Tile Usage Policy de OpenStreetMap lo prohibe): usa solo los nodos y las
calles del JSON, que a su vez salen de datos de OpenStreetMap (c) OpenStreetMap contributors, ODbL.
Solo usa la biblioteca estandar (sin Pillow ni numpy), asi corre en cualquier maquina.

  python3 tools/render_background.py config/ciudad.json data/ciudad.png [--width 1800]

La imagen cubre exactamente map.bounds y su relacion de aspecto respeta los metros reales
(como la de equipetrol.png), de modo que lat/lon se mapean linealmente a pixeles.
"""
import argparse
import json
import math
import struct
import zlib

BACKGROUND = (238, 236, 230)
GRID = (226, 223, 215)
CASING = (176, 171, 160)
ROAD = (252, 252, 250)
SUPERSAMPLE = 2


def write_png(path, width, height, planes):
    """planes = (R, G, B) como bytearray de width*height."""
    r, g, b = planes
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        row = slice(y * width, (y + 1) * width)
        interleaved = bytearray(width * 3)
        interleaved[0::3] = r[row]
        interleaved[1::3] = g[row]
        interleaved[2::3] = b[row]
        raw += interleaved

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def disk_offsets(radius):
    r = math.ceil(radius)
    return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)
            if dx * dx + dy * dy <= radius * radius]


def stamp_line(planes, w, h, p, q, offsets, color):
    (x1, y1), (x2, y2) = p, q
    steps = max(1, int(math.hypot(x2 - x1, y2 - y1)))
    for s in range(steps + 1):
        t = s / steps
        cx, cy = int(round(x1 + (x2 - x1) * t)), int(round(y1 + (y2 - y1) * t))
        for dx, dy in offsets:
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                i = y * w + x
                planes[0][i], planes[1][i], planes[2][i] = color


def downsample(plane, w, h, k):
    out = bytearray((w // k) * (h // k))
    ow = w // k
    for y in range(h // k):
        rows = [plane[(y * k + j) * w:(y * k + j + 1) * w] for j in range(k)]
        acc = [0] * ow
        for row in rows:
            for i in range(k):
                acc = [a + v for a, v in zip(acc, row[i::k][:ow])]
        out[y * ow:(y + 1) * ow] = bytes(a // (k * k) for a in acc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="config JSON de la ciudad")
    ap.add_argument("out", help="PNG a generar")
    ap.add_argument("--width", type=int, default=1800, help="ancho final en pixeles")
    ap.add_argument("--grid-m", type=float, default=500.0, help="separacion de la cuadricula tenue (m)")
    args = ap.parse_args()
    if not (math.isfinite(args.grid_m) and args.grid_m > 0):
        ap.error("--grid-m debe ser un numero finito mayor que 0")
    if not 0 < args.width <= 10000:
        ap.error("--width debe estar entre 1 y 10000")

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    b = cfg["map"]["bounds"]
    if not (b["north"] > b["south"] and b["east"] > b["west"]):
        ap.error("map.bounds invalidos: se requiere north > south y east > west")
    ref_lat = (b["north"] + b["south"]) / 2
    width_m = (b["east"] - b["west"]) * 111320.0 * math.cos(math.radians(ref_lat))
    height_m = (b["north"] - b["south"]) * 110574.0
    out_w = args.width
    out_h = round(out_w * height_m / width_m)
    w, h = out_w * SUPERSAMPLE, out_h * SUPERSAMPLE
    px_per_m = w / width_m

    def to_px(lat, lon):
        return ((lon - b["west"]) / (b["east"] - b["west"]) * w,
                (b["north"] - lat) / (b["north"] - b["south"]) * h)

    planes = tuple(bytearray([c]) * (w * h) for c in BACKGROUND)

    step = args.grid_m * px_per_m
    if step < 1:
        ap.error(f"--grid-m {args.grid_m} es demasiado pequeño: la rejilla debe separarse al menos 1 pixel")
    gx = step
    while gx < w:
        for y in range(h):
            i = y * w + int(gx)
            planes[0][i], planes[1][i], planes[2][i] = GRID
        gx += step
    gy = step
    while gy < h:
        i0 = int(gy) * w
        planes[0][i0:i0 + w] = bytes([GRID[0]]) * w
        planes[1][i0:i0 + w] = bytes([GRID[1]]) * w
        planes[2][i0:i0 + w] = bytes([GRID[2]]) * w
        gy += step

    pos = {n["id"]: to_px(n["lat"], n["lon"]) for n in cfg["nodes"]}
    edges = [(pos[s["from"]], pos[s["to"]]) for s in cfg["streets"]]
    casing = disk_offsets(3.4 * SUPERSAMPLE / 2 * 1.0 + 1.2)
    fill = disk_offsets(2.0 * SUPERSAMPLE / 2 * 1.0 + 0.6)
    for p, q in edges:
        stamp_line(planes, w, h, p, q, casing, CASING)
    for p, q in edges:
        stamp_line(planes, w, h, p, q, fill, ROAD)

    small = tuple(downsample(pl, w, h, SUPERSAMPLE) for pl in planes)
    write_png(args.out, out_w, out_h, small)
    print(f"{args.out}: {out_w}x{out_h} px, {len(edges)} calles, {args.grid_m:.0f} m por cuadro de cuadricula")


if __name__ == "__main__":
    main()
