#!/usr/bin/env python3
"""Metricas de UNA corrida de la simulacion (PDF 2.2 "The simulation itself" y 2.6.5).

Uso:
  sim_metrics.py <log> [--stderr <stderr>] [--format json|csv] [--header]

Lee el log de eventos (una linea JSON por evento) y, si se da, el stderr de la corrida, y calcula:
  - latencia de despacho por pedido: orderCreated -> PRIMER orderAssigned, en tiempo SIMULADO (ms):
    cantidad, media, p50, p90, p99 y maximo (los reasignados cuentan una sola vez, por su primera asignacion);
  - pedidos entregados por hora simulada (entregados / duracion simulada del log), y los rechazos;
  - de stderr: la linea "[estadisticas] cuadros: N, fps medio X, tiempo de dibujo p50 Y ms / p95 Z ms"
    (no existe en --headless: esas columnas quedan vacias / null).
Sin pedidos no falla: las metricas que no se pueden calcular salen null (JSON) o vacias (CSV).
Solo biblioteca estandar de Python 3.
"""
import argparse
import json
import re
import sys

COLUMNAS = [
    "orders_created", "orders_assigned", "orders_delivered", "orders_rejected",
    "latency_count", "latency_mean", "latency_p50", "latency_p90", "latency_p99", "latency_max",
    "sim_duration_ms", "orders_per_sim_hour",
    "frames", "fps_mean", "draw_p50_ms", "draw_p95_ms",
]

FPS = re.compile(
    r"^\[estadisticas\] cuadros: (\d+), fps medio ([0-9.]+), "
    r"tiempo de dibujo p50 ([0-9.]+) ms / p95 ([0-9.]+) ms\s*$")


def percentil(valores_ordenados, fraccion):
    """Percentil con interpolacion lineal; None si no hay valores."""
    if not valores_ordenados:
        return None
    if len(valores_ordenados) == 1:
        return float(valores_ordenados[0])
    posicion = fraccion * (len(valores_ordenados) - 1)
    abajo = int(posicion)
    arriba = min(abajo + 1, len(valores_ordenados) - 1)
    peso = posicion - abajo
    return valores_ordenados[abajo] * (1 - peso) + valores_ordenados[arriba] * peso


def leer_log(ruta):
    """Devuelve (latencias_ms, contadores, duracion_ms). Las lineas que no son JSON se ignoran."""
    creados = {}
    latencias = []
    ya_asignados = set()
    contadores = {"orders_created": 0, "orders_assigned": 0, "orders_delivered": 0, "orders_rejected": 0}
    duracion_ms = 0
    with open(ruta, encoding="utf-8", errors="replace") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                evento = json.loads(linea)
            except ValueError:
                continue
            if not isinstance(evento, dict) or not isinstance(evento.get("t"), int):
                continue
            tiempo = evento["t"]
            duracion_ms = max(duracion_ms, tiempo)
            nombre = evento.get("event")
            pedido = evento.get("order")
            if nombre == "orderCreated" and pedido is not None:
                contadores["orders_created"] += 1
                creados.setdefault(pedido, tiempo)
            elif nombre == "orderAssigned" and pedido is not None:
                if pedido not in ya_asignados:
                    ya_asignados.add(pedido)
                    contadores["orders_assigned"] += 1
                    if pedido in creados:
                        latencias.append(tiempo - creados[pedido])
            elif nombre == "orderDelivered":
                contadores["orders_delivered"] += 1
            elif nombre == "orderRejected":
                contadores["orders_rejected"] += 1
    return latencias, contadores, duracion_ms


def leer_fps(ruta):
    """Ultima linea de cuadros de stderr, o None."""
    if not ruta:
        return None
    encontrado = None
    with open(ruta, encoding="utf-8", errors="replace") as f:
        for linea in f:
            m = FPS.match(linea)
            if m:
                encontrado = {"frames": int(m.group(1)), "fps_mean": float(m.group(2)),
                              "draw_p50_ms": float(m.group(3)), "draw_p95_ms": float(m.group(4))}
    return encontrado


def calcular(ruta_log, ruta_stderr=None):
    latencias, contadores, duracion_ms = leer_log(ruta_log)
    latencias.sort()
    metricas = dict(contadores)
    metricas["latency_count"] = len(latencias)
    metricas["latency_mean"] = sum(latencias) / len(latencias) if latencias else None
    metricas["latency_p50"] = percentil(latencias, 0.50)
    metricas["latency_p90"] = percentil(latencias, 0.90)
    metricas["latency_p99"] = percentil(latencias, 0.99)
    metricas["latency_max"] = float(latencias[-1]) if latencias else None
    metricas["sim_duration_ms"] = duracion_ms
    horas = duracion_ms / 3_600_000.0
    metricas["orders_per_sim_hour"] = contadores["orders_delivered"] / horas if horas > 0 else None
    fps = leer_fps(ruta_stderr) or {"frames": None, "fps_mean": None, "draw_p50_ms": None, "draw_p95_ms": None}
    metricas.update(fps)
    return metricas


def a_csv(metricas):
    celdas = []
    for columna in COLUMNAS:
        valor = metricas[columna]
        celdas.append("" if valor is None else (f"{valor:.3f}" if isinstance(valor, float) else str(valor)))
    return ",".join(celdas)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--stderr", default=None, help="stderr de la corrida (para la linea de cuadros)")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--header", action="store_true", help="con --format csv, imprime antes la fila de nombres")
    args = parser.parse_args(argv)
    try:
        metricas = calcular(args.log, args.stderr)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps({columna: metricas[columna] for columna in COLUMNAS}, indent=2))
    else:
        if args.header:
            print(",".join(COLUMNAS))
        print(a_csv(metricas))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
