#!/usr/bin/env python3
"""Prueba de tools/sim_metrics.py con un log chico de valores conocidos (y con 0 pedidos).
Uso: python3 tests/test_sim_metrics.py"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

AQUI = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(AQUI, "..", "tools", "sim_metrics.py")
sys.path.insert(0, os.path.join(AQUI, "..", "tools"))
import sim_metrics  # noqa: E402

LOG_CONOCIDO = [
    {"t": 0, "event": "restaurantPaused", "restaurant": "r0"},
    {"t": 1000, "event": "orderCreated", "order": "o1", "restaurant": "r0", "urgent": False},
    {"t": 1500, "event": "orderCreated", "order": "o2", "restaurant": "r0", "urgent": False},
    {"t": 2000, "event": "orderCreated", "order": "o3", "restaurant": "r0", "urgent": True},
    {"t": 3000, "event": "orderAssigned", "order": "o1", "courier": "c1"},
    {"t": 4000, "event": "orderAssigned", "order": "o2", "courier": "c2"},
    {"t": 5000, "event": "orderRejected", "order": "o3", "reason": "noCourier"},
    {"t": 6000, "event": "orderAssigned", "order": "o1", "courier": "c3"},  # reasignado: no cuenta otra vez
    {"t": 7200000, "event": "orderDelivered", "order": "o1", "courier": "c3"},
    {"t": 7200000, "event": "simulationStopping"},
]

LINEA_FPS = "[estadisticas] cuadros: 120, fps medio 29.8, tiempo de dibujo p50 3.25 ms / p95 7.50 ms\n"


def escribir(directorio, nombre, contenido):
    ruta = os.path.join(directorio, nombre)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)
    return ruta


class PruebasMetricas(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_log_conocido(self):
        log = escribir(self.tmp.name, "a.log", "".join(json.dumps(e) + "\n" for e in LOG_CONOCIDO))
        err = escribir(self.tmp.name, "a.err", "Backend: CPU\n" + LINEA_FPS)
        m = sim_metrics.calcular(log, err)
        self.assertEqual((m["orders_created"], m["orders_assigned"], m["orders_delivered"], m["orders_rejected"]),
                         (3, 2, 1, 1))
        self.assertEqual(m["latency_count"], 2)  # latencias 2000 y 2500 ms simulados
        self.assertAlmostEqual(m["latency_mean"], 2250.0)
        self.assertAlmostEqual(m["latency_p50"], 2250.0)
        self.assertAlmostEqual(m["latency_p90"], 2450.0)
        self.assertAlmostEqual(m["latency_p99"], 2495.0)
        self.assertEqual(m["latency_max"], 2500.0)
        self.assertEqual(m["sim_duration_ms"], 7200000)
        self.assertAlmostEqual(m["orders_per_sim_hour"], 0.5)  # 1 entregado en 2 h simuladas
        self.assertEqual((m["frames"], m["fps_mean"], m["draw_p50_ms"], m["draw_p95_ms"]), (120, 29.8, 3.25, 7.5))

    def test_sin_stderr_ni_cuadros(self):
        log = escribir(self.tmp.name, "b.log", "".join(json.dumps(e) + "\n" for e in LOG_CONOCIDO))
        err = escribir(self.tmp.name, "b.err", "Backend: CPU\n")  # --headless: sin linea de cuadros
        for ruta_err in (None, err):
            m = sim_metrics.calcular(log, ruta_err)
            self.assertIsNone(m["fps_mean"])
            self.assertIsNone(m["frames"])

    def test_cero_pedidos(self):
        log = escribir(self.tmp.name, "c.log", '{"t":0,"event":"restaurantPaused","restaurant":"r0"}\nbasura\n\n')
        m = sim_metrics.calcular(log)
        self.assertEqual(m["orders_created"], 0)
        self.assertEqual(m["latency_count"], 0)
        for clave in ("latency_mean", "latency_p50", "latency_p90", "latency_p99", "latency_max"):
            self.assertIsNone(m[clave])
        self.assertIsNone(m["orders_per_sim_hour"])  # duracion simulada 0: no se divide

    def test_log_vacio(self):
        log = escribir(self.tmp.name, "d.log", "")
        m = sim_metrics.calcular(log)
        self.assertIsNone(m["orders_per_sim_hour"])  # duracion 0: no se puede dividir
        self.assertIsNone(m["latency_p50"])

    def test_cli_json_y_csv(self):
        log = escribir(self.tmp.name, "e.log", "".join(json.dumps(e) + "\n" for e in LOG_CONOCIDO))
        salida = subprocess.run([sys.executable, SCRIPT, log], capture_output=True, text=True, timeout=30)
        self.assertEqual(salida.returncode, 0, salida.stderr)
        self.assertEqual(json.loads(salida.stdout)["orders_created"], 3)
        salida = subprocess.run([sys.executable, SCRIPT, log, "--format", "csv", "--header"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(salida.returncode, 0, salida.stderr)
        encabezado, fila = salida.stdout.strip().split("\n")
        self.assertEqual(len(encabezado.split(",")), len(fila.split(",")))
        self.assertEqual(subprocess.run([sys.executable, SCRIPT, os.path.join(self.tmp.name, "no_existe.log")],
                                        capture_output=True, text=True, timeout=30).returncode, 1)


if __name__ == "__main__":
    unittest.main()
