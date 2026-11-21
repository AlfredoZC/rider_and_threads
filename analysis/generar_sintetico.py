"""Fabrica un CSV SINTETICO del modo bench (una app ficticia con las 4 versiones) para probar el analisis.

Los tiempos son leyes de potencia y el ruido de cada grupo es una permutacion de FACTORS (mediana
exactamente 1), asi las medianas son exactas y el crossover es conocido:
    mejor CPU (cpuN con 16 hilos) = 1e-5 * n / 6.6 ms
    GPU extremo a extremo           = A * sqrt(n) ms, con A tal que ambos se cruzan en n* = 10**5.3 (~199526)
Uso: C:\\Python313\\python.exe analysis\\generar_sintetico.py [carpeta_salida]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

APP = "demo_sintetica"
SIZES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
THREADS = [1, 2, 4, 6, 8, 12, 16]
SPEEDUP = {1: 0.97, 2: 1.9, 4: 3.6, 6: 5.1, 8: 5.7, 12: 6.2, 16: 6.6}
REPS = 10
SEED = 12345
FACTORS = [0.97, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0, 1.01, 1.03, 1.08]  # mediana = 1.0
EXPECTED_CROSSOVER = 10 ** 5.3  # no es el punto medio geometrico de 1e5 y 1e6: prueba la interpolacion
CPU1_PER_ELEM = 1e-5
A_E2E = CPU1_PER_ELEM / SPEEDUP[16] * math.sqrt(EXPECTED_CROSSOVER)

ENV = {
    "schema": 1,
    "synthetic": True,
    "build": {"compiler": "gcc 13.3.0", "cuda": True, "cuda_flags": "-O3 -DNDEBUG -lineinfo archs=75;89",
              "cxx_flags": "-std=c++17  -O3 -DNDEBUG -Wall -Wextra", "git_commit": "sintetico",
              "sanitizer": "none", "type": "Release"},
    "cpu": {"hardware_concurrency": 16, "model": "13th Gen Intel(R) Core(TM) i7-13620H"},
    "cuda": {"driver": "12.9", "runtime": "12.9"},
    "gpu": {"available": True, "bus_width_bits": 96, "compute_capability": "8.9", "memory_bytes": 6438780928,
            "memory_clock_khz": 8001000, "name": "NVIDIA GeForce RTX 4050 Laptop GPU", "peak_bandwidth_gbs": 192.024,
            "sm_clock_khz": 2355000, "sm_count": 20},
    "memory_total_kb": 10182848,
    "os": "Linux 6.6.87.2-microsoft-standard-WSL2",
    "run": {"apps": [APP], "reps": REPS, "seed": SEED, "thread_sweep": THREADS},
}


def base_times(n: float) -> dict:
    cpu1 = CPU1_PER_ELEM * n
    e2e = A_E2E * math.sqrt(n)
    copies = e2e * (0.3 + 0.1 * math.log10(n / 1e3))   # 30 % a 70 % del tiempo en copias
    kernel = 0.2 * e2e
    return {"cpu1": cpu1, "e2e": e2e, "h2d": copies * 2 / 3, "d2h": copies / 3, "kernel": kernel}


def build_rows(sizes=SIZES, reps=REPS) -> pd.DataFrame:
    rows = []
    group = 0
    for n in sizes:
        b = base_times(n)
        variants = [("cpu1", 1)] + [("cpuN", t) for t in THREADS] + [("gpu_kernel", 1), ("gpu_e2e", 1)]
        for version, threads in variants:
            group += 1
            for rep in range(reps):
                f = FACTORS[(rep + group) % len(FACTORS)]
                h2d = kernel = d2h = ""
                if version == "cpu1":
                    ms = b["cpu1"] * f
                elif version == "cpuN":
                    ms = b["cpu1"] / SPEEDUP[threads] * f
                elif version == "gpu_kernel":
                    ms = b["kernel"] * f
                    kernel = f"{ms:.9g}"
                else:
                    ms = b["e2e"] * f
                    h2d, kernel, d2h = (f"{b[k] * f:.9g}" for k in ("h2d", "kernel", "d2h"))
                rows.append([APP, version, int(n), rep, f"{ms:.9g}", "yes", threads, SEED, h2d, kernel, d2h])
    return pd.DataFrame(rows, columns=["application", "version", "size", "rep", "ms", "correct", "threads", "seed",
                                       "h2d_ms", "kernel_ms", "d2h_ms"])


def write(out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / "results.csv"
    build_rows().to_csv(csv, index=False, lineterminator="\n")
    (out_dir / "results.csv.env.json").write_text(json.dumps(ENV, indent=2) + "\n", encoding="utf-8")
    return csv


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data" / "sintetico"
    print(write(target))
