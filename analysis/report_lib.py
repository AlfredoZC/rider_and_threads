"""Analisis del CSV del modo bench para el informe del segundo parcial GPU (enunciado, secciones 2.1-2.6).

Entrada: el CSV que escribe `delivery_sim <config> --headless --bench <csv>` y su `<csv>.env.json`.
Columnas exactas: application,version,size,rep,ms,correct,threads,seed,h2d_ms,kernel_ms,d2h_ms.

Que calcula (todo a partir de la mediana de >= 10 repeticiones, seccion 2.2):
  * validacion: columnas exactas, ninguna fila correct=no, >= 10 repeticiones por grupo
    (app, version, size, threads), >= 5 tamanos y >= 2 ordenes de magnitud por aplicacion, build Release;
  * mediana y dispersion (min, max, IQR) por (app, version, size, threads);
  * "mejor CPU" = minimo entre cpu1 y el mejor cpuN del barrido de hilos;
  * speedups kernel y extremo a extremo contra la mejor CPU;
  * crossover = primer tamano en que gpu_e2e < mejor CPU, interpolado en log-log;
  * tabla por aplicacion con las columnas exactas de 2.4, barrido de hilos, roofline y resumen de 2.6.

Uso: ver `generar_informe.py` (sin Jupyter) e `informe.ipynb`.
"""
from __future__ import annotations

import ast
import json
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent

EXPECTED_COLUMNS = [
    "application", "version", "size", "rep", "ms", "correct",
    "threads", "seed", "h2d_ms", "kernel_ms", "d2h_ms",
]
VERSIONS = ["cpu1", "cpuN", "gpu_kernel", "gpu_e2e"]
MIN_REPS = 10          # seccion 2.2: "at least 10 timed runs"
MIN_SIZES = 5          # seccion 2.2: "at least 5 problem sizes"
MIN_DECADES = 2.0      # "spanning at least two orders of magnitude"
HIGH_SPREAD = 0.5      # (max - min) / mediana por encima de esto: posible estrangulamiento termico

# Encabezados exactos de la tabla de la seccion 2.4 ("N" se reemplaza por el numero de hilos).
TABLE_COLUMNS = [
    "Size", "CPU 1 thread (ms)", "CPU N threads (ms)", "GPU kernel (ms)", "GPU end to end (ms)",
    "Speedup vs best CPU (kernel / end to end)", "Correct?",
]

# Nucleos de la laptop de referencia. El env.json solo trae el modelo y hardware_concurrency;
# la division P/E sale de la ficha del fabricante (Intel ARK, i7-13620H: 6 P con HT + 4 E = 16 hilos).
CPU_TOPOLOGY = {
    "i7-13620H": {"p_cores": 6, "e_cores": 4, "logical": 16,
                  "source": "Intel ARK i7-13620H: 6 nucleos P (2 hilos c/u) + 4 nucleos E = 10 nucleos, 16 hilos"},
}

# FP32 por SM y por ciclo (una FMA cuenta 2 FLOP). Ada (cc 8.9): 2x las operaciones FP32 por ciclo y SM
# de cc 8.0 (Ada Tuning Guide de NVIDIA), es decir 128. Los valores INT32 son los de la tabla de throughput
# de la CUDA C Programming Guide (64 resultados/ciclo/SM de suma entera de 32 bits en 7.5-8.9).
FP32_LANES_PER_SM = {"7.5": 64, "8.0": 64, "8.6": 128, "8.7": 128, "8.9": 128, "9.0": 128}
INT32_LANES_PER_SM = {"7.5": 64, "8.0": 64, "8.6": 64, "8.7": 64, "8.9": 64, "9.0": 64}

VERSION_LABELS = {
    "cpu1": "CPU 1 hilo",
    "cpuN": "CPU N hilos (mejor N)",
    "gpu_kernel": "GPU solo kernel",
    "gpu_e2e": "GPU extremo a extremo",
}


class BenchDataError(ValueError):
    """El CSV no se puede analizar (p. ej. columnas distintas de las del ADR 0003)."""


@dataclass
class Issue:
    level: str          # "error" (incumple el enunciado), "aviso" o "info"
    app: str | None
    message: str

    def __str__(self) -> str:
        where = f"[{self.app}] " if self.app else ""
        return f"{self.level.upper()}: {where}{self.message}"


@dataclass
class Dataset:
    df: pd.DataFrame
    env: dict
    csv_paths: list[Path]
    issues: list[Issue] = field(default_factory=list)

    @property
    def apps(self) -> list[str]:
        return list(dict.fromkeys(self.df["application"]))

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]


# --------------------------------------------------------------------------------------------------
# Carga y validacion
# --------------------------------------------------------------------------------------------------

def read_raw(csv_path: str | Path) -> pd.DataFrame:
    """Lee el CSV como texto (sin convertir) para poder validar columnas y valores tal cual vienen."""
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False)


def load_env(csv_path: str | Path) -> dict:
    env_path = Path(str(csv_path) + ".env.json")
    if not env_path.exists():
        return {}
    with open(env_path, encoding="utf-8") as f:
        return json.load(f)


def validate_columns(raw: pd.DataFrame) -> list[Issue]:
    cols = list(raw.columns)
    if cols == EXPECTED_COLUMNS:
        return []
    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in EXPECTED_COLUMNS]
    parts = []
    if missing:
        parts.append(f"faltan {missing}")
    if extra:
        parts.append(f"sobran {extra}")
    if not missing and not extra:
        parts.append(f"orden distinto: {cols}")
    return [Issue("error", None, "columnas distintas de las esperadas " + str(EXPECTED_COLUMNS) + ": " + "; ".join(parts))]


def convert(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[Issue]]:
    """Convierte tipos. Las filas con valores invalidos se marcan y se descartan."""
    issues: list[Issue] = []
    df = raw.copy()
    df["application"] = df["application"].str.strip()
    df["version"] = df["version"].str.strip()
    df["correct"] = df["correct"].str.strip().str.lower()
    for col in ["size", "rep", "threads", "seed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["ms", "h2d_ms", "kernel_ms", "d2h_ms"]:
        df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    bad = pd.Series(False, index=df.index)
    bad_version = ~df["version"].isin(VERSIONS)
    if bad_version.any():
        issues.append(Issue("error", None, f"{int(bad_version.sum())} filas con version fuera de {VERSIONS}: "
                            f"{sorted(set(df.loc[bad_version, 'version']))[:5]}"))
    bad_correct = ~df["correct"].isin(["yes", "no"])
    if bad_correct.any():
        issues.append(Issue("error", None, f"{int(bad_correct.sum())} filas con correct distinto de yes/no"))
    bad_num = df[["size", "rep", "threads", "ms"]].isna().any(axis=1) | ~np.isfinite(df["ms"].fillna(np.nan))
    bad_num |= (df["ms"] < 0) | (df["size"] <= 0) | (df["threads"] < 1)
    if bad_num.any():
        issues.append(Issue("error", None, f"{int(bad_num.sum())} filas con size/rep/threads/ms invalidos (vacios, "
                            "no numericos, negativos o size <= 0)"))
    bad |= bad_version | bad_correct | bad_num
    df = df[~bad].copy()
    for col in ["size", "rep", "threads"]:
        df[col] = df[col].astype("int64")
    df["seed"] = df["seed"].astype("Int64")
    return df.reset_index(drop=True), issues


def validate(df: pd.DataFrame, env: dict | None = None) -> list[Issue]:
    """Reglas de la seccion 2.2 sobre datos ya convertidos. Devuelve la lista de incumplimientos."""
    env = env or {}
    issues: list[Issue] = []
    keys = ["application", "version", "size", "threads"]

    no = df[df["correct"] == "no"]
    if len(no):
        for (app, ver, size, thr), g in no.groupby(keys):
            issues.append(Issue("error", app, f"correct=no en {len(g)} corrida(s): version={ver} size={size} "
                                f"threads={thr} (reps {sorted(g['rep'].tolist())[:10]})"))

    counts = df.groupby(keys).size()
    few = counts[counts < MIN_REPS]
    for (app, ver, size, thr), n in few.items():
        issues.append(Issue("error", app, f"solo {n} repeticiones (< {MIN_REPS}) en version={ver} size={size} threads={thr}"))

    dup = df.groupby(keys + ["rep"]).size()
    dup = dup[dup > 1]
    if len(dup):
        apps = sorted(set(i[0] for i in dup.index))
        for app in apps:
            issues.append(Issue("aviso", app, f"{int((dup.loc[app] > 1).sum())} repeticiones duplicadas (mismo rep en el mismo grupo)"))

    for app, g in df.groupby("application", sort=False):
        sizes = sorted(g["size"].unique())
        if len(sizes) < MIN_SIZES:
            issues.append(Issue("error", app, f"solo {len(sizes)} tamanos (< {MIN_SIZES}): {sizes}"))
        if sizes and sizes[0] > 0:
            decades = math.log10(sizes[-1] / sizes[0])
            if decades < MIN_DECADES:
                issues.append(Issue("error", app, f"los tamanos cubren {decades:.2f} ordenes de magnitud (< {MIN_DECADES:g}): "
                                    f"{sizes[0]} a {sizes[-1]}"))
        present = set(g["version"])
        missing = [v for v in VERSIONS if v not in present]
        if missing:
            issues.append(Issue("aviso", app, f"faltan versiones {missing} (la tabla de 2.4 queda incompleta)"))
        seeds = g["seed"].dropna().unique()
        if len(seeds) > 1:
            issues.append(Issue("aviso", app, f"mas de una semilla ({list(seeds)[:5]}): 2.2 pide la misma entrada para todas las versiones"))
        hc = (env.get("cpu") or {}).get("hardware_concurrency")
        cpun = g[g["version"] == "cpuN"]
        if hc and len(cpun) and int(cpun["threads"].max()) < int(hc):
            issues.append(Issue("aviso", app, f"el barrido de hilos llega a {int(cpun['threads'].max())}, no a los {hc} hilos logicos"))
        e2e = g[g["version"] == "gpu_e2e"]
        if len(e2e) and e2e[["h2d_ms", "kernel_ms", "d2h_ms"]].isna().all(axis=None):
            issues.append(Issue("info", app, "gpu_e2e sin h2d_ms/kernel_ms/d2h_ms: no se puede graficar el reparto copias/kernel"))

    build = env.get("build") or {}
    if build:
        if build.get("type") != "Release":
            issues.append(Issue("error", None, f"build '{build.get('type')}': 2.2 exige Release (nunca Debug ni sanitizer)"))
        if build.get("sanitizer") not in (None, "none", ""):
            issues.append(Issue("error", None, f"build con sanitizer '{build.get('sanitizer')}': sus tiempos no valen"))
    elif env == {}:
        issues.append(Issue("aviso", None, "no se encontro <csv>.env.json: faltan los datos de hardware y build"))
    return issues


def load(csv_paths: str | Path | Iterable[str | Path]) -> Dataset:
    """Carga uno o varios CSV del modo bench (misma maquina), con su env.json, y valida.

    Lanza BenchDataError solo si las columnas no son las esperadas (sin ellas no hay analisis posible);
    los demas incumplimientos quedan en `Dataset.issues`.
    """
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    paths = [Path(p) for p in csv_paths]
    frames, issues, envs = [], [], []
    for p in paths:
        raw = read_raw(p)
        col_issues = validate_columns(raw)
        if col_issues:
            raise BenchDataError(f"{p}: {col_issues[0].message}")
        df, conv_issues = convert(raw)
        frames.append(df)
        issues += conv_issues
        envs.append(load_env(p))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EXPECTED_COLUMNS)
    env = merge_envs(envs, issues)
    issues += validate(df, env)
    return Dataset(df=df, env=env, csv_paths=paths, issues=issues)


def merge_envs(envs: list[dict], issues: list[Issue]) -> dict:
    envs = [e for e in envs if e]
    if not envs:
        return {}
    base = json.loads(json.dumps(envs[0]))
    for e in envs[1:]:
        for path in [("cpu", "model"), ("gpu", "name"), ("build", "type"), ("build", "cxx_flags")]:
            a = (base.get(path[0]) or {}).get(path[1])
            b = (e.get(path[0]) or {}).get(path[1])
            if a != b:
                issues.append(Issue("error", None, f"los CSV vienen de entornos distintos: {'.'.join(path)} = {a!r} vs {b!r}"))
        apps = list(dict.fromkeys((base.get("run") or {}).get("apps", []) + (e.get("run") or {}).get("apps", [])))
        base.setdefault("run", {})["apps"] = apps
    return base


# --------------------------------------------------------------------------------------------------
# Estadisticas
# --------------------------------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mediana y dispersion por (application, version, size, threads)."""
    keys = ["application", "version", "size", "threads"]
    g = df.groupby(keys, sort=True)
    out = g["ms"].agg(n="count", median="median", min="min", max="max")
    out["q1"] = g["ms"].quantile(0.25)
    out["q3"] = g["ms"].quantile(0.75)
    out["iqr"] = out["q3"] - out["q1"]
    out["spread"] = (out["max"] - out["min"]) / out["median"].where(out["median"] > 0)
    out["h2d_med"] = g["h2d_ms"].median()
    out["kernel_med"] = g["kernel_ms"].median()
    out["d2h_med"] = g["d2h_ms"].median()
    out["n_correct"] = g["correct"].agg(lambda s: int((s == "yes").sum()))
    return out.reset_index()


@dataclass
class Crossover:
    kind: str                 # "cross" | "gpu_always" | "never" | "no_data"
    size: float | None = None
    lo: float | None = None   # tamanos medidos que encierran el cruce
    hi: float | None = None
    note: str = ""
    upper: float | None = None  # si despues del cruce la GPU vuelve a perder: tamano (interpolado) en que pierde

    def text(self, unit: str | None = None) -> str:
        if self.kind == "no_data":
            return "sin datos de GPU"
        if self.kind == "never":
            return "no hay cruce (la GPU no gana en el rango medido)"
        if self.kind == "gpu_always":
            t = f"la GPU gana desde el menor tamano medido ({fmt_size(self.size, unit)})"
        else:
            t = f"≈ {fmt_size(self.size, unit, approx=True)} (entre {fmt_size(self.lo, unit)} y {fmt_size(self.hi, unit)})"
        if self.upper is not None:
            t += f"; vuelve a perder desde ≈ {fmt_size(self.upper, unit, approx=True)}"
        return t + (f"; {self.note}" if self.note else "")


def crossover(sizes, cpu_ms, gpu_ms) -> Crossover:
    """Primer tamano en que la GPU (gpu_ms) baja de la CPU (cpu_ms), interpolando en log-log.

    Entre dos tamanos medidos s0 < s1 con d = log(gpu) - log(cpu), d0 >= 0 y d1 < 0, el cruce es
    log s* = log s0 + d0 / (d0 - d1) * (log s1 - log s0): exacto si ambos tiempos son leyes de potencia
    en ese tramo.
    """
    pts = sorted((float(s), float(c), float(g)) for s, c, g in zip(sizes, cpu_ms, gpu_ms)
                 if np.isfinite(c) and np.isfinite(g) and c > 0 and g > 0 and s > 0)
    if not pts:
        return Crossover("no_data")
    s = np.array([p[0] for p in pts])
    d = np.log([p[2] for p in pts]) - np.log([p[1] for p in pts])
    neg = np.nonzero(d < 0)[0]
    if len(neg) == 0:
        return Crossover("never")
    i = int(neg[0])
    later = [j for j in range(i + 1, len(s)) if d[j] >= 0]
    upper = None
    if later:
        j = later[0]
        tj = d[j - 1] / (d[j - 1] - d[j])
        upper = math.exp(math.log(s[j - 1]) + tj * (math.log(s[j]) - math.log(s[j - 1])))
    if i == 0:
        return Crossover("gpu_always", size=s[0], lo=s[0], hi=s[0], upper=upper)
    s0, s1, d0, d1 = s[i - 1], s[i], d[i - 1], d[i]
    t = d0 / (d0 - d1)
    size = math.exp(math.log(s0) + t * (math.log(s1) - math.log(s0)))
    return Crossover("cross", size=size, lo=s0, hi=s1, upper=upper)


def compare(summary: pd.DataFrame) -> pd.DataFrame:
    """Por (aplicacion, tamano): cpu1, mejor cpuN (y su N), mejor CPU, GPU y speedups."""
    rows = []
    for (app, size), g in summary.groupby(["application", "size"], sort=True):
        def med(version):
            r = g[g["version"] == version]
            return (float(r["median"].iloc[0]), r.iloc[0]) if len(r) else (np.nan, None)

        cpu1, cpu1_row = med("cpu1")
        cpun_rows = g[g["version"] == "cpuN"]
        if len(cpun_rows):
            best_row = cpun_rows.loc[cpun_rows["median"].idxmin()]
            cpun, cpun_threads = float(best_row["median"]), int(best_row["threads"])
        else:
            best_row, cpun, cpun_threads = None, np.nan, None
        gk, gk_row = med("gpu_kernel")
        ge, ge_row = med("gpu_e2e")
        cands = [(v, lbl) for v, lbl in [(cpu1, "cpu1"), (cpun, f"cpuN({cpun_threads})")] if np.isfinite(v)]
        best, best_label = min(cands) if cands else (np.nan, "")
        rows.append({
            "application": app, "size": int(size),
            "cpu1_ms": cpu1, "cpuN_ms": cpun, "cpuN_threads": cpun_threads,
            "best_cpu_ms": best, "best_cpu": best_label,
            "gpu_kernel_ms": gk, "gpu_e2e_ms": ge,
            "speedup_kernel": best / gk if np.isfinite(gk) and gk > 0 else np.nan,
            "speedup_e2e": best / ge if np.isfinite(ge) and ge > 0 else np.nan,
            "n_runs": int(g["n"].sum()), "n_correct": int(g["n_correct"].sum()),
            "_rows": {"cpu1": cpu1_row, "cpuN": best_row, "gpu_kernel": gk_row, "gpu_e2e": ge_row},
        })
    return pd.DataFrame(rows)


def apply_cpu_baselines(cmp: pd.DataFrame, kernels: dict | None) -> pd.DataFrame:
    """La "mejor CPU" de una aplicacion puede ser otra aplicacion del CSV que resuelve la MISMA tarea solo en CPU
    (p. ej. apsp_floyd contra apsp_dijkstra) o la aplicacion principal de una variante (histogram_naive contra
    delivery_histogram). kernels.json lo declara con "cpu_baselines": [apps]. Recalcula mejor CPU y speedups."""
    if not kernels or cmp.empty:
        return cmp
    cmp = cmp.copy()
    by_key = {(r["application"], int(r["size"])): r for _, r in cmp.iterrows()}
    for idx, r in cmp.iterrows():
        meta = app_meta(kernels, r["application"])
        for base in meta.get("cpu_baselines") or []:
            other = by_key.get((base, int(r["size"])))
            if other is None:
                continue
            for col, label in [("cpu1_ms", f"{base}:cpu1"), ("cpuN_ms", f"{base}:cpuN({other['cpuN_threads']})")]:
                value = other[col]
                if np.isfinite(value) and (not np.isfinite(cmp.at[idx, "best_cpu_ms"]) or value < cmp.at[idx, "best_cpu_ms"]):
                    cmp.at[idx, "best_cpu_ms"] = value
                    cmp.at[idx, "best_cpu"] = label
        best = cmp.at[idx, "best_cpu_ms"]
        gk, ge = cmp.at[idx, "gpu_kernel_ms"], cmp.at[idx, "gpu_e2e_ms"]
        cmp.at[idx, "speedup_kernel"] = best / gk if np.isfinite(gk) and gk > 0 else np.nan
        cmp.at[idx, "speedup_e2e"] = best / ge if np.isfinite(ge) and ge > 0 else np.nan
    return cmp


def compare_all(summary: pd.DataFrame, kernels: dict | None = None) -> pd.DataFrame:
    return apply_cpu_baselines(compare(summary), kernels)


def is_main_app(kernels: dict, app: str) -> bool:
    """Las variantes de comparacion y las CPU de referencia no son aplicaciones del resumen de 2.6."""
    meta = app_meta(kernels, app)
    return not meta.get("variant_of") and not meta.get("baseline_for")


def variant_table(cmp: pd.DataFrame, main: str, variant: str, unit: str | None = None) -> pd.DataFrame:
    """Kernel principal contra la variante de comparacion (misma tarea, mismos tamanos)."""
    a = cmp[cmp["application"] == main].set_index("size")
    b = cmp[cmp["application"] == variant].set_index("size")
    rows = []
    for size in sorted(set(a.index) & set(b.index)):
        ka, kb = a.at[size, "gpu_kernel_ms"], b.at[size, "gpu_kernel_ms"]
        ea, eb = a.at[size, "gpu_e2e_ms"], b.at[size, "gpu_e2e_ms"]
        rows.append([fmt_size(size, unit), fmt_num(ka), fmt_num(kb), fmt_x(kb / ka if ka > 0 else np.nan),
                     fmt_num(ea), fmt_num(eb)])
    return pd.DataFrame(rows, columns=["Size", f"Kernel {main} (ms)", f"Kernel {variant} (ms)",
                                       f"{variant} / {main} (kernel)", f"E2E {main} (ms)", f"E2E {variant} (ms)"])


def app_crossover(cmp: pd.DataFrame, app: str, gpu_col: str = "gpu_e2e_ms") -> Crossover:
    c = cmp[cmp["application"] == app].sort_values("size")
    return crossover(c["size"], c["best_cpu_ms"], c[gpu_col])


def thread_sweep(summary: pd.DataFrame, app: str, size: int | None = None) -> tuple[int | None, pd.DataFrame]:
    """Speedup de cpuN contra cpu1 en el mayor tamano con el barrido de hilos completo."""
    s = summary[(summary["application"] == app)]
    cpun = s[s["version"] == "cpuN"]
    if cpun.empty:
        return None, pd.DataFrame()
    all_threads = set(cpun["threads"])
    if size is None:
        full = [sz for sz, g in cpun.groupby("size") if set(g["threads"]) == all_threads
                and not s[(s["size"] == sz) & (s["version"] == "cpu1")].empty]
        if not full:
            return None, pd.DataFrame()
        size = max(full)
    base = s[(s["size"] == size) & (s["version"] == "cpu1")]
    if base.empty:
        return size, pd.DataFrame()
    t1 = float(base["median"].iloc[0])
    g = cpun[cpun["size"] == size].sort_values("threads").copy()
    g["speedup"] = t1 / g["median"]
    g["speedup_lo"] = t1 / g["max"]
    g["speedup_hi"] = t1 / g["min"]
    g["efficiency"] = g["speedup"] / g["threads"]
    return int(size), g[["threads", "n", "median", "min", "max", "speedup", "speedup_lo", "speedup_hi", "efficiency"]]


def cpu_topology(env: dict) -> dict | None:
    model = (env.get("cpu") or {}).get("model", "")
    for key, topo in CPU_TOPOLOGY.items():
        if key in model:
            return topo
    return None


# --------------------------------------------------------------------------------------------------
# Techos de la GPU y roofline
# --------------------------------------------------------------------------------------------------

def gpu_peaks(env: dict) -> dict:
    """Picos teoricos a partir del env.json: FP32 = SMs x nucleos FP32/SM x 2 (FMA) x reloj."""
    gpu = env.get("gpu") or {}
    if not gpu.get("available"):
        return {}
    cc = str(gpu.get("compute_capability"))
    sms = int(gpu.get("sm_count", 0))
    clock_hz = float(gpu.get("sm_clock_khz", 0)) * 1e3
    fp32_lanes = FP32_LANES_PER_SM.get(cc)
    int32_lanes = INT32_LANES_PER_SM.get(cc)
    return {
        "name": gpu.get("name"), "cc": cc, "sms": sms, "clock_ghz": clock_hz / 1e9,
        "fp32_lanes_per_sm": fp32_lanes,
        "fp32_gflops": sms * fp32_lanes * 2 * clock_hz / 1e9 if fp32_lanes else np.nan,
        "int32_lanes_per_sm": int32_lanes,
        "int32_gops": sms * int32_lanes * clock_hz / 1e9 if int32_lanes else np.nan,
        "bw_peak_gbs": float(gpu.get("peak_bandwidth_gbs", np.nan)),
    }


def measured_bandwidth(summary: pd.DataFrame, app: str = "calib_copy") -> dict | None:
    """Ancho de banda de VRAM medido por calib_copy gpu_kernel: 2 x bytes / tiempo (lee y escribe cada byte).

    Se toma el mayor tamano: los buffers chicos caben en la cache L2 y dan valores por encima del pico de DRAM.
    """
    g = summary[(summary["application"] == app) & (summary["version"] == "gpu_kernel")].sort_values("size")
    if g.empty:
        return None
    per_size = pd.DataFrame({
        "size": g["size"].astype(int),
        "kernel_ms": g["median"],
        "gbs": 2.0 * g["size"] / (g["median"] * 1e-3) / 1e9,
        "gbs_best": 2.0 * g["size"] / (g["min"] * 1e-3) / 1e9,
    }).reset_index(drop=True)
    last = per_size.iloc[-1]
    return {"gbs": float(last["gbs"]), "size": int(last["size"]), "per_size": per_size}


_ALLOWED_FUNCS = {"log2": math.log2, "log10": math.log10, "sqrt": math.sqrt, "ceil": math.ceil,
                  "floor": math.floor, "min": min, "max": max}
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def eval_formula(expr: str, variables: dict) -> float:
    """Evalua una formula aritmetica de kernels.json sin `eval`: solo numeros, variables conocidas,
    + - * / // % **, y log2/log10/sqrt/ceil/floor/min/max."""
    tree = ast.parse(str(expr), mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"variable desconocida '{node.id}' en '{expr}'")
            return float(variables[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 16:
                raise ValueError(f"exponente demasiado grande en '{expr}'")
            return float(_BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS
                and not node.keywords):
            return float(_ALLOWED_FUNCS[node.func.id](*[ev(a) for a in node.args]))
        raise ValueError(f"expresion no permitida en '{expr}': {ast.dump(node)[:60]}")

    return ev(tree)


def load_kernels(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ANALYSIS_DIR / "kernels.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def app_meta(kernels: dict, app: str) -> dict:
    return (kernels.get("apps") or {}).get(app, {})


def size_variables(meta: dict, size: float) -> dict:
    """n = size, los parametros fijos y las variables que dependen del tamano ("size_vars": {var: {"50": 10, ...,
    "default": "formula de n"}}), p. ej. la cantidad de pedidos de una matriz pedidos x couriers."""
    variables = {"n": float(size), **{k: float(v) for k, v in (meta.get("params") or {}).items()}}
    for name, table in (meta.get("size_vars") or {}).items():
        key = str(int(size))
        value = table.get(key, table.get("default"))
        variables[name] = float(value) if isinstance(value, (int, float)) else eval_formula(str(value), variables)
    return variables


def roofline_points(summary: pd.DataFrame, kernels: dict) -> pd.DataFrame:
    """Intensidad aritmetica (ops / bytes de memoria global) y rendimiento logrado (ops / tiempo del kernel)."""
    rows = []
    gk = summary[summary["version"] == "gpu_kernel"]
    for app, g in gk.groupby("application", sort=False):
        meta = app_meta(kernels, app)
        if not meta or not meta.get("ops") or not meta.get("bytes"):
            continue
        for _, r in g.sort_values("size").iterrows():
            variables = size_variables(meta, float(r["size"]))
            ops = eval_formula(meta["ops"], variables)
            nbytes = eval_formula(meta["bytes"], variables)
            if ops <= 0 or nbytes <= 0 or r["median"] <= 0:
                continue
            rows.append({
                "application": app, "size": int(r["size"]), "op_type": meta.get("op_type", "fp32"),
                "ops": ops, "bytes": nbytes, "ai": ops / nbytes,
                "kernel_ms": float(r["median"]),
                "gops": ops / (float(r["median"]) * 1e-3) / 1e9,
                "gbs": nbytes / (float(r["median"]) * 1e-3) / 1e9,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Formato
# --------------------------------------------------------------------------------------------------

def fmt_num(v, sig: int = 3) -> str:
    """3 cifras significativas en notacion fija (0.0000960, 13.3, 1472)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/d"
    v = float(v)
    if v == 0:
        return "0"
    mag = math.floor(math.log10(abs(v)))
    decimals = max(0, sig - 1 - mag)
    if decimals > 8:
        return f"{v:.{sig - 1}e}"
    s = f"{v:,.{decimals}f}"
    return s


def fmt_size(size, unit: str | None = None, approx: bool = False) -> str:
    if size is None or not np.isfinite(size):
        return "n/d"
    if unit == "bytes":
        for factor, name in [(1 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")]:
            if size >= factor:
                v = size / factor
                return (f"{v:.3g} {name}" if approx or v != int(v) else f"{int(v)} {name}")
        return f"{size:.3g} B" if approx else f"{int(size)} B"
    text = fmt_num(size) if approx else f"{int(round(size)):,}"
    return text + (f" {unit}" if unit and approx else "")


def fmt_x(v) -> str:
    return "n/d" if v is None or not np.isfinite(v) else f"{fmt_num(v)}×"


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in r.tolist()) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# Tablas del informe
# --------------------------------------------------------------------------------------------------

def app_table(cmp: pd.DataFrame, app: str, unit: str | None = None) -> tuple[pd.DataFrame, str]:
    """Tabla de la seccion 2.4 con sus columnas exactas. Devuelve (tabla, nota sobre N)."""
    c = cmp[cmp["application"] == app].sort_values("size")
    threads = sorted(set(int(t) for t in c["cpuN_threads"].dropna()))
    cols = list(TABLE_COLUMNS)
    if len(threads) == 1:
        cols[2] = f"CPU {threads[0]} threads (ms)"
        note = (f"N = {threads[0]}: el mejor numero de hilos del barrido en todos los tamanos. "
                "Mejor CPU = min(CPU 1 thread, CPU N threads).")
    else:
        note = ("N = el mejor numero de hilos del barrido en cada tamano (entre parentesis). "
                "Mejor CPU = min(CPU 1 thread, CPU N threads).")
    rows = []
    for _, r in c.iterrows():
        cpun = fmt_num(r["cpuN_ms"])
        if len(threads) > 1 and pd.notna(r["cpuN_threads"]):
            cpun += f" (N={int(r['cpuN_threads'])})"
        ok = r["n_correct"] == r["n_runs"]
        rows.append([
            fmt_size(r["size"], unit), fmt_num(r["cpu1_ms"]), cpun, fmt_num(r["gpu_kernel_ms"]),
            fmt_num(r["gpu_e2e_ms"]), f"{fmt_x(r['speedup_kernel'])} / {fmt_x(r['speedup_e2e'])}",
            f"{'sí' if ok else 'NO'} ({r['n_correct']}/{r['n_runs']})",
        ])
    return pd.DataFrame(rows, columns=cols), note


def dispersion_table(cmp: pd.DataFrame, app: str, unit: str | None = None) -> pd.DataFrame:
    c = cmp[cmp["application"] == app].sort_values("size")
    rows = []
    for _, r in c.iterrows():
        for v in VERSIONS:
            row = r["_rows"].get(v)
            if row is None:
                continue
            spread = row["spread"]
            rows.append([
                fmt_size(r["size"], unit), v, int(row["threads"]), int(row["n"]), fmt_num(row["median"]),
                fmt_num(row["min"]), fmt_num(row["max"]), fmt_num(row["iqr"]),
                (f"{spread:.0%}" + (" (alta)" if spread > HIGH_SPREAD else "")) if np.isfinite(spread) else "n/d",
            ])
    return pd.DataFrame(rows, columns=["Size", "Versión", "Hilos", "n", "Mediana (ms)", "Mín (ms)", "Máx (ms)",
                                       "IQR (ms)", "(máx−mín)/mediana"])


def sweep_table(sweep: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Hilos": sweep["threads"].astype(int),
        "Mediana (ms)": [fmt_num(v) for v in sweep["median"]],
        "Speedup vs CPU 1 thread": [fmt_x(v) for v in sweep["speedup"]],
        "Rango (con mín/máx)": [f"{fmt_num(a)}–{fmt_num(b)}" for a, b in zip(sweep["speedup_lo"], sweep["speedup_hi"])],
        "Eficiencia": [f"{v:.0%}" for v in sweep["efficiency"]],
    })


def verdict(cross: Crossover) -> str:
    if cross.kind in ("cross", "gpu_always") and cross.upper is not None:
        return "usar GPU solo en el rango ganador" if cross.kind == "cross" else "usar GPU solo hasta el umbral superior"
    return {
        "no_data": "sin datos de GPU",
        "never": "mantener CPU",
        "gpu_always": "usar GPU",
        "cross": f"usar GPU sobre el umbral",
    }[cross.kind]


def winner_at(app_rows: pd.DataFrame, size: float) -> str:
    """Quien gana (mejor CPU o GPU extremo a extremo) en un tamano, interpolando log(gpu/cpu) en log(tamano)."""
    c = app_rows[np.isfinite(app_rows["gpu_e2e_ms"]) & np.isfinite(app_rows["best_cpu_ms"])].sort_values("size")
    if c.empty:
        return "n/d"
    x = np.log(c["size"].to_numpy(dtype=float))
    d = np.log(c["gpu_e2e_ms"].to_numpy()) - np.log(c["best_cpu_ms"].to_numpy())
    if size < c["size"].min() or size > c["size"].max():
        return f"fuera del rango medido ({int(c['size'].min()):,}–{int(c['size'].max()):,})"
    return "GPU" if np.interp(math.log(size), x, d) < 0 else "CPU"


def scale_verdicts(app_rows: pd.DataFrame, scales: dict | None) -> str:
    """Veredicto por escala del proyecto (brief y ciudad) si kernels.json da los tamanos de operacion."""
    if not scales:
        return ""
    return "; ".join(f"{name} ({int(n):,}): {winner_at(app_rows, float(n))}" for name, n in scales.items() if n is not None)


def grade(points: int) -> float:
    """Nota del enunciado: >= 10 puntos: 50 + 50 (p - 10) / 20; si no, 50 p / 10."""
    return 50 + 50 * (points - 10) / 20 if points >= 10 else 50 * points / 10


CONCEPTS = ["03_memory_tiling", "04_performance", "05_conv", "06_stencil", "07_histogram",
            "08_reduction", "09_scan", "10_filter", "11_merge", "12_sorting"]


def summary_table(ds: Dataset, kernels: dict) -> tuple[pd.DataFrame, dict]:
    """Tabla resumen de la seccion 2.6 (punto 1) y el calculo de puntos y nota."""
    summ = summarize(ds.df)
    cmp = compare_all(summ, kernels)
    errors_by_app = {}
    for i in ds.issues:
        if i.level == "error":
            errors_by_app.setdefault(i.app, []).append(i.message)
    rows, per_concept = [], {c: 0 for c in CONCEPTS}
    ordinals = {1: "1ª", 2: "2ª", 3: "3ª"}
    for app in ds.apps:
        if not is_main_app(kernels, app):
            continue
        meta = app_meta(kernels, app)
        unit = meta.get("size_unit")
        cross = app_crossover(cmp, app)
        c = cmp[cmp["application"] == app]
        best = c.loc[c["speedup_e2e"].idxmax()] if c["speedup_e2e"].notna().any() else None
        best_txt = f"{fmt_x(best['speedup_e2e'])} ({fmt_size(best['size'], unit)})" if best is not None else "n/d"
        v = verdict(cross)
        if cross.kind == "cross" and cross.upper is not None:
            v += f" (≈ {fmt_size(cross.size, unit, approx=True)} a ≈ {fmt_size(cross.upper, unit, approx=True)})"
        elif cross.kind == "cross":
            v += f" (≥ {fmt_size(cross.size, unit, approx=True)})"
        sv = scale_verdicts(c, meta.get("scales"))
        if sv:
            v += f"; {sv}"
        concept = meta.get("concept")
        counts = bool(meta.get("counts_for_points")) and meta.get("status") not in ("template", "synthetic")
        all_ok = int(c["n_correct"].sum()) == int(c["n_runs"].sum())
        if not counts:
            if meta.get("status") == "measured":
                why_not = "calibración" if concept is None else "variante de comparación"
            else:
                why_not = meta.get("status") or "sin entrada en kernels.json"
            pts, why = 0, "no suma (" + why_not + ")"
        elif errors_by_app.get(app) or errors_by_app.get(None):
            pts, why = 0, "0: incumple la validacion"
        elif not all_ok:
            pts, why = 0, "0: hay corridas incorrectas"
        else:
            pts, why = 1, "1"
            if concept in per_concept:
                per_concept[concept] += 1
        if meta.get("cpu_baselines"):
            best_txt += " (mejor CPU incluye " + ", ".join(meta["cpu_baselines"]) + ")"
        rows.append([app, concept or "—", ordinals.get(meta.get("ordinal"), "—"),
                     cross.text(unit), best_txt, v, why])
    rows.sort(key=lambda r: (r[1] == "—", r[1], r[0]))
    table = pd.DataFrame(rows, columns=["Aplicación", "Concepto (carpeta ec2)", "Aplicación del concepto",
                                        "Crossover (GPU e2e < mejor CPU)", "Mejor speedup e2e", "Veredicto", "Puntos"])
    capped = {c: min(3, n) for c, n in per_concept.items()}
    total = sum(capped.values())
    missing = [c for c, n in per_concept.items() if n == 0]
    return table, {"per_concept": capped, "total": total, "grade": grade(total), "missing_concepts": missing}


def display_path(p: str | Path) -> str:
    """Ruta relativa a la raiz del repo (sin la carpeta del usuario) si el archivo esta dentro."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ANALYSIS_DIR.parent).as_posix()
    except ValueError:
        return p.name


def run_command(env: dict, csv_path: str | Path, config: str = "config/ciudad.json") -> str:
    """Comando que produjo el CSV. Si el env.json no lo trae (schema 1 no lo guarda), se reconstruye."""
    run = env.get("run") or {}
    if run.get("command"):
        cmd = run["command"]
        if isinstance(cmd, list):
            cmd = " ".join(("./" + c if i == 0 and "/" not in c else c) for i, c in enumerate(cmd))
        return str(cmd)
    parts = ["./delivery_sim", run.get("config", config), "--headless", "--bench", Path(csv_path).name]
    if run.get("apps"):
        parts += ["--apps", ",".join(run["apps"])]
    if run.get("reps"):
        parts += ["--reps", str(run["reps"])]
    if run.get("thread_sweep"):
        parts += ["--threads", ",".join(str(t) for t in run["thread_sweep"])]
    if run.get("seed") is not None:
        parts += ["--seed", str(run["seed"])]
    return " ".join(parts)


def setup_table(env: dict) -> pd.DataFrame:
    cpu, gpu, build, cuda, run = (env.get(k) or {} for k in ("cpu", "gpu", "build", "cuda", "run"))
    topo = cpu_topology(env)
    rows = [
        ("CPU", cpu.get("model", "n/d")),
        ("Hilos lógicos (hardware_concurrency)", cpu.get("hardware_concurrency", "n/d")),
        ("Núcleos físicos (env.json, /proc/cpuinfo de WSL)", cpu.get("physical_cores", "n/d")),
        ("Núcleos P / E (fabricante)", f"{topo['p_cores']} P + {topo['e_cores']} E = {topo['p_cores'] + topo['e_cores']} "
         f"({topo['source']})" if topo else "n/d"),
        ("Memoria visible", f"{env.get('memory_total_kb', 0) / 1024 / 1024:.1f} GiB" if env.get("memory_total_kb") else "n/d"),
        ("Sistema operativo", env.get("os", "n/d")),
        ("GPU", gpu.get("name", "no disponible")),
        ("Compute capability / SMs", f"{gpu.get('compute_capability', 'n/d')} / {gpu.get('sm_count', 'n/d')}"),
        ("Memoria de la GPU", f"{gpu.get('memory_bytes', 0) / 2**30:.2f} GiB" if gpu.get("memory_bytes") else "n/d"),
        ("Reloj SM (máx.)", f"{gpu.get('sm_clock_khz', 0) / 1e6:.3f} GHz" if gpu.get("sm_clock_khz") else "n/d"),
        ("Memoria: reloj / bus", f"{gpu.get('memory_clock_khz', 0) / 1e6:.3f} GHz / {gpu.get('bus_width_bits', 'n/d')} bits"
         if gpu.get("memory_clock_khz") else "n/d"),
        ("Ancho de banda pico teórico", f"{gpu.get('peak_bandwidth_gbs', float('nan')):.1f} GB/s" if gpu.get("peak_bandwidth_gbs") else "n/d"),
        ("CUDA runtime / driver NVIDIA (CUDA máx. del driver)",
         f"{cuda.get('runtime', 'n/d')} / {cuda.get('driver', 'n/d')} ({cuda.get('driver_max_cuda', 'n/d')})" if cuda else "n/d"),
        ("Compilador", build.get("compiler", "n/d")),
        ("Flags C++", build.get("cxx_flags", "n/d")),
        ("Flags CUDA", build.get("cuda_flags", "n/d")),
        ("Build / sanitizer", f"{build.get('type', 'n/d')} / {build.get('sanitizer', 'n/d')}"),
        ("Commit", build.get("git_commit", "n/d")),
        ("Semilla / repeticiones", f"{run.get('seed', 'n/d')} / {run.get('reps', 'n/d')}"),
        ("Barrido de hilos", ", ".join(str(t) for t in run.get("thread_sweep", [])) or "n/d"),
    ]
    return pd.DataFrame(rows, columns=["Dato", "Valor"])


# --------------------------------------------------------------------------------------------------
# Graficos (paleta y especificaciones de la skill dataviz; modo claro, PNG para el informe)
# --------------------------------------------------------------------------------------------------

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Paleta categorica por defecto, en su orden fijo (validada: scripts/validate_palette.js, modo claro).
SERIES_COLOR = {"cpu1": "#2a78d6", "cpuN": "#eb6834", "gpu_kernel": "#1baf7a", "gpu_e2e": "#eda100"}
SERIES_MARKER = {"cpu1": "o", "cpuN": "s", "gpu_kernel": "^", "gpu_e2e": "D"}
COPY_COLOR = "#6250d6"   # violeta: copias H2D + D2H
OTHER_COLOR = "#c3c2b7"  # resto (lanzamiento, sincronizacion, host), con textura

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": TEXT2, "ytick.labelcolor": TEXT2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6,
    "lines.markersize": 6,
}


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return matplotlib, plt


def _log_formatter():
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"{v:g}")


def plot_time_vs_size(cmp: pd.DataFrame, app: str, unit: str | None = None, cross: Crossover | None = None):
    """(a) Tiempo vs tamano, log-log, las 4 versiones con barras min-max y el cruce marcado."""
    mpl, plt = _plt()
    c = cmp[cmp["application"] == app].sort_values("size")
    cross = cross or crossover(c["size"], c["best_cpu_ms"], c["gpu_e2e_ms"])
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ends = []
        for v in VERSIONS:
            rows = [(r["size"], r["_rows"][v]) for _, r in c.iterrows() if r["_rows"].get(v) is not None]
            if not rows:
                continue
            x = np.array([s for s, _ in rows], dtype=float)
            med = np.array([row["median"] for _, row in rows])
            lo = med - np.array([row["min"] for _, row in rows])
            hi = np.array([row["max"] for _, row in rows]) - med
            ax.errorbar(x, med, yerr=[lo, hi], color=SERIES_COLOR[v], marker=SERIES_MARKER[v],
                        markeredgecolor=SURFACE, markeredgewidth=1.0, capsize=3, elinewidth=0.9,
                        label=VERSION_LABELS[v], zorder=3)
            ends.append((x[-1], med[-1], VERSION_LABELS[v]))
        if c["best_cpu"].astype(str).str.contains(":").any():
            ax.plot(c["size"], c["best_cpu_ms"], color=MUTED, linestyle="--", linewidth=1.4, zorder=2,
                    label="mejor CPU (incluye " + ", ".join(sorted({b.split(":")[0] for b in c["best_cpu"] if ":" in b})) + ")")
            ends.append((c["size"].iloc[-1], c["best_cpu_ms"].iloc[-1], "mejor CPU"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(_log_formatter())
        sizes = c["size"].tolist()
        ax.set_xticks(sizes)
        ax.set_xticklabels([fmt_size(s, unit) for s in sizes], rotation=0)
        ax.minorticks_off()
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        if sizes:
            ax.set_xlim(sizes[0] / 1.6, sizes[-1] * 9)
        # Etiquetas directas al final de cada serie, separadas para que no se pisen (texto en tinta, no en color).
        if ends:
            y0, y1 = ax.get_ylim()
            gap = 0.07 * (math.log10(y1) - math.log10(y0))
            ends.sort(key=lambda e: e[1])
            placed = []
            for x_end, y_end, label in ends:
                p = math.log10(y_end)
                if placed and p < placed[-1] + gap:
                    p = placed[-1] + gap
                placed.append(p)
                ax.text(x_end * 1.5, 10 ** p, label, va="center", ha="left", fontsize=8, color=TEXT2)
        if cross.kind == "cross":
            ax.axvline(cross.size, color=TEXT2, linestyle="--", linewidth=1.0, zorder=2)
            ax.text(cross.size * 1.08, 0.97, f"cruce ≈ {fmt_size(cross.size, unit, approx=True)}",
                    transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=8.5, color=TEXT)
        else:
            ax.text(0.01, 0.97, cross.text(unit), transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=TEXT2)
        ax.set_xlabel(f"Tamaño del problema ({unit})" if unit else "Tamaño del problema")
        ax.set_ylabel("Tiempo (ms, mediana; barras = mín–máx)")
        ax.set_title(f"{app}: tiempo vs tamaño (log–log)")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, handletextpad=0.4, columnspacing=1.2)
        fig.tight_layout()
    return fig


def plot_gpu_split(cmp: pd.DataFrame, app: str, unit: str | None = None):
    """(b) Reparto del tiempo GPU extremo a extremo en copias (H2D + D2H), kernel y resto, por tamano."""
    mpl, plt = _plt()
    c = cmp[cmp["application"] == app].sort_values("size")
    rows = []
    for _, r in c.iterrows():
        e = r["_rows"].get("gpu_e2e")
        if e is None:
            continue
        total = float(e["median"])
        copies = np.nansum([e["h2d_med"], e["d2h_med"]])
        kernel = 0.0 if pd.isna(e["kernel_med"]) else float(e["kernel_med"])
        other = max(0.0, total - copies - kernel)
        denom = max(total, copies + kernel)
        rows.append((r["size"], total, copies / denom, kernel / denom, other / denom,
                     not (pd.isna(e["h2d_med"]) and pd.isna(e["d2h_med"]) and pd.isna(e["kernel_med"]))))
    if not rows or not any(r[5] for r in rows):
        return None
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        x = np.arange(len(rows))
        parts = [("Copias H2D + D2H", 2, COPY_COLOR, None), ("Kernel", 3, SERIES_COLOR["gpu_kernel"], None),
                 ("Resto (lanzamiento, sincronización, host)", 4, OTHER_COLOR, "///")]
        bottom = np.zeros(len(rows))
        for label, idx, color, hatch in parts:
            vals = np.array([r[idx] for r in rows]) * 100
            ax.bar(x, vals, bottom=bottom, width=0.62, color=color, edgecolor=SURFACE, linewidth=1.5,
                   hatch=hatch, label=label, zorder=3)
            for xi, b, v in zip(x, bottom, vals):
                if v >= 9:
                    ink = TEXT if color in (OTHER_COLOR, SERIES_COLOR["gpu_kernel"]) else "#ffffff"
                    ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color=ink, zorder=4)
            bottom += vals
        for xi, r in zip(x, rows):
            ax.text(xi, 101.5, f"{fmt_num(r[1])} ms", ha="center", va="bottom", fontsize=7.5, color=TEXT2)
        ax.set_xticks(x)
        ax.set_xticklabels([fmt_size(r[0], unit) for r in rows])
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("% del tiempo GPU extremo a extremo")
        ax.set_xlabel(f"Tamaño del problema ({unit})" if unit else "Tamaño del problema")
        ax.set_title(f"{app}: copias vs kernel en la GPU (sobre cada barra, la mediana e2e)")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3)
        fig.tight_layout()
    return fig


def plot_thread_sweep(summary: pd.DataFrame, app: str, env: dict, unit: str | None = None, size: int | None = None):
    """(c) Speedup de cpuN contra cpu1 vs hilos, con los nucleos P, E y logicos marcados."""
    mpl, plt = _plt()
    size, sw = thread_sweep(summary, app, size)
    if sw is None or sw.empty:
        return None
    topo = cpu_topology(env) or {}
    hc = int((env.get("cpu") or {}).get("hardware_concurrency") or sw["threads"].max())
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        t = sw["threads"].to_numpy()
        tmax = max(int(t.max()), hc)
        ideal = np.arange(1, tmax + 1)
        log_y = float(sw["speedup"].min()) < 0.5
        ax.plot(ideal, ideal, color=MUTED, linestyle="--", linewidth=1.0, zorder=2, label="ideal (speedup = hilos)")
        ax.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
        ax.errorbar(t, sw["speedup"], yerr=[sw["speedup"] - sw["speedup_lo"], sw["speedup_hi"] - sw["speedup"]],
                    color=SERIES_COLOR["cpuN"], marker=SERIES_MARKER["cpuN"], markeredgecolor=SURFACE,
                    capsize=3, elinewidth=0.9, zorder=3, label="cpuN medido (mediana, barras mín/máx)")
        best = sw.loc[sw["speedup"].idxmax()]
        nb = int(best["threads"])
        ax.annotate(f"máx {fmt_x(best['speedup'])} con {nb} {'hilo' if nb == 1 else 'hilos'}",
                    (best["threads"], best["speedup"]), textcoords="offset points", xytext=(6, 8),
                    fontsize=8, color=TEXT)
        marks = []
        if topo:
            marks = [(topo["p_cores"], f"{topo['p_cores']} P"),
                     (topo["p_cores"] + topo["e_cores"], f"+{topo['e_cores']} E = {topo['p_cores'] + topo['e_cores']} físicos"),
                     (topo["logical"], f"{topo['logical']} lógicos")]
        else:
            marks = [(hc, f"{hc} lógicos")]
        for xv, label in marks:
            ax.axvline(xv, color=MUTED, linestyle=":", linewidth=1.0, zorder=1)
            ax.text(xv + 0.12, 0.985, label, transform=ax.get_xaxis_transform(), va="top", ha="left",
                    fontsize=7.5, color=TEXT2)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(_log_formatter())
        else:
            ax.set_ylim(0, max(tmax, float(sw["speedup_hi"].max())) * 1.08)
        ax.set_xlim(0.5, tmax + 2.5)
        ax.set_xticks(sorted(set(t.tolist()) | {hc}))
        ax.set_xlabel("Hilos del pool (cpuN)")
        ax.set_ylabel("Speedup vs CPU 1 hilo (mediana; barras con mín/máx)")
        ax.set_title(f"{app}: barrido de hilos en {fmt_size(size, unit)}" + (f" {unit}" if unit and unit != "bytes" else ""))
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
        fig.tight_layout()
    return fig


def plot_roofline(summary: pd.DataFrame, env: dict, kernels: dict, points: pd.DataFrame | None = None):
    """Roofline unico: techos de ancho de banda (teorico y medido) y de computo (FP32 e INT32)."""
    mpl, plt = _plt()
    peaks = gpu_peaks(env)
    if not peaks:
        return None
    bw = measured_bandwidth(summary)
    points = roofline_points(summary, kernels) if points is None else points
    fp32, int32, bw_peak = peaks["fp32_gflops"], peaks["int32_gops"], peaks["bw_peak_gbs"]
    bw_meas = bw["gbs"] if bw else None
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ai_min = 1e-2
        ai_max = 1e3
        if not points.empty:
            ai_min = min(ai_min, points["ai"].min() / 3)
            ai_max = max(ai_max, points["ai"].max() * 3)
        x = np.logspace(math.log10(ai_min), math.log10(ai_max), 400)
        ax.plot(x, np.minimum(x * bw_peak, fp32), color=MUTED, linestyle="--", linewidth=1.2, zorder=2,
                label=f"memoria: pico teórico {bw_peak:.0f} GB/s")
        if bw_meas:
            ax.plot(x, np.minimum(x * bw_meas, fp32), color=TEXT, linewidth=1.8, zorder=2,
                    label=f"memoria: medido {bw_meas:.0f} GB/s ({bw_meas / bw_peak:.0%}; calib_copy {fmt_size(bw['size'], 'bytes')})")
        ax.axhline(int32, color=TEXT2, linestyle=":", linewidth=1.2, zorder=2,
                   label=f"INT32 {int32 / 1000:.2f} TOP/s ({peaks['sms']} SM × {peaks['int32_lanes_per_sm']} × "
                         f"{peaks['clock_ghz']:.3f} GHz): kernels enteros")
        ridge = fp32 / (bw_meas or bw_peak)
        ax.text(ai_max / 1.1, fp32 / 1.2, f"pico FP32 {fp32 / 1000:.2f} TFLOP/s\n"
                f"({peaks['sms']} SM × {peaks['fp32_lanes_per_sm']} × 2 × {peaks['clock_ghz']:.3f} GHz)",
                ha="right", va="top", fontsize=8, color=TEXT)
        ax.axvline(ridge, color=GRID, linewidth=1.0, zorder=1)
        ax.text(ridge * 1.08, 0.45, f"equilibrio {ridge:.0f} op/B", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=TEXT2, rotation=90, va="bottom")
        if points.empty:
            ax.text(0.98, 0.04, "Sin kernels con operaciones aritméticas en este CSV\n(solo calibración: da el techo medido)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=TEXT2)
        else:
            # Solo las aplicaciones principales (sin variantes) y su mayor tamano; etiqueta corta = numero de concepto.
            mains = [a for a in points["application"].unique() if is_main_app(kernels, a)]
            legend_lines = []
            for app in mains:
                g = points[points["application"] == app].sort_values("size")
                hollow = g["op_type"].iloc[0] != "fp32"
                last = g.iloc[-1]
                concept = app_meta(kernels, app).get("concept") or ""
                code = concept[:2] if concept[:2].isdigit() else app[:6]
                ax.scatter([last["ai"]], [last["gops"]], s=52, zorder=5, linewidths=1.6,
                           facecolors=SURFACE if hollow else SERIES_COLOR["cpu1"], edgecolors=SERIES_COLOR["cpu1"])
                offset = {"12": (-16, -13), "09": (7, -11)}.get(code, (6, 4))  # evita que 09 y 12 se pisen
                ax.annotate(code, (last["ai"], last["gops"]), textcoords="offset points", xytext=offset, fontsize=8.5,
                            color=TEXT, fontweight="semibold")
                legend_lines.append(f"{code} {app}: {fmt_num(last['gops'])} G{'OP' if hollow else 'FLOP'}/s")
            ax.text(0.99, 0.02, "\n".join(sorted(legend_lines)), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color=TEXT2, family="monospace")
            ax.scatter([], [], s=52, color=SERIES_COLOR["cpu1"], label="kernel en su mayor tamaño (relleno = FP32, hueco = entero)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(_log_formatter())
        ax.yaxis.set_major_formatter(_log_formatter())
        ax.set_xlim(ai_min, ai_max)
        ymin = 1.0
        if not points.empty:
            ymin = min(ymin, points["gops"].min() / 3)
        ax.set_ylim(ymin, fp32 * 2.2)
        ax.set_xlabel("Intensidad aritmética (operaciones por byte de memoria global)")
        ax.set_ylabel("Rendimiento (GFLOP/s o GOP/s)")
        ax.set_title(f"Roofline: {peaks['name']} (cc {peaks['cc']})")
        ax.legend(loc="upper left", fontsize=7.8, handlelength=2.6)
        fig.tight_layout()
    return fig


def pstate_log_for(csv_paths) -> Path | None:
    """Registro de nvidia-smi de la corrida: gpu_pstate.csv junto al CSV o pstate_sesion.csv en la carpeta de arriba."""
    for csv_path in csv_paths:
        for candidate in (Path(csv_path).parent / "gpu_pstate.csv", Path(csv_path).parent.parent / "pstate_sesion.csv"):
            if candidate.exists():
                return candidate
    return None


def load_pstate_log(path: str | Path) -> pd.DataFrame:
    """Lee el registro de `nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,... --format=csv -lms 500`."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.split(" [")[0].strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["t_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    df["sm_mhz"] = pd.to_numeric(df["clocks.current.sm"].astype(str).str.replace(" MHz", ""), errors="coerce")
    df["pstate"] = df["pstate"].astype(str).str.strip()
    return df


def plot_gpu_pstate(df: pd.DataFrame, title: str = "Estado P y reloj SM de la GPU durante la corrida"):
    """Reloj SM vs tiempo; los tramos en P0 (rendimiento maximo) se sombrean."""
    mpl, plt = _plt()
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        ax.step(df["t_s"], df["sm_mhz"], where="post", color=SERIES_COLOR["gpu_kernel"], linewidth=1.4)
        p0 = (df["pstate"] == "P0").to_numpy()
        t = df["t_s"].to_numpy()
        for i in range(len(t) - 1):
            if p0[i]:
                ax.axvspan(t[i], t[i + 1], color=SERIES_COLOR["gpu_kernel"], alpha=0.12, linewidth=0)
        counts = df["pstate"].value_counts()
        ax.text(0.01, 0.97, "muestras: " + ", ".join(f"{k} = {v}" for k, v in counts.items()) + " (sombreado = P0)",
                transform=ax.transAxes, va="top", fontsize=8, color=TEXT2)
        ax.set_xlabel("Tiempo desde el inicio (s)")
        ax.set_ylabel("Reloj SM (MHz)")
        ax.set_ylim(0, max(3200, float(df["sm_mhz"].max()) * 1.1))
        ax.set_title(title)
        fig.tight_layout()
    return fig


def save(fig, path: str | Path) -> Path | None:
    if fig is None:
        return None
    _, plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_figures(ds: Dataset, kernels: dict, fig_dir: str | Path) -> dict:
    """Genera todas las figuras. Devuelve {app: {"tiempo": path, "reparto": path, "hilos": path}, "_roofline": path}."""
    fig_dir = Path(fig_dir)
    summ = summarize(ds.df)
    cmp = compare_all(summ, kernels)
    out = {}
    for app in ds.apps:
        unit = app_meta(kernels, app).get("size_unit")
        out[app] = {
            "tiempo": save(plot_time_vs_size(cmp, app, unit), fig_dir / f"{app}_tiempo_vs_tamano.png"),
            "reparto": save(plot_gpu_split(cmp, app, unit), fig_dir / f"{app}_reparto_gpu.png"),
            "hilos": save(plot_thread_sweep(summ, app, ds.env, unit), fig_dir / f"{app}_barrido_hilos.png"),
        }
    out["_roofline"] = save(plot_roofline(summ, ds.env, kernels), fig_dir / "roofline.png")
    log = pstate_log_for(ds.csv_paths)
    if log is not None:
        out["_pstate"] = save(plot_gpu_pstate(load_pstate_log(log)), fig_dir / "gpu_pstate.png")
    return out


# --------------------------------------------------------------------------------------------------
# Informe en Markdown
# --------------------------------------------------------------------------------------------------

def render_report(ds: Dataset, kernels: dict, figures: dict | None, md_path: str | Path,
                  config: str = "config/ciudad.json") -> str:
    md_dir = Path(md_path).resolve().parent
    summ = summarize(ds.df)
    cmp = compare_all(summ, kernels)

    def rel(p):
        if p is None:
            return None
        try:
            return Path(p).resolve().relative_to(md_dir).as_posix()
        except ValueError:
            return Path(p).resolve().as_posix()

    L = ["# Informe de benchmarking: tablas y figuras (generado)", ""]
    L.append("Generado por `analysis/generar_informe.py` (o `analysis/informe.ipynb`) con `analysis/report_lib.py`.")
    L.append("")
    for p in ds.csv_paths:
        L.append(f"- CSV: `{display_path(p)}` ({len(ds.df)} filas en total)")
        L.append(f"- Comando que lo produjo (reconstruido del env.json): `{run_command(ds.env, p, config)}`")
    if ds.env.get("synthetic"):
        L.append("- **Datos SINTÉTICOS** (solo para probar el pipeline; no son mediciones).")
    L.append("")
    L.append("## Validación (sección 2.2)")
    L.append("")
    if ds.issues:
        for i in ds.issues:
            L.append(f"- {i}")
    else:
        L.append(f"- Sin incumplimientos: columnas exactas, ninguna fila `correct=no`, ≥ {MIN_REPS} repeticiones por grupo, "
                 f"≥ {MIN_SIZES} tamaños y ≥ {MIN_DECADES:g} órdenes de magnitud por aplicación, build Release.")
    L.append("")
    L.append("## Set-up (del env.json)")
    L.append("")
    L.append(markdown_table(setup_table(ds.env)))
    L.append("")
    L.append("Máquina única: CPU y GPU se midieron en la misma laptop (el enunciado, 2.3, admite \"your own computer if it "
             "has an NVIDIA GPU\"). Donde 2.2 y 2.4 dicen Colab, se usa esta máquina; por eso la tabla \"same-machine\" "
             "de 2.4 coincide con la tabla de cada aplicación.")
    L.append("")
    peaks = gpu_peaks(ds.env)
    bw = measured_bandwidth(summ)
    if peaks:
        L.append("## Techos de la GPU (roofline)")
        L.append("")
        L.append(f"- FP32: {peaks['sms']} SM × {peaks['fp32_lanes_per_sm']} núcleos × 2 (FMA) × {peaks['clock_ghz']:.3f} GHz "
                 f"= **{peaks['fp32_gflops']:.0f} GFLOP/s**")
        L.append(f"- INT32: {peaks['sms']} SM × {peaks['int32_lanes_per_sm']} op/ciclo × {peaks['clock_ghz']:.3f} GHz "
                 f"= **{peaks['int32_gops']:.0f} GOP/s** (techo para kernels enteros: histograma, scan, radix...)")
        L.append(f"- Ancho de banda pico teórico (2 × reloj de memoria × bus / 8): **{peaks['bw_peak_gbs']:.1f} GB/s**")
        if bw:
            L.append(f"- Ancho de banda medido (calib_copy gpu_kernel, {fmt_size(bw['size'], 'bytes')}, 2 × bytes / tiempo): "
                     f"**{bw['gbs']:.1f} GB/s** ({bw['gbs'] / peaks['bw_peak_gbs']:.0%} del pico)")
            t = bw["per_size"]
            L.append("")
            L.append(markdown_table(pd.DataFrame({
                "Size": [fmt_size(s, "bytes") for s in t["size"]],
                "Kernel (ms)": [fmt_num(v) for v in t["kernel_ms"]],
                "GB/s (mediana)": [f"{v:.1f}" for v in t["gbs"]],
                "GB/s (mejor corrida)": [f"{v:.1f}" for v in t["gbs_best"]],
                "Nota": ["por encima del pico de DRAM: el buffer cabe en la caché L2" if v > peaks["bw_peak_gbs"] else ""
                         for v in t["gbs"]],
            })))
        L.append("")
    table, totals = summary_table(ds, kernels)
    L.append("## Resumen (sección 2.6, punto 1)")
    L.append("")
    L.append(markdown_table(table))
    L.append("")
    line = ("Puntos por concepto (máx. 3): " + ", ".join(f"{c} = {n}" for c, n in totals["per_concept"].items())
            + f"; total **{totals['total']}** → nota **{totals['grade']:.1f}/100** (50 + 50 (p − 10) / 20 con p ≥ 10; "
            "50 p / 10 si no).")
    if totals["missing_concepts"]:
        line += f" Conceptos sin aplicación medida (mínimo no cumplido): {', '.join(totals['missing_concepts'])}."
    L.append(line)
    L.append("")
    for app in ds.apps:
        meta = app_meta(kernels, app)
        unit = meta.get("size_unit")
        L.append(f"## {app}" + (f": {meta['title']}" if meta.get("title") else ""))
        L.append("")
        if meta.get("concept"):
            L.append(f"Concepto: `{meta['concept']}`" + (f" ({meta.get('ordinal')}ª aplicación)" if meta.get("ordinal") else ""))
            L.append("")
        if unit:
            L.append(f"Size = {meta.get('size_meaning', unit)}.")
            L.append("")
        tbl, note = app_table(cmp, app, unit)
        L.append("### Tabla (sección 2.4)")
        L.append("")
        L.append(markdown_table(tbl))
        L.append("")
        L.append(note)
        L.append("")
        cross = app_crossover(cmp, app)
        cross_k = app_crossover(cmp, app, "gpu_kernel_ms")
        L.append(f"- Crossover extremo a extremo (GPU e2e < mejor CPU): {cross.text(unit)} → veredicto: **{verdict(cross)}**.")
        L.append(f"- Crossover solo kernel (datos ya en la GPU): {cross_k.text(unit)}.")
        L.append("")
        L.append("### Dispersión (mediana, mín, máx, IQR)")
        L.append("")
        L.append(markdown_table(dispersion_table(cmp, app, unit)))
        L.append("")
        size, sw = thread_sweep(summ, app)
        if sw is not None and not sw.empty:
            L.append(f"### Barrido de hilos ({fmt_size(size, unit)})")
            L.append("")
            L.append(markdown_table(sweep_table(sw)))
            L.append("")
        figs = (figures or {}).get(app, {})
        for key, title in [("tiempo", "Tiempo vs tamaño"), ("reparto", "Copias vs kernel"), ("hilos", "Barrido de hilos")]:
            if figs.get(key):
                L.append(f"![{app}: {title}]({rel(figs[key])})")
                L.append("")
    L.append("## Roofline")
    L.append("")
    if figures and figures.get("_roofline"):
        L.append(f"![Roofline]({rel(figures['_roofline'])})")
        L.append("")
    pts = roofline_points(summ, kernels)
    if pts.empty:
        L.append("Ningún kernel del CSV tiene operaciones aritméticas en `kernels.json` (solo calibración): la figura "
                 "muestra los techos; los kernels de las aplicaciones se ubican cuando estén en el CSV.")
    else:
        L.append(markdown_table(pd.DataFrame({
            "Aplicación": pts["application"], "Size": pts["size"], "Ops": [fmt_num(v) for v in pts["ops"]],
            "Bytes": [fmt_num(v) for v in pts["bytes"]], "Intensidad (op/B)": [fmt_num(v) for v in pts["ai"]],
            "Kernel (ms)": [fmt_num(v) for v in pts["kernel_ms"]], "Logrado (GOP/s)": [fmt_num(v) for v in pts["gops"]],
            "Tráfico (GB/s)": [fmt_num(v) for v in pts["gbs"]],
        })))
    L.append("")
    L.append("### Derivación de operaciones y bytes (de `kernels.json`)")
    L.append("")
    for app in ds.apps:
        meta = app_meta(kernels, app)
        if meta:
            L.append(f"- **{app}**: ops = `{meta.get('ops')}`, bytes = `{meta.get('bytes')}` "
                     f"({meta.get('op_type', 'n/d')}). {meta.get('derivation', '')}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------------------------------
# Efecto en la simulacion completa (tools/sim_compare.sh) y robustez (corrida con 30 repeticiones)
# --------------------------------------------------------------------------------------------------

SIM_METRICS = [
    ("latency_mean", "Latencia de despacho: media (ms)"), ("latency_p50", "Latencia p50 (ms)"),
    ("latency_p90", "Latencia p90 (ms)"), ("latency_p99", "Latencia p99 (ms)"), ("latency_max", "Latencia máx. (ms)"),
    ("orders_per_sim_hour", "Pedidos servidos por hora simulada"), ("orders_created", "Pedidos creados"),
    ("orders_assigned", "Pedidos asignados"), ("orders_delivered", "Pedidos entregados"),
    ("orders_rejected", "Pedidos rechazados"), ("fps_mean", "Cuadros por segundo (media)"),
    ("draw_p50_ms", "Tiempo de dibujo p50 (ms)"), ("draw_p95_ms", "Tiempo de dibujo p95 (ms)"),
]
SIM_BACKENDS = {"cpu": "--backend cpu", "auto": "--backend auto (GPU)"}


def load_sim_compare(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def sim_table(df: pd.DataFrame) -> pd.DataFrame:
    """Media de las corridas por backend y, entre parentesis, el minimo y el maximo."""
    rows = []
    for col, label in SIM_METRICS:
        row = [label]
        for backend in SIM_BACKENDS:
            v = df[df["backend"] == backend][col]
            row.append(f"{fmt_num(v.mean())} ({fmt_num(v.min())}–{fmt_num(v.max())})" if len(v) else "n/d")
        cpu, gpu = df[df["backend"] == "cpu"][col].mean(), df[df["backend"] == "auto"][col].mean()
        row.append(f"{(gpu / cpu - 1):+.0%}" if cpu else "n/d")
        rows.append(row)
    ok = df.groupby("backend")["validator_ok"].apply(lambda s: f"{int((s == 'yes').sum())}/{len(s)}")
    rows.append(["Corridas que validan (tools/validate_log.py)"] + [ok.get(b, "n/d") for b in SIM_BACKENDS] + [""])
    return pd.DataFrame(rows, columns=["Métrica (media de las corridas; mín–máx)"] + list(SIM_BACKENDS.values())
                        + ["GPU vs CPU"])


def plot_sim_latency(df: pd.DataFrame):
    """Percentiles de la latencia de despacho por backend; un punto por corrida y la media marcada."""
    mpl, plt = _plt()
    metrics = [("latency_p50", "p50"), ("latency_mean", "media"), ("latency_p90", "p90"), ("latency_p99", "p99")]
    colors = {"cpu": SERIES_COLOR["cpu1"], "auto": SERIES_COLOR["gpu_e2e"]}
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 4.0))
        x = np.arange(len(metrics))
        for offset, backend in [(-0.15, "cpu"), (0.15, "auto")]:
            g = df[df["backend"] == backend]
            means = [g[m].mean() for m, _ in metrics]
            for xi, (m, _) in zip(x, metrics):
                ax.scatter(np.full(len(g), xi + offset), g[m], s=22, color=colors[backend], alpha=0.55, zorder=3,
                           edgecolors=SURFACE, linewidths=0.8)
            ax.scatter(x + offset, means, marker="_", s=420, color=colors[backend], linewidths=2.2, zorder=4,
                       label=f"{SIM_BACKENDS[backend]} (raya = media; puntos = corridas)")
        ax.set_xticks(x)
        ax.set_xticklabels([name for _, name in metrics])
        ax.set_ylabel("Latencia de despacho por pedido (ms)")
        ax.set_title("Simulación completa (ciudad, 30 s reales): latencia de despacho")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, None)
        ax.legend(loc="upper left")
        fig.tight_layout()
    return fig


def robustness(base: Dataset, other: Dataset, kernels: dict) -> tuple[pd.DataFrame, dict]:
    """Compara dos corridas (p. ej. 10 y 30 repeticiones): quien gana (mejor CPU o GPU e2e) en cada punto y
    la diferencia relativa de las medianas."""
    a = compare_all(summarize(base.df), kernels).set_index(["application", "size"])
    b = compare_all(summarize(other.df), kernels).set_index(["application", "size"])
    common = sorted(set(a.index) & set(b.index))
    rows, flips, points, diffs = [], 0, 0, []
    for key in common:
        ra, rb = a.loc[key], b.loc[key]
        if not (np.isfinite(ra["gpu_e2e_ms"]) and np.isfinite(rb["gpu_e2e_ms"])):
            continue
        points += 1
        win_a = "GPU" if ra["gpu_e2e_ms"] < ra["best_cpu_ms"] else "CPU"
        win_b = "GPU" if rb["gpu_e2e_ms"] < rb["best_cpu_ms"] else "CPU"
        flips += win_a != win_b
        for col in ("best_cpu_ms", "gpu_kernel_ms", "gpu_e2e_ms"):
            if np.isfinite(ra[col]) and np.isfinite(rb[col]) and ra[col] >= 1.0:
                diffs.append(abs(rb[col] / ra[col] - 1))
        if win_a != win_b:
            rows.append([key[0], key[1], win_a, win_b])
    stats = {"points": points, "flips": flips,
             "diff_median": float(np.median(diffs)) if diffs else np.nan,
             "diff_p25": float(np.percentile(diffs, 25)) if diffs else np.nan,
             "diff_p75": float(np.percentile(diffs, 75)) if diffs else np.nan, "n_diffs": len(diffs)}
    return pd.DataFrame(rows, columns=["Aplicación", "Size", "Ganador (10 rep.)", "Ganador (30 rep.)"]), stats
