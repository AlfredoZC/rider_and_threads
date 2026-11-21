"""Resumen de los perfiles de Nsight Compute (ncu) y Nsight Systems (nsys) de analysis/data/final/perfiles/.

ncu:  <app>_ncu.ncu-rep exportado con `ncu --import <rep> --page details --csv` a <app>_ncu_details.csv.
nsys: <app>_nsys.sqlite (`nsys export --type sqlite`). En WSL2 esta version de nsys registra la API de CUDA del hilo
      de CPU pero no la actividad del lado de la GPU (kernels y copias), asi que el reparto se lee desde la API:
      cudaMemcpy (sincronica con memoria paginable) = copia; cudaEventSynchronize / cudaDeviceSynchronize = espera a
      los kernels; cudaLaunchKernel = costo de lanzar.
Uso: C:\\Python313\\python.exe analysis\\perfiles.py   (escribe perfiles/resumen_perfiles.md)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROFILE_DIR = HERE / "data" / "final" / "perfiles"
sys.path.insert(0, str(HERE))

import report_lib as rl  # noqa: E402

# (aplicacion, tamano, kernel perfilado), en el orden de las secciones del informe
PROFILED = [
    ("apsp_floyd", 3741, "floydWarshallPhase3"), ("eta_batch_soa", 5000, "computeEtaSoA"),
    ("eta_batch_aos", 5000, "computeEtaAoS"), ("demand_heatmap", 2048, "smoothDemandKernel"),
    ("demand_heatmap_tiles", 2048, "smoothDemandTiledKernel"), ("coverage_field", 1024, "coverageJacobiStep"),
    ("delivery_histogram", 10_000_000, "histogramPrivatizedKernel"),
    ("histogram_naive", 10_000_000, "histogramGlobalAtomicsKernel"), ("best_courier", 5000, "bestCourierConvergent"),
    ("best_courier_divergent", 5000, "bestCourierDivergent"), ("orders_in_progress", 10_000_000, "scanBlockKoggeStone"),
    ("candidate_filter", 200_000, "filterCandidatesBlockPrivate"),
    ("candidate_filter_naive", 200_000, "filterCandidatesNaive"), ("delivery_timeline", 10_000_000, "mergeRound"),
    ("delivery_percentiles", 10_000_000, "radixScatter"),
]


def ncu_metrics(app: str, directory: Path = PROFILE_DIR) -> dict | None:
    path = directory / f"{app}_ncu_details.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["Metric Name"].notna()]
    if df.empty:
        return None

    def value(name, section=None):
        rows = df[df["Metric Name"] == name]
        if section:
            rows = rows[rows["Section Name"] == section]
        if rows.empty:
            return None
        raw = str(rows["Metric Value"].iloc[0]).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            return raw

    limits = {key: value(f"Block Limit {key}") for key in ("Registers", "Shared Mem", "Warps", "SM")}
    numeric = {k: v for k, v in limits.items() if isinstance(v, float)}
    lowest = min(numeric.values()) if numeric else None
    limiter = ", ".join(k for k, v in numeric.items() if v == lowest) if numeric else "n/d"
    active_threads = value("Avg. Active Threads Per Warp")
    not_predicated = value("Avg. Not Predicated Off Threads Per Warp")
    return {
        "kernel": str(df["Kernel Name"].iloc[0]), "block": str(df["Block Size"].iloc[0]), "grid": str(df["Grid Size"].iloc[0]),
        "duration_ms": (value("Duration") or 0) * {"ms": 1, "us": 1e-3, "usecond": 1e-3, "msecond": 1, "ns": 1e-6}.get(
            str(df[df["Metric Name"] == "Duration"]["Metric Unit"].iloc[0]), 1),
        "sm_ghz": value("SM Frequency"), "sm_pct": value("Compute (SM) Throughput"),
        "mem_pct": value("Memory Throughput", "GPU Speed Of Light Throughput"), "dram_pct": value("DRAM Throughput"),
        "l1_pct": value("L1/TEX Cache Throughput"), "l2_pct": value("L2 Cache Throughput"),
        "mem_gbs": value("Memory Throughput", "Memory Workload Analysis"), "l2_hit": value("L2 Hit Rate"),
        "theo_occ": value("Theoretical Occupancy"), "ach_occ": value("Achieved Occupancy"),
        "limiter": limiter, "limits": limits, "registers": value("Registers Per Thread"),
        "warp_eff": active_threads / 32 if isinstance(active_threads, float) else None,
        "pred_eff": not_predicated / 32 if isinstance(not_predicated, float) else None,
        "branch_eff": value("Branch Efficiency"), "divergent": value("Avg. Divergent Branches"),
        "cycles_per_issue": value("Warp Cycles Per Issued Instruction"),
    }


def nsys_api(app: str, directory: Path = PROFILE_DIR) -> pd.DataFrame | None:
    """Llamadas a la API de CUDA del proceso (nombre, inicio y fin en ns)."""
    path = directory / f"{app}_nsys.sqlite"
    if not path.exists():
        return None
    with sqlite3.connect(path) as con:
        df = pd.read_sql_query(
            "SELECT r.start, r.end, s.value AS name, r.globalTid AS tid FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id = r.nameId", con)
    df["name"] = df["name"].str.replace(r"_v\d+$", "", regex=True)
    df["ms"] = (df["end"] - df["start"]) / 1e6
    return df


def api_split(api: pd.DataFrame) -> dict:
    """Tiempo del hilo de CPU en la API de CUDA por categoria (toda la corrida de --bench-once)."""
    def total(pattern):
        rows = api[api["name"].str.contains(pattern)]
        return float(rows["ms"].sum()), int(len(rows))
    copy_ms, copies = total(r"^cudaMemcpy")
    sync_ms, syncs = total(r"Synchronize")
    launch_ms, launches = total(r"^cudaLaunchKernel")
    alloc_ms, allocs = total(r"^cudaMalloc|^cudaFree")
    return {"copy_ms": copy_ms, "copies": copies, "sync_ms": sync_ms, "syncs": syncs, "launch_ms": launch_ms,
            "launches": launches, "alloc_ms": alloc_ms, "allocs": allocs}


def fmt_pct(v) -> str:
    return f"{v:.0f} %" if isinstance(v, float) else "n/d"


def fmt_branch(m: dict) -> str:
    """ncu informa 0 % de eficiencia de saltos cuando no hay saltos divergentes que medir."""
    eff, divergent = m.get("branch_eff"), m.get("divergent")
    if isinstance(eff, float) and eff == 0 and isinstance(divergent, float) and divergent == 0:
        return "— (0 saltos divergentes)"
    return fmt_pct(eff) if isinstance(eff, float) else "n/d"


def profile_table(kernels: dict | None = None) -> pd.DataFrame:
    rows = []
    for app, size, _ in PROFILED:
        m = ncu_metrics(app)
        if not m:
            continue
        rows.append([app, f"{size:,}", f"{m['block']} × {m['grid']}", f"{rl.fmt_num(m['duration_ms'])}",
                     f"{fmt_pct(m['ach_occ'])} / {fmt_pct(m['theo_occ'])}", m["limiter"],
                     fmt_pct(m["sm_pct"]), fmt_pct(m["dram_pct"]), fmt_pct(m["l2_pct"]), fmt_pct(m["l1_pct"]),
                     fmt_pct(m["warp_eff"] * 100 if m["warp_eff"] is not None else None),
                     fmt_branch(m)])
    return pd.DataFrame(rows, columns=["Aplicación", "Size", "Bloque × grid", "Kernel (ms, ncu)",
                                       "Ocupación lograda / teórica", "Limitante de ocupación", "SM", "DRAM", "L2",
                                       "L1/TEX", "Eficiencia de warp", "Eficiencia de saltos"])


def app_profile_text(app: str) -> str:
    size, kernel = next(((s, k) for a, s, k in PROFILED if a == app), (None, None))
    m = ncu_metrics(app)
    api = nsys_api(app)
    if not m and api is None:
        return "Sin perfiles."
    lines = []
    if m:
        lines.append(
            f"- **ncu** (`{kernel}`, size {size:,}, bloque {m['block']}, grid {m['grid']}, {int(m['registers'] or 0)} "
            f"registros por hilo): ocupación lograda **{fmt_pct(m['ach_occ'])}** de {fmt_pct(m['theo_occ'])} teórica, "
            f"limitada por **{m['limiter']}** (bloques por SM: registros {m['limits']['Registers']:.0f}, shared "
            f"{m['limits']['Shared Mem']:.0f}, warps {m['limits']['Warps']:.0f}). Throughput: SM {fmt_pct(m['sm_pct'])}, "
            f"memoria {fmt_pct(m['mem_pct'])} (DRAM {fmt_pct(m['dram_pct'])}, L2 {fmt_pct(m['l2_pct'])}, L1/TEX "
            f"{fmt_pct(m['l1_pct'])}; {rl.fmt_num(m['mem_gbs'])} GB/s; acierto en L2 {fmt_pct(m['l2_hit'])}). "
            f"Eficiencia de warp {fmt_pct(m['warp_eff'] * 100 if m['warp_eff'] is not None else None)} de hilos activos "
            f"({fmt_pct(m['pred_eff'] * 100 if m['pred_eff'] is not None else None)} sin predicar), eficiencia de saltos "
            f"{fmt_branch(m)}, "
            f"{rl.fmt_num(m['cycles_per_issue'])} ciclos por instrucción emitida.")
    if api is not None and not api.empty:
        s = api_split(api)
        lines.append(
            f"- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): "
            f"{s['copies']} `cudaMemcpy` = {rl.fmt_num(s['copy_ms'])} ms; {s['launches']} lanzamientos = "
            f"{rl.fmt_num(s['launch_ms'])} ms; {s['syncs']} esperas (`*Synchronize`) = {rl.fmt_num(s['sync_ms'])} ms; "
            f"reservas y liberaciones = {rl.fmt_num(s['alloc_ms'])} ms.")
    lines.append(f"- Archivos versionados: `analysis/data/final/perfiles/{app}_ncu_details.csv`, `{app}_ncu.out`, "
                 f"`{app}_nsys.out` y `{app}_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) "
                 f"no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.")
    return "\n".join(lines)


def plot_api_timeline(app: str, title_size: str):
    """Linea de tiempo de la API de CUDA en el hilo de CPU durante la ultima medicion (copias, lanzamientos, esperas)."""
    import numpy as np
    mpl, plt = rl._plt()
    api = nsys_api(app)
    if api is None or api.empty:
        return None
    categories = [("cudaMemcpy", "Copias (cudaMemcpy)", "#6250d6"), ("LaunchKernel", "Lanzamientos", "#eb6834"),
                  ("Synchronize", "Esperas a la GPU (kernels)", "#1baf7a")]
    relevant = api[api["name"].str.contains("cudaMemcpy|LaunchKernel|Synchronize")].sort_values("start")
    if relevant.empty:
        return None
    # La ultima ida y vuelta: desde la penultima tanda de copias hasta el final.
    copies = relevant[relevant["name"].str.startswith("cudaMemcpy")]
    t_end = relevant["end"].max()
    t_start = copies["start"].iloc[-4] if len(copies) >= 4 else relevant["start"].min()
    window = relevant[relevant["start"] >= t_start]
    with mpl.rc_context(rl.STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 2.8))
        for row, (pattern, label, color) in enumerate(categories):
            part = window[window["name"].str.contains(pattern)]
            bars = [((s - t_start) / 1e6, max((e - s) / 1e6, (t_end - t_start) / 1e6 / 800)) for s, e in zip(part["start"], part["end"])]
            ax.broken_barh(bars, (row - 0.35, 0.7), facecolors=color, edgecolor=rl.SURFACE, linewidth=0.3)
            total = part["ms"].sum()
            ax.text(1.01, row, f"{rl.fmt_num(total)} ms ({len(part)})", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.5, color=rl.TEXT2)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels([c[1] for c in categories])
        ax.invert_yaxis()
        ax.set_xlabel("Tiempo (ms) desde el inicio de la última medición")
        ax.set_title(f"{app}, {title_size}: API de CUDA (nsys)")
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
    return fig


def main() -> int:
    lines = ["# Resumen de perfiles (ncu y nsys)", "", rl.markdown_table(profile_table()), ""]
    for app, size, _ in PROFILED:
        lines += [f"## {app}", "", app_profile_text(app), ""]
    out = PROFILE_DIR / "resumen_perfiles.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
