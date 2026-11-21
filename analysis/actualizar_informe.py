"""Rellena los bloques automaticos de docs/informe/informe.md con las tablas y figuras de report_lib.

Un bloque es  <!-- auto:CLAVE --> ... <!-- /auto:CLAVE -->  y se reemplaza entero en cada corrida; el texto fuera
de los bloques (el analisis escrito a mano) no se toca. Claves:
    fuentes, setup, techos, resumen, pstate, roofline, derivaciones,
    tabla:APP, cruce:APP, dispersion:APP, hilos:APP, figuras:APP
Uso (desde la raiz del repo):
    C:\\Python313\\python.exe analysis\\actualizar_informe.py            # todos los analysis/data/*/results.csv
    C:\\Python313\\python.exe analysis\\actualizar_informe.py --csv analysis/data/local_calib/results.csv
Las figuras se escriben en docs/informe/img/ (y, con --capturas DIR, se copian tambien alli).
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

import perfiles  # noqa: E402
import report_lib as rl  # noqa: E402

BLOCK = re.compile(r"(<!-- auto:(?P<key>[^ ]+?) -->)(?P<body>.*?)(<!-- /auto:(?P=key) -->)", re.S)


FINAL_DIR = HERE / "data" / "final"
R30_DIR = HERE / "data" / "final_r30"


def default_csvs() -> list[Path]:
    """Corrida final limpia (una carpeta por aplicacion con bench.csv + bench.csv.env.json)."""
    return sorted(FINAL_DIR.glob("*/bench.csv"))


def sim_blocks(img: Path, img_rel: str) -> dict[str, str]:
    path = FINAL_DIR / "sim_compare" / "sim_compare.csv"
    if not path.exists():
        return {}
    df = rl.load_sim_compare(path)
    rl.save(rl.plot_sim_latency(df), img / "sim_latencia_despacho.png")
    runs = df.groupby("backend").size()
    text = (rl.markdown_table(rl.sim_table(df)) + "\n\n"
            + f"{runs.get('auto', 0)} corridas con `--backend auto` (GPU) y {runs.get('cpu', 0)} con `--backend cpu`, "
            f"alternadas, de 30 s reales cada una en modo `{df['mode'].iloc[0]}`, escenario `config/ciudad.json`. "
            f"Fuente: `{rl.display_path(path)}` (comando `tools/sim_compare.sh`, que valida cada log con "
            "`tools/validate_log.py` y calcula las métricas con `tools/sim_metrics.py`).\n\n"
            f"![Latencia de despacho por backend]({img_rel}/sim_latencia_despacho.png)")
    return {"simulacion": text}


def profile_blocks(img: Path, img_rel: str) -> dict[str, str]:
    if not perfiles.PROFILE_DIR.exists():
        return {}
    blocks = {"perfiles": rl.markdown_table(perfiles.profile_table())}
    for app, _, _ in perfiles.PROFILED:
        blocks[f"perfil:{app}"] = perfiles.app_profile_text(app)
    figs = []
    for app, size in (("coverage_field", "1024 × 1024, 200 barridos"), ("delivery_timeline", "10 M entregas, 11 rondas")):
        path = rl.save(perfiles.plot_api_timeline(app, size), img / f"nsys_{app}.png")
        if path:
            figs.append(f"![nsys: {app}]({img_rel}/{path.name})")
    blocks["nsys"] = "\n\n".join(figs)
    return blocks


def correctness_block() -> dict[str, str]:
    """Resumen de los logs de la CI (compute-sanitizer, TSan, ASan, tests) copiados en final/correctitud/."""
    folder = FINAL_DIR / "correctitud"
    if not folder.exists():
        return {}
    pattern = re.compile(r"^OK\s+(memcheck|racecheck) (\S+) \(size (\d+)\): =+ (.*)$")
    rows = {}
    for name in ("gpu-memcheck.log", "gpu-racecheck.log"):
        for line in (folder / name).read_text(encoding="utf-8").splitlines():
            m = pattern.match(line.strip())
            if m:
                rows.setdefault(m.group(2), {"size": m.group(3)})[m.group(1)] = m.group(4)
    table = pd.DataFrame([[app, r["size"], r.get("memcheck", "n/d"), r.get("racecheck", "n/d")] for app, r in sorted(rows.items())],
                         columns=["App de bench", "Size", "compute-sanitizer memcheck", "compute-sanitizer racecheck"])
    unit = (folder / "unit.log").read_text(encoding="utf-8").splitlines()
    passed = sum(1 for line in unit if line.startswith("PASS"))
    failed = sum(1 for line in unit if line.startswith("FAIL"))
    tsan = [line.strip() for line in (folder / "city-tsan.log").read_text(encoding="utf-8").splitlines() if line.startswith("OK")]
    results = (folder / "results.tsv").read_text(encoding="utf-8").splitlines()
    text = (rl.markdown_table(table) + "\n\n"
            + f"- Tests unitarios (`tests/run.sh`, sin CUDA): **{passed} PASS, {failed} FAIL** (`unit.log`).\n"
            + "- ThreadSanitizer sobre la ciudad completa: " + "; ".join(t.replace("OK    ", "") for t in tsan) + " (`city-tsan.log`).\n"
            + "- `--headless` con TSan y con ASan: duración, SIGINT y SIGTERM sin reportes y con log válido "
              "(`headless-tsan.log`, `headless-asan.log`).\n"
            + f"- CI completa sobre `098eb41`: {sum(1 for r in results if 'VERDE' in r)}/{len(results)} etapas en verde "
              "(`results.tsv`). Logs en `analysis/data/final/correctitud/`.")
    return {"correctitud": text}


def eta_bytes_block() -> dict[str, str]:
    """Bytes utiles y movidos (modelo del bench) y GB/s efectivos de SoA y AoS, de las lineas describe de bench.out."""
    path = FINAL_DIR / "eta_batch" / "bench.out"
    if not path.exists():
        return {}
    pattern = re.compile(r"^(eta_batch_(?:soa|aos))\s+(\d+)\s+gpu_kernel\s+\S+\s+\S+\s+(\S+)\s+\S+\s+\S+\s+(\d+) x (\d+) pares, "
                         r"utiles ([\d.]+) MB \(([\d.]+) GB/s\), movidos \(modelo\) ([\d.]+) MB \(([\d.]+) GB/s\)")
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            data[(m.group(1), int(m.group(2)))] = m.groups()
    rows = []
    for size in sorted({k[1] for k in data}):
        soa, aos = data.get(("eta_batch_soa", size)), data.get(("eta_batch_aos", size))
        if not soa or not aos:
            continue
        rows.append([f"{size:,}", f"{soa[3]} × {soa[4]}", soa[5], soa[6], aos[7], aos[8], soa[2], aos[2]])
    table = pd.DataFrame(rows, columns=["Couriers", "Pares (pedidos × couriers)", "Bytes útiles (MB)",
                                        "SoA: GB/s efectivos", "AoS: bytes movidos, modelo (MB)", "AoS: GB/s (modelo)",
                                        "SoA kernel (ms)", "AoS kernel (ms)"])
    return {"eta_bytes": rl.markdown_table(table) + "\n\nFuente: líneas `describe` de "
            f"`{rl.display_path(path)}` (mediana de gpu_kernel; útiles = 12 B por par, movidos AoS = 36 B por par)."}


def robustness_block(base: rl.Dataset, kernels: dict) -> dict[str, str]:
    csvs = sorted(R30_DIR.glob("*/bench.csv"))
    if not csvs:
        return {}
    other = rl.load(csvs)
    flips, stats = rl.robustness(base, other, kernels)
    text = (f"Se repitió la sesión con `--reps 30` (`{rl.display_path(R30_DIR)}`, todas las aplicaciones menos apsp). "
            f"Puntos (aplicación, tamaño) comparables: **{stats['points']}**; puntos en que cambia el ganador entre la "
            f"mejor CPU y la GPU extremo a extremo: **{stats['flips']}**. Diferencia relativa de las medianas entre las dos "
            f"corridas, en los tiempos ≥ 1 ms (mejor CPU, GPU kernel y GPU e2e; {stats['n_diffs']} valores): mediana "
            f"{stats['diff_median']:.1%}, cuartiles {stats['diff_p25']:.1%}–{stats['diff_p75']:.1%}.")
    if not flips.empty:
        text += "\n\n" + rl.markdown_table(flips)
    return {"robustez": text}


def build_blocks(ds: rl.Dataset, kernels: dict, figures: dict, img_rel: str) -> dict[str, str]:
    summ = rl.summarize(ds.df)
    cmp = rl.compare_all(summ, kernels)
    blocks: dict[str, str] = {}

    lines = []
    for p in ds.csv_paths:
        env = rl.load_env(p)
        run = env.get("run") or {}
        lines.append(f"- `{rl.display_path(p)}`: aplicaciones {', '.join(run.get('apps', []))}; commit "
                     f"`{(env.get('build') or {}).get('git_commit', 'n/d')}`; comando `{rl.run_command(env, p)}` "
                     "(ejecutado en `build/` de la copia WSL).")
    def describe_issue(issue):
        meta = rl.app_meta(kernels, issue.app) if issue.app else {}
        if "faltan versiones" in issue.message and (meta.get("variant_of") or meta.get("baseline_for")):
            role = f"variante de {meta['variant_of']}" if meta.get("variant_of") else f"CPU de referencia de {meta['baseline_for']}"
            return f"{issue} (esperado: {role})"
        return str(issue)

    issues = [describe_issue(i) for i in ds.issues]
    lines.append("")
    n_err = len(ds.errors())
    lines.append(f"Validación automática (2.2): {n_err} error(es)" + (", sin observaciones." if not issues else "; observaciones:"))
    lines += [f"- {i}" for i in issues]
    blocks["fuentes"] = "\n".join(lines)

    blocks["setup"] = rl.markdown_table(rl.setup_table(ds.env))

    peaks = rl.gpu_peaks(ds.env)
    bw = rl.measured_bandwidth(summ)
    t = []
    if peaks:
        t.append(f"- Pico FP32 = {peaks['sms']} SM × {peaks['fp32_lanes_per_sm']} núcleos × 2 (FMA) × "
                 f"{peaks['clock_ghz']:.3f} GHz = **{peaks['fp32_gflops']:.0f} GFLOP/s**.")
        t.append(f"- Pico INT32 = {peaks['sms']} SM × {peaks['int32_lanes_per_sm']} op/ciclo × {peaks['clock_ghz']:.3f} GHz "
                 f"= **{peaks['int32_gops']:.0f} GOP/s** (techo de los kernels enteros).")
        t.append(f"- Ancho de banda pico teórico = 2 × reloj de memoria × bus / 8 = **{peaks['bw_peak_gbs']:.1f} GB/s**.")
        if bw:
            t.append(f"- Ancho de banda medido (`calib_copy` gpu_kernel, {rl.fmt_size(bw['size'], 'bytes')}, 2 × bytes / tiempo) = "
                     f"**{bw['gbs']:.1f} GB/s** ({bw['gbs'] / peaks['bw_peak_gbs']:.0%} del pico).")
            t.append(f"- Punto de equilibrio (FP32 / ancho de banda medido) = **{peaks['fp32_gflops'] / bw['gbs']:.0f} FLOP/B**.")
    blocks["techos"] = "\n".join(t)

    table, totals = rl.summary_table(ds, kernels)
    blocks["resumen"] = (rl.markdown_table(table) + "\n\n" +
                         "Puntos por concepto (máx. 3): " + ", ".join(f"{c} = {n}" for c, n in totals["per_concept"].items())
                         + f". Total **{totals['total']}** → nota **{totals['grade']:.1f}/100**."
                         + (f" Conceptos sin aplicación medida todavía: {', '.join(totals['missing_concepts'])}."
                            if totals["missing_concepts"] else ""))

    if figures.get("_pstate"):
        log = rl.pstate_log_for(ds.csv_paths)
        df = rl.load_pstate_log(log)
        counts = df["pstate"].value_counts()
        temp = pd.to_numeric(df.get("temperature.gpu"), errors="coerce") if "temperature.gpu" in df else None
        blocks["pstate"] = (f"![Estado P de la GPU]({img_rel}/gpu_pstate.png)\n\n"
                            f"Registro `{rl.display_path(log)}`: {len(df)} muestras de `nvidia-smi` en {df['t_s'].iloc[-1]:.0f} s; "
                            + ", ".join(f"{k} = {v}" for k, v in counts.items())
                            + f"; reloj SM {df['sm_mhz'].min():.0f}–{df['sm_mhz'].max():.0f} MHz"
                            + (f"; temperatura {temp.min():.0f}–{temp.max():.0f} °C" if temp is not None and temp.notna().any() else "")
                            + ".")

    pts = rl.roofline_points(summ, kernels)
    r = [f"![Roofline]({img_rel}/roofline.png)", ""]
    if pts.empty:
        r.append("Todavía no hay kernels con operaciones aritméticas en los CSV (solo calibración): la figura muestra los techos.")
    else:
        r.append(rl.markdown_table(pd.DataFrame({
            "Aplicación": pts["application"], "Size": pts["size"], "Ops": [rl.fmt_num(v) for v in pts["ops"]],
            "Bytes": [rl.fmt_num(v) for v in pts["bytes"]], "Intensidad (op/B)": [rl.fmt_num(v) for v in pts["ai"]],
            "Kernel (ms)": [rl.fmt_num(v) for v in pts["kernel_ms"]], "Logrado (GOP/s)": [rl.fmt_num(v) for v in pts["gops"]],
            "Tráfico (GB/s)": [rl.fmt_num(v) for v in pts["gbs"]],
        })))
    blocks["roofline"] = "\n".join(r)
    blocks["derivaciones"] = "\n".join(
        f"- **{app}**: ops = `{m.get('ops')}`, bytes = `{m.get('bytes')}` ({m.get('op_type', 'n/d')}). {m.get('derivation', '')}"
        for app in ds.apps if (m := rl.app_meta(kernels, app)))

    for app in ds.apps:
        meta = rl.app_meta(kernels, app)
        unit = meta.get("size_unit")
        tbl, note = rl.app_table(cmp, app, unit)
        if meta.get("cpu_baselines"):
            labels = sorted({str(b) for b in cmp[cmp["application"] == app]["best_cpu"]})
            note += (" La mejor CPU incluye " + ", ".join(meta["cpu_baselines"])
                     + " (misma tarea, solo CPU); en este barrido la mejor CPU fue: " + ", ".join(labels) + ".")
        blocks[f"tabla:{app}"] = rl.markdown_table(tbl) + "\n\n" + note
        if meta.get("variant_of"):
            main = meta["variant_of"]
            blocks[f"variante:{main}"] = (f"Variante `{app}`: {meta.get('title', '')}\n\n"
                                          + rl.markdown_table(rl.variant_table(cmp, main, app, unit)))
        cross = rl.app_crossover(cmp, app)
        cross_k = rl.app_crossover(cmp, app, "gpu_kernel_ms")
        blocks[f"cruce:{app}"] = (f"- Crossover extremo a extremo (GPU e2e < mejor CPU): {cross.text(unit)} → **{rl.verdict(cross)}**.\n"
                                  f"- Crossover solo kernel (datos ya en la GPU): {cross_k.text(unit)}.")
        blocks[f"dispersion:{app}"] = rl.markdown_table(rl.dispersion_table(cmp, app, unit))
        size, sw = rl.thread_sweep(summ, app)
        blocks[f"hilos:{app}"] = (f"Tamaño {rl.fmt_size(size, unit)}:\n\n" + rl.markdown_table(rl.sweep_table(sw))
                                  if sw is not None and not sw.empty else "Sin barrido de hilos.")
        figs = figures.get(app, {})
        captions = {"tiempo": "Tiempo vs tamaño (log–log), las 4 versiones; barras = mín–máx; línea = cruce",
                    "reparto": "Reparto del tiempo GPU extremo a extremo: copias vs kernel",
                    "hilos": "Barrido de hilos de la versión cpuN"}
        blocks[f"figuras:{app}"] = "\n\n".join(f"![{app}: {captions[k]}]({img_rel}/{Path(figs[k]).name})"
                                               for k in ("tiempo", "reparto", "hilos") if figs.get(k))
    return blocks


def fill(text: str, blocks: dict[str, str]) -> tuple[str, list[str], list[str]]:
    used, missing = [], []

    def repl(m):
        key = m.group("key")
        if key in blocks:
            used.append(key)
            return f"{m.group(1)}\n{blocks[key]}\n{m.group(4)}"
        missing.append(key)
        return m.group(0)

    return BLOCK.sub(repl, text), used, missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append")
    ap.add_argument("--informe", default=str(REPO / "docs" / "informe" / "informe.md"))
    ap.add_argument("--img", default=str(REPO / "docs" / "informe" / "img"))
    ap.add_argument("--kernels", default=str(HERE / "kernels.json"))
    ap.add_argument("--capturas", help="carpeta adicional donde copiar las figuras")
    args = ap.parse_args(argv)

    csvs = [Path(c) for c in args.csv] if args.csv else default_csvs()
    ds = rl.load(csvs)
    kernels = rl.load_kernels(args.kernels)
    img = Path(args.img)
    figures = rl.make_figures(ds, kernels, img)
    informe = Path(args.informe)
    img_rel = Path(img.resolve()).relative_to(informe.resolve().parent).as_posix()
    text = informe.read_text(encoding="utf-8")
    blocks = build_blocks(ds, kernels, figures, img_rel)
    if not args.csv:
        blocks.update(sim_blocks(img, img_rel))
        blocks.update(robustness_block(ds, kernels))
        blocks.update(profile_blocks(img, img_rel))
        blocks.update(correctness_block())
        blocks.update(eta_bytes_block())
    new, used, missing = fill(text, blocks)
    informe.write_text(new, encoding="utf-8", newline="\n")
    if args.capturas:
        dest = Path(args.capturas)
        dest.mkdir(parents=True, exist_ok=True)
        for png in img.glob("*.png"):
            shutil.copy2(png, dest / png.name)
    print(f"CSV: {', '.join(rl.display_path(c) for c in csvs)}")
    print(f"bloques actualizados ({len(used)}): {', '.join(used)}")
    if missing:
        print(f"bloques sin datos todavia ({len(missing)}): {', '.join(missing)}")
    print(f"validacion: {len(ds.errors())} error(es)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
