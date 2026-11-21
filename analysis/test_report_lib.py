"""Tests rapidos de report_lib (sin GPU ni WSL). Correr desde la raiz del repo:
    C:\\Python313\\python.exe -m pytest analysis/test_report_lib.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generar_sintetico as gs  # noqa: E402
import report_lib as rl  # noqa: E402


@pytest.fixture()
def synth_csv(tmp_path):
    return gs.write(tmp_path / "sintetico")


def write_csv(tmp_path, df: pd.DataFrame, env: dict | None = gs.ENV) -> Path:
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    if env is not None:
        Path(str(path) + ".env.json").write_text(json.dumps(env), encoding="utf-8")
    return path


def errors_containing(ds, text):
    return [i for i in ds.issues if i.level == "error" and text in i.message]


# ---------------------------------------------------------------- validacion

def test_synthetic_csv_is_valid(synth_csv):
    ds = rl.load(synth_csv)
    assert ds.errors() == []
    assert ds.apps == [gs.APP]
    assert len(ds.df) == len(gs.SIZES) * (2 + 1 + len(gs.THREADS)) * gs.REPS


def test_detects_correct_no(tmp_path):
    df = gs.build_rows()
    df.loc[(df["version"] == "gpu_e2e") & (df["size"] == 10_000) & (df["rep"] == 3), "correct"] = "no"
    ds = rl.load(write_csv(tmp_path, df))
    found = errors_containing(ds, "correct=no")
    assert len(found) == 1 and "gpu_e2e" in found[0].message and "size=10000" in found[0].message
    table, _ = rl.summary_table(ds, rl.load_kernels())
    tbl, _ = rl.app_table(rl.compare(rl.summarize(ds.df)), gs.APP)
    assert tbl.loc[1, "Correct?"].startswith("NO")
    assert tbl.loc[0, "Correct?"].startswith("sí")


def test_detects_fewer_than_10_reps(tmp_path):
    df = gs.build_rows()
    drop = (df["version"] == "cpu1") & (df["size"] == 1_000) & (df["rep"] >= 7)
    ds = rl.load(write_csv(tmp_path, df[~drop]))
    found = errors_containing(ds, "repeticiones")
    assert len(found) == 1 and "solo 7 repeticiones" in found[0].message and "version=cpu1" in found[0].message


def test_detects_too_few_sizes_and_decades(tmp_path):
    df = gs.build_rows(sizes=[1_000, 2_000, 5_000, 10_000])
    ds = rl.load(write_csv(tmp_path, df))
    assert errors_containing(ds, "solo 4 tamanos")
    assert errors_containing(ds, "ordenes de magnitud")


def test_columns_must_be_exact(tmp_path):
    df = gs.build_rows().rename(columns={"kernel_ms": "kernel"})
    with pytest.raises(rl.BenchDataError, match="faltan"):
        rl.load(write_csv(tmp_path, df))
    reordered = gs.build_rows()[["version", "application"] + rl.EXPECTED_COLUMNS[2:]]
    assert rl.validate_columns(reordered)[0].level == "error"


def test_detects_non_release_build(tmp_path):
    env = json.loads(json.dumps(gs.ENV))
    env["build"]["type"] = "Debug"
    env["build"]["sanitizer"] = "thread"
    ds = rl.load(write_csv(tmp_path, gs.build_rows(), env))
    assert errors_containing(ds, "Release")
    assert errors_containing(ds, "sanitizer")


def test_invalid_values_are_flagged(tmp_path):
    df = gs.build_rows()
    df.loc[0, "version"] = "gpu"
    df.loc[1, "ms"] = "abc"
    ds = rl.load(write_csv(tmp_path, df))
    assert errors_containing(ds, "version fuera de")
    assert errors_containing(ds, "invalidos")


# ---------------------------------------------------------------- calculos

def test_crossover_known_value(synth_csv):
    ds = rl.load(synth_csv)
    cmp = rl.compare(rl.summarize(ds.df))
    cross = rl.app_crossover(cmp, gs.APP)
    assert cross.kind == "cross"
    assert cross.lo == 100_000 and cross.hi == 1_000_000
    assert math.isclose(cross.size, gs.EXPECTED_CROSSOVER, rel_tol=1e-6)
    assert rl.verdict(cross) == "usar GPU sobre el umbral"


def test_crossover_never_always_and_relapse():
    sizes = [1, 10, 100, 1000, 10000]
    assert rl.crossover(sizes, [1] * 5, [2] * 5).kind == "never"
    always = rl.crossover(sizes, [2] * 5, [1] * 5)
    assert always.kind == "gpu_always" and always.size == 1
    relapse = rl.crossover(sizes, [1, 1, 1, 1, 1], [2, 0.5, 0.5, 2, 2])
    assert relapse.kind == "cross" and relapse.upper is not None and 10 < relapse.upper < 1000
    assert "vuelve a perder" in relapse.text() and rl.verdict(relapse) == "usar GPU solo en el rango ganador"
    exact = rl.crossover([10, 10000], [1e-3 * 10, 1e-3 * 10000], [0.01 * 10 ** 0.5, 0.01 * 10000 ** 0.5])
    assert exact.kind == "cross"
    assert math.isclose(exact.size, 100.0, rel_tol=1e-9)  # 1e-3 n = 0.01 sqrt(n) -> n = 100
    assert rl.crossover(sizes, [1] * 5, [float("nan")] * 5).kind == "no_data"


def test_best_cpu_and_speedups(synth_csv):
    ds = rl.load(synth_csv)
    cmp = rl.compare(rl.summarize(ds.df))
    row = cmp[cmp["size"] == 1_000_000].iloc[0]
    assert row["cpuN_threads"] == 16
    assert math.isclose(row["best_cpu_ms"], 1e-5 * 1e6 / gs.SPEEDUP[16], rel_tol=1e-6)
    assert row["best_cpu"] == "cpuN(16)"
    base = gs.base_times(1e6)
    assert math.isclose(row["speedup_e2e"], row["best_cpu_ms"] / base["e2e"], rel_tol=1e-6)
    assert math.isclose(row["speedup_kernel"], row["best_cpu_ms"] / base["kernel"], rel_tol=1e-6)


def test_table_has_exact_columns(synth_csv):
    ds = rl.load(synth_csv)
    tbl, note = rl.app_table(rl.compare(rl.summarize(ds.df)), gs.APP)
    assert list(tbl.columns) == ["Size", "CPU 1 thread (ms)", "CPU 16 threads (ms)", "GPU kernel (ms)",
                                 "GPU end to end (ms)", "Speedup vs best CPU (kernel / end to end)", "Correct?"]
    assert "N = 16" in note
    md = rl.markdown_table(tbl)
    assert md.splitlines()[0].startswith("| Size | CPU 1 thread (ms) | CPU 16 threads (ms) |")


def test_thread_sweep_uses_largest_full_size(synth_csv):
    ds = rl.load(synth_csv)
    size, sw = rl.thread_sweep(rl.summarize(ds.df), gs.APP)
    assert size == max(gs.SIZES)
    assert list(sw["threads"]) == gs.THREADS
    assert math.isclose(sw.set_index("threads").loc[16, "speedup"], gs.SPEEDUP[16], rel_tol=1e-6)


def test_dispersion_stats(synth_csv):
    summ = rl.summarize(rl.load(synth_csv).df)
    r = summ[(summ["version"] == "cpu1") & (summ["size"] == 1000)].iloc[0]
    assert r["n"] == 10
    assert math.isclose(r["min"], 1e-2 * min(gs.FACTORS), rel_tol=1e-6)
    assert math.isclose(r["max"], 1e-2 * max(gs.FACTORS), rel_tol=1e-6)
    assert r["iqr"] > 0


def test_formula_evaluator_is_safe():
    assert rl.eval_formula("2*n**3", {"n": 10}) == 2000
    assert rl.eval_formula("16*n**3/T", {"n": 64, "T": 32}) == 16 * 64 ** 3 / 32
    assert rl.eval_formula("6*n*(32/b)", {"n": 10, "b": 8}) == 240
    assert rl.eval_formula("log2(n)", {"n": 8}) == 3
    for bad in ["__import__('os')", "n.__class__", "open('x')", "x + 1", "2**99999", "[1]"]:
        with pytest.raises((ValueError, SyntaxError)):
            rl.eval_formula(bad, {"n": 1})


def test_peaks_and_roofline(synth_csv):
    ds = rl.load(synth_csv)
    peaks = rl.gpu_peaks(ds.env)
    assert math.isclose(peaks["fp32_gflops"], 20 * 128 * 2 * 2.355, rel_tol=1e-9)
    assert math.isclose(peaks["int32_gops"], 20 * 64 * 2.355, rel_tol=1e-9)
    pts = rl.roofline_points(rl.summarize(ds.df), rl.load_kernels())
    assert set(pts["application"]) == {gs.APP}
    assert math.isclose(pts["ai"].iloc[0], 0.5)


def test_grade_formula():
    assert rl.grade(10) == 50 and rl.grade(20) == 75 and rl.grade(30) == 100
    assert rl.grade(5) == 25 and rl.grade(0) == 0


def test_kernels_json_formulas_evaluate():
    kernels = rl.load_kernels()
    concepts = set()
    for app, meta in kernels["apps"].items():
        variables = rl.size_variables(meta, 1000.0)
        assert rl.eval_formula(meta["ops"], variables) >= 0, app
        assert rl.eval_formula(meta["bytes"], variables) >= 0, app
        if meta.get("counts_for_points"):
            concepts.add(meta["concept"])
    assert concepts == set(rl.CONCEPTS)


def test_end_to_end_report(synth_csv, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    ds = rl.load(synth_csv)
    kernels = rl.load_kernels()
    figs = rl.make_figures(ds, kernels, tmp_path / "fig")
    assert all(p is not None and p.exists() for p in figs[gs.APP].values())
    assert figs["_roofline"].exists()
    md = rl.render_report(ds, kernels, figs, tmp_path / "informe.md")
    assert "| Size | CPU 1 thread (ms) | CPU 16 threads (ms) |" in md
    assert "Datos SINTÉTICOS" in md
    assert "--bench results.csv" in md


def test_run_command_from_env_list_and_fallback():
    env = {"run": {"command": ["delivery_sim", "../config/ciudad.json", "--headless", "--bench", "results.csv"]}}
    assert rl.run_command(env, "x.csv") == "./delivery_sim ../config/ciudad.json --headless --bench results.csv"
    env = {"run": {"apps": ["a", "b"], "reps": 10, "seed": 7}}
    assert rl.run_command(env, "dir/r.csv") == "./delivery_sim config/ciudad.json --headless --bench r.csv --apps a,b --reps 10 --seed 7"


def test_actualizar_informe_fills_blocks_idempotently(synth_csv, tmp_path):
    import actualizar_informe as ai
    informe = tmp_path / "informe.md"
    informe.write_text("a\n<!-- auto:tabla:demo_sintetica -->\nviejo\n<!-- /auto:tabla:demo_sintetica -->\n"
                       "<!-- auto:tabla:no_existe -->\nx\n<!-- /auto:tabla:no_existe -->\nb\n", encoding="utf-8")
    args = ["--csv", str(synth_csv), "--informe", str(informe), "--img", str(tmp_path / "img")]
    assert ai.main(args) == 0
    first = informe.read_text(encoding="utf-8")
    assert "viejo" not in first and "| Size | CPU 1 thread (ms) | CPU 16 threads (ms) |" in first
    assert "<!-- auto:tabla:no_existe -->\nx\n" in first and first.startswith("a\n") and first.endswith("b\n")
    assert ai.main(args) == 0
    assert informe.read_text(encoding="utf-8") == first


def test_cpu_baseline_from_other_app_and_size_vars(synth_csv):
    ds = rl.load(synth_csv)
    df = ds.df.copy()
    other = df[df["version"].isin(["cpu1", "cpuN"])].copy()
    other["application"] = "otra_cpu"
    other["ms"] = other["ms"] / 100.0  # una CPU 100 veces mas rapida
    summ = rl.summarize(pd.concat([df, other], ignore_index=True))
    kernels = {"apps": {gs.APP: {"cpu_baselines": ["otra_cpu"]}}}
    cmp = rl.compare_all(summ, kernels)
    row = cmp[(cmp["application"] == gs.APP) & (cmp["size"] == 1_000_000)].iloc[0]
    assert row["best_cpu"].startswith("otra_cpu:")
    assert math.isclose(row["best_cpu_ms"], 1e-5 * 1e6 / gs.SPEEDUP[16] / 100.0, rel_tol=1e-6)
    assert rl.app_crossover(cmp, gs.APP).kind == "never" or rl.app_crossover(cmp, gs.APP).size > gs.EXPECTED_CROSSOVER
    meta = {"size_vars": {"m": {"50": 10, "default": "max(1, floor(n/8))"}}}
    assert rl.size_variables(meta, 50)["m"] == 10 and rl.size_variables(meta, 777)["m"] == 97
