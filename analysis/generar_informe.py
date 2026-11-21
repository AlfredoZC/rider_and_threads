"""Genera las tablas (Markdown) y las figuras (PNG) del informe sin Jupyter.

Uso (desde la raiz del repo):
    C:\\Python313\\python.exe analysis\\generar_informe.py
        [--csv analysis/data/local_calib/results.csv [--csv otro.csv ...]]
        [--kernels analysis/kernels.json] [--fig-dir analysis/figures] [--out analysis/informe_tablas.md]
        [--config config/ciudad.json] [--strict]

Cada CSV debe tener al lado su <csv>.env.json (lo escribe el modo bench). Con varios CSV (p. ej. una corrida por
aplicacion) se concatenan; deben venir de la misma maquina y del mismo build.
Codigo de salida: 0; con --strict, 1 si la validacion encontro errores (el informe se escribe igual, con los
errores listados en su seccion "Validacion").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import report_lib as rl  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", help="CSV del modo bench (repetible)")
    ap.add_argument("--kernels", default=str(HERE / "kernels.json"))
    ap.add_argument("--fig-dir", default=str(HERE / "figures"))
    ap.add_argument("--out", default=str(HERE / "informe_tablas.md"))
    ap.add_argument("--config", default="config/ciudad.json", help="config usado en la corrida (el env.json no lo guarda)")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    csvs = args.csv or [str(p) for p in sorted((HERE / "data" / "final").glob("*/bench.csv"))]
    ds = rl.load(csvs)
    kernels = rl.load_kernels(args.kernels)
    figures = rl.make_figures(ds, kernels, args.fig_dir)
    md = rl.render_report(ds, kernels, figures, args.out, config=args.config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8", newline="\n")

    n_fig = sum(1 for key, f in figures.items() if not key.startswith("_") for p in f.values() if p)
    n_fig += sum(1 for key, p in figures.items() if key.startswith("_") and p)
    print(f"filas: {len(ds.df)}  aplicaciones: {', '.join(ds.apps)}")
    print(f"tablas: {out}")
    print(f"figuras ({n_fig}): {Path(args.fig_dir)}")
    for i in ds.issues:
        print(f"  {i}")
    errors = ds.errors()
    print(f"validacion: {len(errors)} error(es), {len(ds.issues) - len(errors)} aviso(s)/info")
    return 1 if (args.strict and errors) else 0


if __name__ == "__main__":
    sys.exit(main())
