"""Genera docs/informe/informe.pdf desde docs/informe/informe.md (Markdown -> HTML -> PDF con Chrome o Edge headless).

Uso (desde la raiz del repo, despues de analysis/actualizar_informe.py):
    C:\\Python313\\python.exe analysis\\generar_pdf.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

REPO = Path(__file__).resolve().parent.parent
INFORME = REPO / "docs" / "informe" / "informe.md"
BROWSERS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]

CSS = """
@page { size: A4; margin: 14mm 12mm 14mm 12mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.5pt; line-height: 1.38; color: #1b1b1a; }
h1 { font-size: 18pt; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; break-after: avoid; }
h4 { font-size: 10pt; margin: 10pt 0 3pt; break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt; font-size: 7.4pt; break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 0.6pt solid #000; padding: 2pt 4pt; text-align: left; vertical-align: top; background: none; }
th { font-weight: bold; }
th code, td code { background: none; padding: 0; }
hr { display: none; }
img { max-width: 100%; display: block; margin: 6pt auto; break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8pt; background: #f4f3ef; padding: 0 2pt; }
pre { background: #f4f3ef; padding: 6pt; font-size: 7.6pt; white-space: pre-wrap; word-break: break-all; }
pre code { background: none; padding: 0; }
blockquote { border-left: 3px solid #c3c2b7; margin: 6pt 0; padding: 2pt 8pt; color: #52514e; }
"""


def main() -> int:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(INFORME.read_text(encoding="utf-8"))
    html_path = INFORME.with_suffix(".html")
    html_path.write_text(f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Informe de benchmarking</title>"
                         f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    browser = next((b for b in BROWSERS if b.exists()), None) or shutil.which("chrome") or shutil.which("msedge")
    if not browser:
        print("No se encontro Chrome ni Edge para imprimir el PDF", file=sys.stderr)
        return 1
    pdf_path = INFORME.with_suffix(".pdf")
    cmd = [str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not pdf_path.exists():
        print(result.stderr[-2000:], file=sys.stderr)
        return 1
    print(f"{pdf_path} ({pdf_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
