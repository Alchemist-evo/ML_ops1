"""Render reports/report.md to reports/report.pdf (Markdown -> HTML -> Chromium).

Usage: python scripts/build_report.py   (requires `markdown` and a Chromium binary)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

REPORTS = Path(__file__).resolve().parent.parent / "reports"
SRC, HTML, PDF = REPORTS / "report.md", REPORTS / "_report.html", REPORTS / "report.pdf"

CSS = """
@page { size: A4; margin: 13mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9.4pt; line-height: 1.38; color: #1f2937; }
h1 { font-size: 21pt; border-bottom: 2px solid #1d4ed8; padding-bottom: 6px; color: #1e3a8a; }
h2 { font-size: 13.5pt; color: #1e3a8a; margin-top: 16px; border-bottom: 1px solid #d1d5db; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 11pt; color: #1f2937; margin-top: 11px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #d1d5db; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #eff6ff; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #f3f4f6; padding: 1px 4px; border-radius: 3px; }
pre { background: #f3f4f6; padding: 9px 11px; border-radius: 5px; font-size: 8.3pt; overflow-x: auto; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; max-height: 82mm; width: auto; display: block; margin: 5px auto; page-break-inside: avoid; }
td img { max-height: 52mm; margin: 2px auto; }
hr { border: none; border-top: 1px solid #d1d5db; margin: 16px 0; }
a { color: #1d4ed8; }
"""


def main() -> int:
    body = markdown.markdown(SRC.read_text(), extensions=["tables", "fenced_code", "sane_lists"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Heart Disease Risk Prediction - MLOps Report</title><style>{CSS}</style>"
                    f"</head><body>{body}</body></html>")
    chrome = next((c for c in ("chromium", "google-chrome", "google-chrome-stable")
                   if shutil.which(c)), None)
    if not chrome:
        print("No Chromium/Chrome binary found", file=sys.stderr)
        return 1
    subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={PDF}", HTML.as_uri()],
                   check=True, capture_output=True, timeout=180)
    HTML.unlink()
    print(f"Wrote {PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
