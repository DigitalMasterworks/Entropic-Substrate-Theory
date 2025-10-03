#!/usr/bin/env bash
set -Eeuo pipefail
python3 plot_all.py
python3 make_report.py
python3 make_paper_tex.py
# build PDF if LaTeX is available
if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory outputs/paper outputs/paper/main.tex >/dev/null || true
fi
tar -czf substrate_g2_results.tar.gz outputs/*.json outputs/*.md outputs/*.png outputs/*.csv outputs/*.npy outputs/paper/*.tex outputs/paper/*.pdf 2>/dev/null || true
ls -lh substrate_g2_results.tar.gz || true
echo "All set. Plots, report, LaTeX (and PDF if TeX present) are in outputs/. Bundle is substrate_g2_results.tar.gz"