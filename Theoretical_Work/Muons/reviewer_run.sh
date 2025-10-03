#!/usr/bin/env bash
set -Eeuo pipefail

rm -rf outputs && mkdir -p outputs

# 1) Build real S(θ) from Earth + (optional) env_blocks.csv
#    Edit --alt-m if needed; env_blocks.csv is optional.
python3 build_S_from_blocks.py \
  --ring-radius 7.112 --n 3600 \
  --alt-m 228.0 \
  --blocks env_blocks.csv \
  --out outputs/S_real.npy

# 2) Core precession scenarios using the real S(θ)
echo "[A] real S(θ) | spin-specific (kμ=1, kp=0)"
python3 main.py precession --mode file --s-field-path outputs/S_real.npy \
  --kappa-mu 1.0 --kappa-p 0.0 \
  --t-max-us 10000 --dt-us 0.05 --noise-sigma 0 --save-ideal
sleep 1

echo "[B] real S(θ) | universal (kμ=1, kp=1) — cancellation check"
python3 main.py precession --mode file --s-field-path outputs/S_real.npy \
  --kappa-mu 1.0 --kappa-p 1.0 \
  --t-max-us 10000 --dt-us 0.05 --noise-sigma 0 --save-ideal
sleep 1

echo "[C] window-stability on real S(θ)"
python3 main.py precession-windows --mode file --s-field-path outputs/S_real.npy \
  --kappa-mu 1.0 --kappa-p 0.0 \
  --t-max-us 10000 --dt-us 0.05 --noise-sigma 0 \
  --windows "0.0-0.2,0.2-0.6,0.6-1.0"
sleep 1

# 3) Summaries (metrics + report)
python3 process_outputs.py
python3 make_report.py

echo
echo "== Key artifacts =="
echo "  outputs/metrics_summary.csv"
echo "  outputs/metrics_summary.md"
echo "  outputs/final_report.md"