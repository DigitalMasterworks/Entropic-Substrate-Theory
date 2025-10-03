#!/usr/bin/env bash
set -Eeuo pipefail

# --- Clean start
rm -rf ./outputs
mkdir -p ./outputs

echo "== 1) Make S(θ) fields =="
# Uniform (Earth-level S0)
python3 synth.py s-uniform --n 3600 --S0 0.999999999305 --out ./outputs/S_uniform.npy

# Multipole (tiny harmonics)
python3 synth.py s-multipole --n 3600 \
  --harmonics "1:-2e-12:0.0,2:2e-12:1.0" \
  --out ./outputs/S_multipole.npy

# Gentle Ricci-flow → ring sample (near S≈1)
python3 synth.py s-ricci --N 256 --steps 60 --dt 0.01 --seeds 2 \
  --seed-S 0.99 --S-floor 0.98 --ring-r 85 --ring-n 3600 --laplacian 5pt \
  --out ./outputs/S_ricci.npy

echo "== 2) Make synthetic HVP data =="
python3 synth.py make-Rs --s-min 0.1 --s-max 40 --n 5000 \
  --out ./outputs/Rs_data.csv

echo "== 3) Baseline precession tests (noise-free, long window) =="
# (A) Uniform, spin-specific tick
python3 main.py precession --mode uniform --S0 0.999999999305 \
  --kappa-mu 1.0 --kappa-p 0.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0

# (B) Uniform, universal tick (null)
python3 main.py precession --mode uniform --S0 0.999999999305 \
  --kappa-mu 1.0 --kappa-p 1.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0

# (C) File S(θ): multipole, spin-specific
python3 main.py precession --mode file --s-field-path ./outputs/S_multipole.npy \
  --kappa-mu 1.0 --kappa-p 0.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0

# (D) File S(θ): Ricci ring, spin-specific
python3 main.py precession --mode file --s-field-path ./outputs/S_ricci.npy \
  --kappa-mu 1.0 --kappa-p 0.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0

# (E) File S(θ): Ricci ring, universal tick (null)
python3 main.py precession --mode file --s-field-path ./outputs/S_ricci.npy \
  --kappa-mu 1.0 --kappa-p 1.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0

echo "== 4) HVP (toy) integration =="
python3 main.py hvp --rs-file ./outputs/Rs_data.csv --kernel built_in

echo "== 5) Analytic predictor for each saved ring =="
python3 - <<'PY'
import numpy as np, math, sys
aSM = 116592033e-11
def pred(path, kmu=1.0, kp=0.0):
    S = np.load(path)
    dln = math.log((S**kmu).mean()) - math.log((S**kp).mean())
    da = aSM * dln
    print(f"{path}: ΔlnR={dln:.6e}  δaμ(×1e11)={da*1e11:.3f}  (kmu={kmu}, kp={kp})")
for p in ["./outputs/S_uniform.npy","./outputs/S_multipole.npy","./outputs/S_ricci.npy"]:
    pred(p, 1.0, 0.0)
    pred(p, 1.0, 1.0)
PY

echo "== 6) Sensitivity sweep CSV =="
cat > scan.py <<'PY'
#!/usr/bin/env python3
import numpy as np, math, csv
A_SM = 116592033e-11
S0_vals = np.linspace(1.0-1.5e-9, 1.0+1.5e-9, 31)
dk_vals = np.linspace(-1.0, 1.0, 41)   # kappa_diff = κμ−κp
with open("outputs/sensitivity_scan.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["S0","kappa_diff","delta_lnR","delta_a_mu","delta_a_mu_x1e11"])
    for S0 in S0_vals:
        lnS = math.log(S0)
        for dk in dk_vals:
            dlnR = dk * lnS
            da = A_SM * dlnR
            w.writerow([f"{S0:.12f}", f"{dk:.6f}", f"{dlnR:.6e}", f"{da:.6e}", f"{da*1e11:.3f}"])
print("wrote outputs/sensitivity_scan.csv")
PY
chmod +x scan.py
python3 scan.py

echo "== 7) Report maker (markdown summary) =="
cat > report_maker.py <<'PY'
#!/usr/bin/env python3
import json, glob, os
paths = sorted(glob.glob("outputs/precession_run_*_summary.json"))
out = ["# Substrate g−2: Precession Results\n"]
for p in paths:
    with open(p) as f: J=json.load(f)
    cfg = J["config"]; S = J["S_stats"]
    out += [
      f"## {os.path.basename(p)}",
      f"- mode: **{cfg['mode']}**   κμ={cfg['kappa_mu']}, κp={cfg['kappa_p']}",
      f"- S_mean={S['S_mean']:.12f} (min={S['S_min']:.12f}, max={S['S_max']:.12f})",
      f"- R_std={J['ratio_std']:.12e}  R_meas={J['ratio_meas']:.12e}",
      f"- Δln R={J['delta_ln_ratio']:.12e}",
      f"- δaμ={J['delta_a_linear']:.12e}  (×1e11={J['delta_a_linear_x1e11']:.3f})",
      f"- within 1σ room? {J['within_current_room_1sigma']}",
      ""
    ]
text = "\n".join(out)
with open("outputs/g2_report.md","w") as f: f.write(text)
print("wrote outputs/g2_report.md")
PY
chmod +x report_maker.py
python3 report_maker.py

echo "== 8) (Optional) CBO robustness (commented) =="
# python3 main.py precession --mode uniform --S0 0.999999999305 \
#   --kappa-mu 1.0 --kappa-p 0.0 --t-max-us 10000 --dt-us 0.05 \
#   --include-cbo --cbo-amp 0.01 --noise-sigma 50

echo "All done. Outputs in ./outputs/"