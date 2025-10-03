#!/usr/bin/env bash
set -Eeuo pipefail

# ---------- Clean start ----------
rm -rf ./outputs
mkdir -p ./outputs

echo "== 1) Baseline precession (uniform Earth S0) =="

# (A) Uniform, spin-specific tick (kappa_mu=1, kappa_p=0)
python3 main.py precession --mode uniform --S0 0.999999999305 \
  --kappa-mu 1.0 --kappa-p 0.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0 --save-ideal

# (B) Uniform, universal tick (null cancel: kappa_mu = kappa_p)
python3 main.py precession --mode uniform --S0 0.999999999305 \
  --kappa-mu 1.0 --kappa-p 1.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0 --save-ideal

echo "== 2) Radial test (tiny gradient around ring) =="
# NOTE: use '=' so the negative number isn't parsed as a flag
python3 main.py precession --mode radial --S-center 0.9999999996 --dS-dr=-1e-12 --ring-radius 7.112 \
  --kappa-mu 1.0 --kappa-p 0.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0 --save-ideal

echo "== 3) Window-stability fits (E989-style) =="
python3 main.py precession-windows --mode uniform --S0 0.999999999305 \
  --kappa-mu 1.0 --kappa-p 0.0 --t-max-us 10000 --dt-us 0.05 --noise-sigma 0 \
  --windows "0.0-0.2,0.2-0.6,0.6-1.0"

echo "== 4) Create an S(θ) .npy inline for 'predict' route =="
python3 - <<'PY'
import numpy as np
n = 3600
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
S0 = 0.999999999305
S = S0 + (-2e-12)*np.cos(theta + 0.0) + (2e-12)*np.cos(2*theta + 1.0)
S = np.clip(S, 1e-12, 1.0)
np.save("./outputs/S_predict.npy", S)
print("Wrote ./outputs/S_predict.npy with mean", S.mean())
PY

echo "== 5) Analytic predictions (no fitting) on that S(θ) =="
# Spin-specific vs. universal
python3 main.py predict --s-field-path ./outputs/S_predict.npy --kappa-mu 1.0 --kappa-p 0.0
python3 main.py predict --s-field-path ./outputs/S_predict.npy --kappa-mu 1.0 --kappa-p 1.0

echo "== 6) Generate a toy Rs(s) CSV inline for HVP routes =="
python3 - <<'PY'
import numpy as np, csv
s = np.linspace(0.1, 40.0, 5000)
R = 2.5/(1.0+np.exp(-(np.sqrt(s)-2.0)/0.3))*(1.0+0.05*np.log1p(s))
def bw(s, m, g, a):
    m2 = m*m
    return a*(m2*g*g)/((s-m2)**2 + (m2*g*g))
R += bw(s, 0.775, 0.149, 8.0) + bw(s, 1.02, 0.0042, 2.5)
with open("./outputs/Rs_data.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["s","R"]); w.writerows(zip(s,R))
print("Wrote ./outputs/Rs_data.csv with", len(s), "rows")
PY

echo "== 7) HVP integration (toy) + window splits =="
python3 main.py hvp --rs-file ./outputs/Rs_data.csv --kernel built_in
python3 main.py hvp-windows --rs-file ./outputs/Rs_data.csv --windows "SD:0-1.0,W:1.0-9.0,LD:9.0-inf"

echo "== 8) List outputs =="
ls -1 ./outputs
echo "== Done. All artifacts are in ./outputs =="

python3 - <<'PY'
import json, glob, os, math
ROOM_1SIG = 63.0    # ×1e-11
ROOM_2SIG = 126.0   # ×1e-11

rows = []
for p in sorted(glob.glob("outputs/precession_run_*_summary.json")):
    J = json.load(open(p))
    cfg = J["config"]; km, kp = cfg["kappa_mu"], cfg["kappa_p"]
    kd = km - kp
    da_x1e11 = J["delta_a_linear_x1e11"]          # already ×1e-11 units
    tag = f"{os.path.basename(p)} | mode={cfg['mode']} | κμ={km} κp={kp}"
    if abs(kd) < 1e-15:
        rows.append((tag, da_x1e11, kd, None, None, "κ-diff=0 (universal tick)"))
    else:
        sens = da_x1e11 / kd                       # δa per unit κ-diff
        if abs(sens) < 1e-15:
            rows.append((tag, da_x1e11, kd, None, None, "no sensitivity"))
        else:
            bound1 = ROOM_1SIG/abs(sens)
            bound2 = ROOM_2SIG/abs(sens)
            rows.append((tag, da_x1e11, kd, bound1, bound2, ""))

print("\n=== Bounds on |κμ−κp| from your runs ===")
for tag, da, kd, b1, b2, note in rows:
    print(f"- {tag}")
    print(f"    δaμ(×1e11)={da:+.3f}, κμ−κp={kd:+.3f}", ("["+note+"]" if note else ""))
    if b1 is not None:
        print(f"    ⇒ |κμ−κp| ≤ {b1:.3f}  (1σ),  ≤ {b2:.3f}  (2σ)")
print()
PY


python3 - <<'PY'
import json,glob,os,math,datetime
ROOM_1SIG,ROOM_2SIG = 63.0,126.0
outs = []
outs.append("# Substrate g−2: Fermilab-style Precession Test (Auto Report)\n")
outs.append(f"_Generated: {datetime.datetime.utcnow().isoformat()}Z_\n")
for p in sorted(glob.glob("outputs/precession_run_*_summary.json")):
    J=json.load(open(p)); cfg=J["config"]; S=J["S_stats"]
    km,kp=cfg["kappa_mu"],cfg["kappa_p"]; kd=km-kp
    da=J["delta_a_linear_x1e11"]  # ×1e-11
    line=[f"## {os.path.basename(p)}",
          f"- mode: **{cfg['mode']}**   κμ={km}  κp={kp}",
          f"- S_mean={S['S_mean']:.12f} (min={S['S_min']:.12f}, max={S['S_max']:.12f})",
          f"- ΔlnR={J['delta_ln_ratio']:.12e}   δaμ={J['delta_a_linear']:.12e}  (×1e11={da:.3f})",
          f"- within 1σ ‘room’ (±63×1e−11)?  **{J['within_current_room_1sigma']}**"]
    if abs(kd)<1e-15:
        line.append("- κμ−κp = +0.000 ⇒ **no bound** (universal/cancel or zero sensitivity)")
    else:
        sens = da/kd if abs(kd)>1e-15 else float('nan')
        if abs(sens)<1e-15 or not math.isfinite(sens):
            line.append("- Sensitivity: n/a")
        else:
            line.append(f"- Sensitivity ⇒ |κμ−κp| ≤ **{ROOM_1SIG/abs(sens):.3f}** (1σ), ≤ **{ROOM_2SIG/abs(sens):.3f}** (2σ)")
    line.append("")
    outs.extend(line)
open("outputs/final_report.md","w").write("\n".join(outs))
print("wrote outputs/final_report.md")
PY

python3 - <<'PY'
import glob, os, numpy as np
import matplotlib.pyplot as plt

for p in sorted(glob.glob("outputs/precession_run_*_timeseries.csv")):
    d=np.genfromtxt(p,delimiter=",",names=True)
    plt.figure()
    plt.plot(d["t_s"], d["counts"])
    plt.xlabel("time (s)"); plt.ylabel("counts")
    plt.title(os.path.basename(p))
    out=p.replace(".csv",".png")
    plt.tight_layout(); plt.savefig(out,dpi=160); plt.close()
    print("wrote", out)

for p in sorted(glob.glob("outputs/precession_run_*_ideal_timeseries.csv")):
    d=np.genfromtxt(p,delimiter=",",names=True)
    plt.figure()
    plt.plot(d["t_s"], d["counts_ideal"])
    plt.xlabel("time (s)"); plt.ylabel("counts (ideal)")
    plt.title(os.path.basename(p))
    out=p.replace(".csv",".png")
    plt.tight_layout(); plt.savefig(out,dpi=160); plt.close()
    print("wrote", out)
PY