#!/usr/bin/env python3
"""
Hodge_strap.py
Single-file strap to:
 A) inspect cusp eigenfile (if present),
 B) build analytic flat 2-torus spectrum for p-forms,
 C) compute Weyl fit + FFT prime overlay + prime detections,
 D) scan/log near-zero bands and save plots + small summary files.

DROP THIS INTO PROJECT AND RUN LOCALLY:
 python3 Hodge_strap.py
"""

import os, math, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


EIGS_PATH = Path("eigs_1d/eigs_merged.npy")
OUT_DIR = Path(".")
R_MAX = 120
P_MAX_PRIMES = 503
FFT_LEN = 2**16
TAIL_FRAC = 0.20
NEAR_ZERO_BANDS = [1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1.0]


OUT_DIR.mkdir(parents=True, exist_ok=True)

def primes_up_to(n):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2,int(n**0.5)+1):
 if sieve[p]:
 sieve[p*p:n+1:p] = False
 return np.nonzero(sieve)[0].tolist()


def primes_up_to(n):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 r = int(n**0.5)
 for p in range(2, r+1):
 if sieve[p]:
 sieve[p*p:n+1:p] = False
 return np.nonzero(sieve)[0].tolist()

def robust_threshold(vals, kappa=3.0):
 med = float(np.median(vals))
 mad = float(np.median(np.abs(vals-med)))
 if mad == 0:
 mad = np.std(vals) or 1.0
 return med + kappa*mad

summary = {}


cusp_loaded = False
if EIGS_PATH.exists():
 lam = np.load(EIGS_PATH)
 lam = lam[np.isfinite(lam) & (lam > 0)]
 lam = np.sort(lam)
 cusp_loaded = True
 print(f"[cusp] loaded {lam.size} eigenvalues; min={lam.min():.6g}, max={lam.max():.6g}")
 summary['cusp_count'] = int(lam.size)
 summary['cusp_min'] = float(lam.min())
 summary['cusp_max'] = float(lam.max())
else:
 print(f"[cusp] eigenfile not found at {EIGS_PATH}; skipping cusp checks")
 lam = np.array([])


near_zero_counts = []
if cusp_loaded:
 for i in range(len(NEAR_ZERO_BANDS)-1):
 lo, hi = NEAR_ZERO_BANDS[i], NEAR_ZERO_BANDS[i+1]
 c = int(((lam >= lo) & (lam < hi)).sum())
 near_zero_counts.append({'lo':lo, 'hi':hi, 'count':c})
 print(f"[band] λ∈[{lo:.0e},{hi:.0e}) -> {c} eigenvalues")

 with open(OUT_DIR/"near_zero_bands.json","w") as f:
 json.dump(near_zero_counts, f, indent=2)
 summary['near_zero_bands'] = near_zero_counts

 plt.figure(figsize=(7,5))
 sel = lam[lam < 1.0]
 if sel.size>0:
 plt.hist(sel, bins=200)
 plt.xlabel("λ"); plt.ylabel("count")
 plt.title("Near-zero eigenvalue histogram (λ<1.0)")
 plt.tight_layout()
 plt.savefig(OUT_DIR/"near_zero_hist.png")
 plt.close()
 print("[save] near_zero_hist.png")


if cusp_loaded:
 N = np.arange(1, lam.size+1, dtype=float)
 lo_idx = max(0, int(0.05 * lam.size))
 hi_idx = int(0.95 * lam.size)
 lam_fit = lam[lo_idx:hi_idx]
 N_fit = N[lo_idx:hi_idx]
 X = (lam_fit * np.log(np.maximum(lam_fit, 1e-12)))[:,None]
 A = float(np.linalg.lstsq(X, N_fit, rcond=None)[0][0])
 summary['weyl_A_fit'] = A
 print(f"[cusp] least-squares Weyl coefficient A ≈ {A:.6g} (theory 0.25)")


 lam_grid = np.linspace(lam.min(), lam.max(), 2000)
 N_weyl_fit = A * lam_grid * np.log(np.maximum(lam_grid,1e-12))
 plt.figure(figsize=(7,5))
 plt.plot(lam, N, label="counting function")
 plt.plot(lam_grid, N_weyl_fit, "--", label=f"Weyl fit A={A:.4g} * λ log λ")
 plt.xlabel("λ"); plt.ylabel("N(λ)")
 plt.legend(); plt.title("Counting function vs Weyl fit")
 plt.tight_layout(); plt.savefig(OUT_DIR/"weyl_fit.png"); plt.close()
 print("[save] weyl_fit.png")


if cusp_loaded:

 i0 = int((1.0 - TAIL_FRAC) * lam.size)
 lam_tail = lam[i0:]
 N_tail = np.arange(i0+1, lam.size+1, dtype=float)

 shape = lam_tail * np.log(np.maximum(lam_tail,1e-12))
 coef = float(np.linalg.lstsq(shape[:,None], N_tail, rcond=None)[0][0])
 resid = N_tail - coef*shape


 Tvals = np.sqrt(np.maximum(lam_tail - 0.25, 0.0))
 Tg = np.linspace(Tvals[0], Tvals[-1], FFT_LEN)
 Rg = np.interp(Tg, Tvals, resid - resid.mean())
 W = Rg * np.hanning(len(Rg))
 F = np.fft.rfft(W)
 freqs = np.fft.rfftfreq(len(Rg), d=(Tg[-1]-Tg[0])/len(Rg))
 power = np.abs(F)**2


 primes = primes_up_to(P_MAX_PRIMES)
 thr = robust_threshold(power[1:], kappa=3.0)
 detections = []
 for p in primes:
 fp = math.log(p)/(2*math.pi)
 if fp < freqs[-1]:
 k = int(np.argmin(np.abs(freqs - fp)))
 if 2 <= k < len(power)-2:
 Pk = float(power[k])
 side_peak = (power[k-1] < Pk and power[k+1] < Pk)
 detected = int(Pk >= thr and side_peak)
 detections.append({'p':p,'freq':fp,'power':Pk,'detected':detected})

 det_path = OUT_DIR/"prime_detections.txt"
 with open(det_path, "w") as f:
 f.write("# p\tfreq\tpower\tdetected\n")
 for d in detections:
 f.write(f"{d['p']}\t{d['freq']:.6f}\t{d['power']:.6g}\t{d['detected']}\n")
 print(f"[save] {det_path}")


 plt.figure(figsize=(9,5))
 plt.loglog(freqs[1:], power[1:], label="FFT power")
 plt.axhline(thr, ls="--", alpha=0.4, label="thr=median+3·MAD")
 for d in detections:
 if d['detected']:
 plt.axvline(d['freq'], color="r", ls=":", alpha=0.35)
 plt.xlabel("frequency (log p / 2π)"); plt.ylabel("power")
 plt.title("FFT tail power with detected prime frequency markers")
 plt.legend()
 plt.tight_layout(); plt.savefig(OUT_DIR/"fft_prime_detected.png"); plt.close()
 print("[save] fft_prime_detected.png")
 summary['prime_detection_count'] = sum(d['detected'] for d in detections)



vals = []
R = R_MAX
for nx in range(-R, R+1):
 for ny in range(-R, R+1):
 k2 = nx*nx + ny*ny
 vals.append(float(k2))
vals = np.array(vals)
unique, counts = np.unique(vals, return_counts=True)
torus_spec = list(zip(unique.tolist(), counts.tolist()))

zero_row_idx = np.where(unique==0)[0]
if len(zero_row_idx)>0:
 idx0 = zero_row_idx[0]
 zero_counts = {'0forms': int(counts[idx0]), '1forms': int(2*counts[idx0]), '2forms': int(counts[idx0])}
else:
 zero_counts = {'0forms':1,'1forms':2,'2forms':1}

print(f"[torus] zero-mode multiplicities (analytic): {zero_counts}")
summary.update({
 'torus_R_max': R_MAX,
 'torus_zero_0forms': zero_counts['0forms'],
 'torus_zero_1forms': zero_counts['1forms'],
 'torus_zero_2forms': zero_counts['2forms']
})


N_SAVE = 200
low_idxs = np.argsort(unique)[:N_SAVE]
import csv
csv_path = OUT_DIR/"torus_modes.csv"
with open(csv_path,"w",newline="") as f:
 w = csv.writer(f)
 w.writerow(["k2","multiplicity_scalar","multiplicity_0forms","multiplicity_1forms","multiplicity_2forms"])
 for i in low_idxs:
 k2v = unique[i]
 mult = counts[i]
 w.writerow([k2v, mult, mult*1, mult*2, mult*1])
print(f"[save] {csv_path}")


expanded = np.repeat(unique, counts)
plt.figure(figsize=(8,4))
plt.hist(expanded[expanded<=500], bins=80)
plt.xlabel("k^2 (torus eigenvalue)")
plt.ylabel("count (multiplicity)")
plt.title("Torus analytic spectrum (k^2) lower range")
plt.tight_layout(); plt.savefig(OUT_DIR/"torus_spectrum_hist.png"); plt.close()
print("[save] torus_spectrum_hist.png")


proj_path = OUT_DIR/"torus_zero_projector_trace.json"
with open(proj_path,"w") as f:
 json.dump(zero_counts, f, indent=2)
print(f"[save] {proj_path}")


summary_path = OUT_DIR/"Hodge_strap_summary.json"
with open(summary_path, "w") as f:
 json.dump(summary, f, indent=2)
print(f"[save] {summary_path}")

print("done. check outputs in", OUT_DIR.resolve().as_posix())