#!/usr/bin/env python3
"""
FULL TEST: Relative determinant + prime oscillations on REAL eigenvalues.

What it does
------------
1) Load eigenvalues (1D cusp testbed OK) from a.npy file (sorted, unique, >0).
2) Fit the Weyl tail with A fixed to 1/4 and free (B, b, c):
 N(λ) ≈ (1/4) λ log λ + B λ + b sqrt(λ) + c
3) Build "relative" zero counts:
 N_rel(T) = N_data(T^2 + 1/4) - N_model(T^2 + 1/4)
 and compare to the Riemann asymptotic α·(T/(2π)) log T + β, fitted by least squares.
4) Construct a remainder R(T) = N_rel(T) - [α·(T/(2π)) log T + β],
 resample uniformly in T, window, FFT, and look for peaks at frequencies log p / (2π).
5) Save:
 - zero_density_full.png: density comparison (data vs model vs Riemann asymptotic)
 - prime_fft_full.png: FFT with prime frequency markers; detected primes highlighted
 - prime_detections.txt: table of primes, power, z-scores, jitter-stability flags
 - full_test_summary.csv: one-line run summary (B,b,c, α,β, R^2, counts, etc.)

Usage
-----
python3 full_test.py --eigs eigs_1d/eigs_merged.npy --out./out \
 --kappa 3.0 --pmax 503 --holdout 151 251 --window hann

Dependencies
------------
numpy, matplotlib
"""

import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List





def load_eigs(path: str) -> np.ndarray:
 arr = np.load(path)
 arr = np.array(arr).ravel()
 arr = arr[np.isfinite(arr)]
 arr = arr[arr > 0.0]
 arr = np.unique(np.sort(arr))
 if arr.size < 1000:
 raise ValueError(f"Too few eigenvalues ({arr.size}); need ~1e3+ for a stable tail fit.")
 return arr

def counting_function(sorted_lams: np.ndarray, lam_vals: np.ndarray) -> np.ndarray:

 idx = np.searchsorted(sorted_lams, lam_vals, side="right")
 return idx.astype(float)

def fit_weyl_tail(sorted_lams: np.ndarray, A: float = 0.25, tail_frac: float = 0.4) -> Tuple[float, float, float]:
 """
 Fit N(λ) ≈ A λ log λ + B λ + b sqrt(λ) + c on the top `tail_frac` fraction of λ.
 Return (B, b, c).
 """
 n = sorted_lams.size
 start = int((1.0 - tail_frac) * n)
 lam_tail = sorted_lams[start:]

 N_data = np.arange(start + 1, n + 1, dtype=float)

 with np.errstate(divide='ignore', invalid='ignore'):
 y = N_data - A * lam_tail * np.log(lam_tail)
 X = np.column_stack([lam_tail, np.sqrt(lam_tail), np.ones_like(lam_tail)])
 coef, *_ = np.linalg.lstsq(X, y, rcond=None)
 B, b, c = coef.tolist()
 return B, b, c

def primes_up_to(n: int) -> List[int]:
 sieve = np.ones(n + 1, dtype=bool)
 sieve[:2] = False
 for p in range(2, int(n**0.5) + 1):
 if sieve[p]:
 sieve[p*p:n+1:p] = False
 return [i for i in range(2, n + 1) if sieve[i]]

def robust_threshold(values: np.ndarray, kappa: float) -> float:
 med = float(np.median(values))
 mad = float(np.median(np.abs(values - med))) or 1e-12
 return med + kappa * mad

def window_array(n: int, kind: str = "hann") -> np.ndarray:
 if kind.lower() == "hann":
 return np.hanning(n)
 elif kind.lower() == "tukey":
 alpha = 0.5
 w = np.ones(n)
 edge = int(alpha * (n - 1) / 2.0)
 if edge > 0:
 h = np.hanning(2 * edge)
 w[:edge] = h[:edge]
 w[-edge:] = h[-edge:]
 return w
 else:
 return np.ones(n)

def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
 """Fit y ≈ a*x + b by least squares; return (a, b, R^2)."""
 X = np.column_stack([x, np.ones_like(x)])
 coef, *_ = np.linalg.lstsq(X, y, rcond=None)
 a, b = coef.tolist()
 yhat = a * x + b
 ss_res = float(np.sum((y - yhat)**2))
 ss_tot = float(np.sum((y - np.mean(y))**2)) or 1.0
 r2 = 1.0 - ss_res / ss_tot
 return a, b, r2





def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--eigs", type=str, required=True, help="Path to eigs_merged.npy")
 ap.add_argument("--out", type=str, default=".", help="Output directory")
 ap.add_argument("--kappa", type=float, default=3.0, help="MAD threshold multiplier for prime detection")
 ap.add_argument("--pmax", type=int, default=503, help="Max prime to mark")
 ap.add_argument("--holdout", type=int, nargs=2, default=[151, 251], help="Holdout prime interval [a b]")
 ap.add_argument("--window", type=str, default="hann", choices=["hann","tukey","rect"],
 help="Window for FFT")
 ap.add_argument("--tpoints", type=int, default=400, help="Number of T points for N_rel sampling")
 ap.add_argument("--fftpoints", type=int, default=2**15, help="FFT grid size (power of 2 recommended)")
 args = ap.parse_args()

 os.makedirs(args.out, exist_ok=True)
 fig_density = os.path.join(args.out, "zero_density_full.png")
 fig_fft = os.path.join(args.out, "prime_fft_full.png")
 txt_det = os.path.join(args.out, "prime_detections.txt")
 csv_summary = os.path.join(args.out, "full_test_summary.csv")


 eigs = load_eigs(args.eigs)


 A = 0.25
 B, b, c = fit_weyl_tail(eigs, A=A, tail_frac=0.4)


 lam_min = float(eigs[0])
 lam_max = float(eigs[-1])
 T_min = max(5.0, math.sqrt(max(lam_min - 0.25, 1e-9)))
 T_max = max(10.0, min(math.sqrt(lam_max - 0.25), 1e6))
 T_vals = np.linspace(T_min, T_max, args.tpoints)
 lam_T = T_vals**2 + 0.25


 N_data_T = counting_function(eigs, lam_T)
 with np.errstate(divide='ignore', invalid='ignore'):
 N_model_T = A*lam_T*np.log(lam_T) + B*lam_T + b*np.sqrt(lam_T) + c
 N_rel_T = N_data_T - N_model_T


 with np.errstate(divide='ignore', invalid='ignore'):
 N_riem_shape = (T_vals / (2.0 * np.pi)) * np.log(T_vals)

 N_riem_shape[np.isnan(N_riem_shape)] = 0.0
 N_riem_shape[np.isinf(N_riem_shape)] = 0.0

 alpha, beta, r2 = fit_affine(N_riem_shape, N_rel_T)


 trend = alpha * N_riem_shape + beta
 rem = N_rel_T - trend
 rem = rem - np.mean(rem)


 T_uniform = np.linspace(T_vals[0], T_vals[-1], args.fftpoints)
 R_uniform = np.interp(T_uniform, T_vals, rem)


 w = window_array(args.fftpoints, args.window)
 Rw = R_uniform * w
 F = np.fft.rfft(Rw)
 freqs = np.fft.rfftfreq(args.fftpoints, d=(T_uniform[1]-T_uniform[0]))
 power = np.abs(F)**2
 power[0] = 0.0


 plist = primes_up_to(args.pmax)
 a, b_hold = args.holdout
 in_holdout = lambda p: (a <= p <= b_hold)

 thr = robust_threshold(power[1:], args.kappa)
 hits = []
 for p in plist:
 fp = math.log(p) / (2.0 * math.pi)

 k = int(np.argmin(np.abs(freqs - fp)))
 if k <= 1 or k >= len(power)-2:
 continue
 Pk = float(power[k])

 side_peak = (power[k-1] < Pk and power[k+1] < Pk)

 fp_lo = 0.99 * fp
 fp_hi = 1.01 * fp
 k_lo = int(np.argmin(np.abs(freqs - fp_lo)))
 k_hi = int(np.argmin(np.abs(freqs - fp_hi)))
 P_lo = float(power[k_lo])
 P_hi = float(power[k_hi])
 jitter_ok = (Pk > max(P_lo, P_hi))

 detected = (Pk >= thr) and side_peak and jitter_ok
 z_like = (Pk - np.median(power[1:])) / (np.median(np.abs(power[1:] - np.median(power[1:]))) or 1e-12)

 hits.append({
 "p": p,
 "fp": fp,
 "bin": k,
 "power": Pk,
 "detected": bool(detected),
 "side_peak": bool(side_peak),
 "jitter_ok": bool(jitter_ok),
 "holdout": bool(in_holdout(p)),
 "zscore_like": float(z_like),
 })




 plt.figure(figsize=(7.5,5.5))
 plt.plot(T_vals, N_data_T, label="N_data(T^2+1/4)")
 plt.plot(T_vals, N_model_T, label="N_model (Weyl fit)")
 plt.plot(T_vals, N_rel_T, label="Relative: data - model")
 plt.plot(T_vals, trend, "--", label=r"Fit to $\alpha \frac{T}{2\pi}\log T + \beta$")
 plt.xlabel("T")
 plt.ylabel("Counts")
 plt.title("Zero-density: Relative determinant vs. Riemann asymptotic")
 plt.legend()
 note = f"A=1/4, B={B:.6g}, b={b:.6g}, c={c:.6g} | alpha={alpha:.6g}, beta={beta:.6g}, R^2={r2:.4f}"
 plt.figtext(0.5, -0.08, note, ha="center", fontsize=9)
 plt.tight_layout()
 plt.savefig(fig_density, dpi=150, bbox_inches="tight")
 plt.close()


 plt.figure(figsize=(7.5,5.5))

 valid = (freqs > 0) & (power > 0)
 plt.loglog(freqs[valid], power[valid], label="FFT power (relative remainder)")

 plt.axhline(thr, linestyle="--", alpha=0.35, label=f"median+{args.kappa}·MAD")

 for p in plist:
 fp = math.log(p) / (2.0 * math.pi)
 if fp <= 0 or fp >= freqs[-1]:
 continue
 plt.axvline(fp, color="r", linestyle=":", alpha=0.25)


 for h in hits:
 if h["detected"]:
 x = freqs[h["bin"]]
 y = power[h["bin"]]
 plt.scatter([x], [y], s=20, c="#2ca02c", zorder=3)

 plt.xlabel("Frequency")
 plt.ylabel("Power")
 plt.title("Prime frequencies in relative determinant remainder")
 plt.legend()
 plt.tight_layout()
 plt.savefig(fig_fft, dpi=150)
 plt.close()


 with open(txt_det, "w") as f:
 f.write("# Prime detections (relative determinant)\n")
 f.write(f"# Threshold: median+{args.kappa}·MAD = {thr:.6g}\n")
 f.write(f"# Holdout interval: [{a}, {b_hold}]\n")
 f.write("# p\tfp\tbin\tpower\tdetected\tside\tjitter\tin_holdout\tz_like\n")
 for h in hits:
 f.write(f"{h['p']}\t{h['fp']:.6g}\t{h['bin']}\t{h['power']:.6g}\t"
 f"{int(h['detected'])}\t{int(h['side_peak'])}\t{int(h['jitter_ok'])}\t"
 f"{int(h['holdout'])}\t{h['zscore_like']:.3f}\n")


 det_total = sum(1 for h in hits if h["detected"])
 det_hold = sum(1 for h in hits if h["detected"] and h["holdout"])
 with open(csv_summary, "w") as f:
 f.write("A,B,b,c,alpha,beta,R2,kappa,pmax,holdout_a,holdout_b,detected_total,detected_holdout\n")
 f.write(f"{A},{B},{b},{c},{alpha},{beta},{r2},{args.kappa},{args.pmax},{a},{b_hold},{det_total},{det_hold}\n")

 print(f"[OK] Saved:\n {fig_density}\n {fig_fft}\n {txt_det}\n {csv_summary}")
 print(f"[INFO] Detected {det_total} primes total; {det_hold} inside holdout [{a},{b_hold}].")
 print(f"[INFO] Tail fit: A=1/4, B={B:.6g}, b={b:.6g}, c={c:.6g}. Riemann fit: alpha={alpha:.6g}, beta={beta:.6g}, R^2={r2:.4f}.")

if __name__ == "__main__":
 main()