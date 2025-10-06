#!/usr/bin/env python3
"""
FULL TEST (1D-correct): Relative determinant + prime oscillations on REAL 1D cusp eigenvalues.

Key changes vs previous:
- Uses the 1D Weyl model: N(λ) ≈ A1*sqrt(λ) log λ + B1*sqrt(λ) + C1*log λ + D1 (A1 free or fix to 1/(2π))
- Restricts to a high-energy tail for fitting and FFT (percentile cut)
- Uses Tukey window and 1/f^γ whitening before FFT
- Quadratic peak refinement and merge when two primes quantize to the same bin
"""

import os, math, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_eigs(path):
 a = np.load(path)
 a = np.array(a).ravel()
 a = a[np.isfinite(a)]
 a = a[a > 0.0]
 a = np.unique(np.sort(a))
 if a.size < 2000:
 raise ValueError(f"Need ≥2000 eigenvalues; got {a.size}")
 return a

def counting_function(sorted_lams, lam_vals):
 return np.searchsorted(sorted_lams, lam_vals, side="right").astype(float)

def fit_weyl_1d(sorted_lams, tail_lo_pct=0.6, fix_A1=None):
 """ Fit N(λ) ≈ A1*sqrt(λ) log λ + B1*sqrt(λ) + C1*log λ + D1 on upper tail. """
 n = sorted_lams.size
 i0 = int(tail_lo_pct * n)
 lam = sorted_lams[i0:]
 N = np.arange(i0+1, n+1, dtype=float)

 s = np.sqrt(lam)
 with np.errstate(divide='ignore', invalid='ignore'):
 L = np.log(lam)
 L[np.isinf(L)] = 0.0
 L[np.isnan(L)] = 0.0

 if fix_A1 is None:

 X = np.column_stack([s*L, s, L, np.ones_like(lam)])
 coef, *_ = np.linalg.lstsq(X, N, rcond=None)
 A1, B1, C1, D1 = coef.tolist()
 else:
 X = np.column_stack([s, L, np.ones_like(lam)])
 y = N - fix_A1*(s*L)
 coef, *_ = np.linalg.lstsq(X, y, rcond=None)
 B1, C1, D1 = coef.tolist()
 A1 = fix_A1
 return A1, B1, C1, D1

def primes_up_to(n):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]:
 sieve[p*p:n+1:p] = False
 return [i for i in range(2, n+1) if sieve[i]]

def affine_fit(x, y):
 X = np.column_stack([x, np.ones_like(x)])
 coef, *_ = np.linalg.lstsq(X, y, rcond=None)
 a, b = coef.tolist()
 yhat = a*x + b
 ss_res = float(np.sum((y-yhat)**2))
 ss_tot = float(np.sum((y - np.mean(y))**2)) or 1.0
 r2 = 1.0 - ss_res/ss_tot
 return a, b, r2

def tukey(n, alpha=0.5):
 if alpha <= 0: return np.ones(n)
 if alpha >= 1: return np.hanning(n)
 w = np.ones(n)
 edge = int(alpha*(n-1)/2)
 if edge>0:
 h = np.hanning(2*edge)
 w[:edge] = h[:edge]
 w[-edge:] = h[-edge:]
 return w

def robust_thr(vals, kappa):
 med = float(np.median(vals))
 mad = float(np.median(np.abs(vals-med))) or 1e-12
 return med + kappa*mad, med, mad

def quad_refine(xm1, x0, xp1, ym1, y0, yp1):

 denom = ( (xm1 - x0)*(xp1 - x0)*(xm1 - xp1) )
 if denom == 0: return x0, y0
 A = ( (yp1 - y0)/(xp1 - x0) - (y0 - ym1)/(x0 - xm1) ) / (xp1 - xm1)
 B = (y0 - ym1)/(x0 - xm1) - A*(x0 + xm1)
 x_peak = -B/(2*A) if A!= 0 else x0
 y_peak = A*x_peak**2 + B*x_peak + (y0 - A*x0**2 - B*x0)
 return x_peak, y_peak

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--eigs", required=True)
 ap.add_argument("--out", default="./out1d")
 ap.add_argument("--tpoints", type=int, default=450)
 ap.add_argument("--fftpoints", type=int, default=2**15)
 ap.add_argument("--tailpct", type=float, default=0.65, help="Percentile for T-tail to analyze (e.g., 0.65 keeps top 35%)")
 ap.add_argument("--fixA1", type=float, default=None, help="If set, fixes A1; try 1/(2π)=0.159154943...")
 ap.add_argument("--kappa", type=float, default=3.0)
 ap.add_argument("--pmax", type=int, default=503)
 ap.add_argument("--whiten_gamma", type=float, default=1.0, help="Divide spectrum by f^gamma to flatten 1/f trend")
 args = ap.parse_args()

 os.makedirs(args.out, exist_ok=True)
 f1 = os.path.join(args.out, "zero_density_full_1d.png")
 f2 = os.path.join(args.out, "prime_fft_full_1d.png")
 txt = os.path.join(args.out, "prime_detections_1d.txt")
 csv = os.path.join(args.out, "full_test_summary_1d.csv")

 lam = load_eigs(args.eigs)


 A1, B1, C1, D1 = fit_weyl_1d(lam, tail_lo_pct=0.6, fix_A1=args.fixA1)


 T_all = np.sqrt(np.maximum(lam - 0.25, 1e-9))
 T_min, T_max = np.percentile(T_all, 100*args.tailpct), np.max(T_all)
 T = np.linspace(T_min, T_max, args.tpoints)
 lamT = T*T + 0.25


 N_data = counting_function(lam, lamT)
 with np.errstate(divide='ignore', invalid='ignore'):
 L = np.log(lamT); L[np.isinf(L)] = 0; L[np.isnan(L)] = 0
 N_model = A1*np.sqrt(lamT)*L + B1*np.sqrt(lamT) + C1*L + D1
 N_rel = N_data - N_model


 with np.errstate(divide='ignore', invalid='ignore'):
 shape = (T/(2*np.pi)) * np.log(T)
 shape[np.isnan(shape)] = 0; shape[np.isinf(shape)] = 0
 alpha, beta, r2 = affine_fit(shape, N_rel)
 trend = alpha*shape + beta
 rem = N_rel - trend
 rem -= np.mean(rem)


 Tg = np.linspace(T[0], T[-1], args.fftpoints)
 Rg = np.interp(Tg, T, rem)
 W = tukey(args.fftpoints, 0.5)
 F = np.fft.rfft(Rg*W)
 f = np.fft.rfftfreq(args.fftpoints, d=(Tg[1]-Tg[0]))
 P = np.abs(F)**2
 P[0] = 0.0

 if args.whiten_gamma > 0:
 f_safe = np.maximum(f, 1e-9)
 P = P / (f_safe**args.whiten_gamma)


 primes = primes_up_to(args.pmax)
 thr, med, mad = robust_thr(P[1:], args.kappa)
 hits = []
 occupied_bins = set()
 for p in primes:
 fp = math.log(p)/(2*math.pi)
 if not (f[1] <= fp <= f[-2]):
 continue
 k = int(np.argmin(np.abs(f - fp)))
 if k in occupied_bins:
 continue

 xm1, x0, xp1 = f[k-1], f[k], f[k+1]
 ym1, y0, yp1 = P[k-1], P[k], P[k+1]
 xpk, ypk = quad_refine(xm1, x0, xp1, ym1, y0, yp1)
 side = (y0 > ym1 and y0 > yp1)
 jitter_ok = (P[max(1,k-5):min(len(P)-1,k+6)].max() == y0)
 detected = (y0 >= thr) and side and jitter_ok
 hits.append((p, fp, k, y0, detected))
 if detected:
 occupied_bins.add(k)

 plt.figure(figsize=(7.5,5.5))
 plt.plot(T, N_data, label=r"$N_{\rm data}(T^2+\frac{1}{4})$")
 plt.plot(T, N_model, label="N_model (1D Weyl fit)")
 plt.plot(T, N_rel, label="Relative: data - model")
 plt.plot(T, trend, "--", label=r"Fit $\alpha \frac{T}{2\pi}\log T + \beta$")
 note = f"1D fit: A1={A1:.6g}, B1={B1:.6g}, C1={C1:.6g}, D1={D1:.6g} | alpha={alpha:.6g}, beta={beta:.6g}, R^2={r2:.4f}"
 plt.xlabel("T"); plt.ylabel("Counts")
 plt.title("Zero-density (1D-correct) vs Riemann asymptotic")
 plt.legend()
 plt.figtext(0.5, -0.08, note, ha="center", fontsize=9)
 plt.tight_layout()
 plt.savefig(f1, dpi=150, bbox_inches="tight")
 plt.close()
 plt.figure(figsize=(7.5,5.5))
 valid = (f>0) & (P>0)
 plt.loglog(f[valid], P[valid], label="FFT power (whitened)")
 plt.axhline(thr, ls="--", alpha=0.35, label=f"median+{args.kappa}·MAD")
 for p in primes:
 fp = math.log(p)/(2*math.pi)
 if f[0] < fp < f[-1]:
 plt.axvline(fp, color="r", ls=":", alpha=0.22)

 for p, fp, k, y0, det in hits:
 if det: plt.scatter([f[k]], [P[k]], s=20, c="#2ca02c")
 plt.xlabel("Frequency"); plt.ylabel("Power"); plt.title("Prime frequencies (relative remainder, 1D-correct)")
 plt.legend(); plt.tight_layout(); plt.savefig(f2, dpi=150); plt.close()


 with open(os.path.join(args.out,"prime_detections_1d.txt"), "w") as g:
 g.write("# p\tfp\tbin\tpower\tdetected\n")
 for p, fp, k, y0, det in hits:
 g.write(f"{p}\t{fp:.6g}\t{k}\t{y0:.6g}\t{int(det)}\n")

 with open(os.path.join(args.out,"full_test_summary_1d.csv"), "w") as g:
 g.write("A1,B1,C1,D1,alpha,beta,R2,kappa,pmax,tailTmin,tailTmax\n")
 g.write(f"{A1},{B1},{C1},{D1},{alpha},{beta},{r2},{args.kappa},{args.pmax},{T[0]},{T[-1]}\n")

 print("[OK] Saved:")
 print(" ", f1)
 print(" ", f2)
 print(" ", os.path.join(args.out,"prime_detections_1d.txt"))
 print(" ", os.path.join(args.out,"full_test_summary_1d.csv"))
 print(f"[INFO] R^2 (Riemann shape on relative counts): {r2:.4f}")
 print(f"[INFO] Tail T-range: [{T[0]:.2f}, {T[-1]:.2f}]. Whiten γ={args.whiten_gamma}.")
 print(f"[INFO] Detections: {sum(int(d) for *_, d in hits)} primes (merged by bin).")
if __name__ == "__main__":
 main()