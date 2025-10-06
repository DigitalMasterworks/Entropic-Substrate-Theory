#!/usr/bin/env python3
"""
Final figure maker: relative determinant + prime FFT on harvested eigenvalues.

- Locks A1 = 1/(2π) in the 1D Weyl law.
- Fits only the top 20% of the tail.
- Outputs side-by-side density & FFT figures with R², detections, etc.
"""

import os, math, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_eigs(path):
 arr = np.load(path)
 arr = np.array(arr).ravel()
 arr = arr[np.isfinite(arr)]
 arr = arr[arr > 0.0]
 arr = np.unique(np.sort(arr))
 return arr

def counting_function(sorted_lams, lam_vals):
 return np.searchsorted(sorted_lams, lam_vals, side="right").astype(float)

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

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--eigs", required=True)
 ap.add_argument("--out", default="./out_final")
 ap.add_argument("--tailpct", type=float, default=0.8, help="Top tail fraction to use (e.g. 0.8 keeps top 20%)")
 ap.add_argument("--fftpoints", type=int, default=2**16)
 ap.add_argument("--pmax", type=int, default=503)
 ap.add_argument("--kappa", type=float, default=3.0)
 args = ap.parse_args()

 os.makedirs(args.out, exist_ok=True)
 f1 = os.path.join(args.out, "final_density_fft.png")
 txt = os.path.join(args.out, "prime_detections_final.txt")


 lam = load_eigs(args.eigs)


 A1 = 1.0/(2*math.pi)


 n = lam.size
 i0 = int(args.tailpct * n)
 lam_tail = lam[i0:]
 N_tail = np.arange(i0+1, n+1, dtype=float)


 s = np.sqrt(lam_tail)
 L = np.log(lam_tail); L[np.isnan(L)|np.isinf(L)] = 0.0
 X = np.column_stack([s, L, np.ones_like(lam_tail)])
 y = N_tail - A1*(s*L)
 coef, *_ = np.linalg.lstsq(X, y, rcond=None)
 B1, C1, D1 = coef.tolist()
 N_model = A1*s*L + B1*s + C1*L + D1
 N_rel = N_tail - N_model


 T = np.sqrt(lam_tail-0.25)
 shape = (T/(2*np.pi))*np.log(T)
 alpha, beta, r2 = affine_fit(shape, N_rel)
 trend = alpha*shape + beta
 rem = N_rel - trend
 rem -= np.mean(rem)


 Tg = np.linspace(T[0], T[-1], args.fftpoints)
 Rg = np.interp(Tg, T, rem)
 W = tukey(args.fftpoints, 0.5)
 F = np.fft.rfft(Rg*W)
 f = np.fft.rfftfreq(args.fftpoints, d=(Tg[1]-Tg[0]))
 P = np.abs(F)**2; P[0]=0
 thr, med, mad = robust_thr(P[1:], args.kappa)

 primes = primes_up_to(args.pmax)
 detections=[]
 for p in primes:
 fp = math.log(p)/(2*math.pi)
 if fp<=f[1] or fp>=f[-1]: continue
 k = int(np.argmin(np.abs(f-fp)))
 power = float(P[k])
 side = (P[k] > P[k-1] and P[k] > P[k+1])
 detected = (power>=thr and side)
 detections.append((p,fp,power,detected))


 fig, axs = plt.subplots(1,2,figsize=(13,5))
 axs[0].plot(T, N_rel, label="Relative counts")
 axs[0].plot(T, trend, "--", label=r"Fit $\alpha\frac{T}{2\pi}\log T + \beta$")
 axs[0].set_xlabel("T"); axs[0].set_ylabel("Relative counts")
 axs[0].set_title(f"Zero-density fit (R²={r2:.3f})")
 axs[0].legend()

 valid=(f>0)&(P>0)
 axs[1].loglog(f[valid], P[valid], label="FFT power")
 axs[1].axhline(thr, ls="--", alpha=0.4, label=f"thr={thr:.2e}")
 for p,fp,power,det in detections:
 if det:
 axs[1].axvline(fp,color="r",ls=":",alpha=0.3)
 axs[1].scatter([fp],[power],c="g",s=15)
 axs[1].set_xlabel("frequency (log p / 2π)")
 axs[1].set_ylabel("power")
 axs[1].set_title("Prime frequencies (relative determinant)")
 axs[1].legend()
 plt.tight_layout(); plt.savefig(f1,dpi=150); plt.close()


 with open(txt,"w") as g:
 g.write("# p\tfreq\tpower\tdetected\n")
 for p,fp,power,det in detections:
 g.write(f"{p}\t{fp:.6g}\t{power:.6g}\t{int(det)}\n")

 print("[OK] Saved:")
 print(" ",f1)
 print(" ",txt)
 print(f"[INFO] Fit: alpha={alpha:.6g}, beta={beta:.6g}, R²={r2:.3f}")
 print(f"[INFO] Detections: {sum(1 for *_,det in detections if det)} primes detected")

if __name__=="__main__":
 main()