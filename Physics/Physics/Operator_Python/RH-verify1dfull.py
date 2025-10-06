#!/usr/bin/env python3
"""
verify_1d_full.py — combined test for 1D operator eigenvalues

- Fits N(T) vs T log T + T + 1 (rescaled Weyl law)
- Runs FFT on oscillatory remainder and checks for log p peaks
"""

import numpy as np, math
from numpy.linalg import lstsq

def primes_upto(n: int):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]: sieve[p*p::p] = False
 return np.nonzero(sieve)[0].tolist()

def fit_rescaled(lams, tail_frac=0.6, top_clip=0.20, bins=16):
 """
 Plateau-stable tail fit.

 Model: N(T)/T ≈ α log(T/T0) + β (γ/T negligible on the tail)
 Steps:
 1) take high-T tail, clip the very top (resolution saturation).
 2) split tail into bins in log T; in each bin, fit local slope d(N/T)/d(log T).
 3) take the median slope over the last half of bins as α_plateau.
 4) solve for β on the same tail (robust), then recover γ on the full set.

 Returns α (uncentered), β (uncentered), γ, T, N(T).
 """
 lams = np.sort(lams[np.isfinite(lams) & (lams > 1e-12)])
 Ntot = len(lams)
 T = np.sqrt(lams)
 N_of_T = np.arange(1, Ntot + 1, dtype=float)


 i_lo = int((1.0 - tail_frac) * Ntot)
 i_hi = int((1.0 - top_clip) * Ntot)
 if i_hi <= i_lo + 4:
 i_hi = min(Ntot, i_lo + 5)

 T_tail = T[i_lo:i_hi]
 N_tail = N_of_T[i_lo:i_hi]
 logT_tail = np.log(T_tail + 1e-12)


 T0 = float(np.exp(np.mean(logT_tail)))
 logT0 = np.log(T0)



 lo, hi = logT_tail[0], logT_tail[-1]
 edges = np.linspace(lo, hi, bins + 1)
 slopes = []
 for b in range(bins):
 a, c = edges[b], edges[b+1]
 mask = (logT_tail >= a) & (logT_tail <= c)
 if mask.sum() >= 4:
 Xb = np.vstack([logT_tail[mask], np.ones(mask.sum())]).T
 Yb = (N_tail[mask] / (T_tail[mask] + 1e-12))

 sb, *_ = lstsq(Xb, Yb, rcond=None)
 slopes.append(float(sb[0]))
 if not slopes:

 Xg = np.vstack([logT_tail, np.ones_like(logT_tail)]).T
 Yg = N_tail / (T_tail + 1e-12)
 sg, *_ = lstsq(Xg, Yg, rcond=None)
 alpha_plateau = float(sg[0])
 else:

 half = max(1, len(slopes)//2)
 alpha_plateau = float(np.median(slopes[-half:]))


 Xc = np.vstack([logT_tail - logT0, np.ones_like(T_tail)]).T
 Yc = N_tail / (T_tail + 1e-12)


 b_vals = Yc - alpha_plateau * Xc[:, 0]
 beta_centered = float(np.mean(b_vals))


 beta_uncentered = beta_centered - alpha_plateau * logT0


 trend = alpha_plateau * T * np.log(T + 1e-12) + beta_uncentered * T
 gamma = float(np.median(N_of_T - trend))



 j1 = i_lo + int(0.80 * (i_hi - i_lo))
 j2 = i_hi - 1
 slope_two_pt = ((N_of_T[j2]/T[j2]) - (N_of_T[j1]/T[j1])) / \
 (np.log(T[j2]+1e-12) - np.log(T[j1]+1e-12))

 print(f"[rescaled] N(T) ≈ {alpha_plateau:.6f} T log T + {beta_uncentered:.6f} T + {gamma:.2f}")
 print(f"[rescaled] plateau α (last-half bins) = {alpha_plateau:.6f}")
 print(f"[rescaled] robust two-point α ≈ {slope_two_pt:.6f}")
 alpha_r_units = alpha_plateau / (2*math.pi)
 print(f"[rescaled] alpha ( T units) ≈ {alpha_plateau:.6f}; alpha in Riemann t-units ≈ {alpha_r_units:.6f}")
 print(f"[rescaled] Riemann target α = 1/(2π) ≈ {1/(2*math.pi):.6f}; target in T-units = 1.000000")

 return alpha_plateau, beta_uncentered, gamma, T, N_of_T

def fft_prime_check(lams, Pmax=101, grid_pts=65536, snr_sigma=3.0):
 """FFT of oscillatory remainder; check log p peaks."""
 lam = np.sort(lams[lams > 1e-12])
 Ntot = len(lam)
 N_of_lam = np.arange(1, Ntot + 1, dtype=float)


 T = np.sqrt(lam)
 Tmax = T[-1]
 Tn = T / Tmax
 Xn = np.vstack([Tn * np.log(Tn + 1e-12), Tn, np.ones_like(Tn)]).T
 coef, *_ = lstsq(Xn, N_of_lam, rcond=None)
 a, b, c = coef

 alpha = a / Tmax
 beta = (b / Tmax) - (a / Tmax) * math.log(Tmax)
 gamma = c

 pred = alpha * T * np.log(T + 1e-12) + beta * T + gamma

 osc = (N_of_lam - pred)
 osc = osc / (np.std(osc) + 1e-12)


 t_uniform = np.linspace(T[0], T[-1], grid_pts)
 osc_uniform = np.interp(t_uniform, T, osc)


 F = np.fft.rfft(osc_uniform - np.mean(osc_uniform))
 freqs = np.fft.rfftfreq(grid_pts, d=(t_uniform[1] - t_uniform[0]))
 power = np.abs(F)


 P = power[1:]
 med = np.median(P)
 mad = np.median(np.abs(P - med)) + 1e-18
 thresh = med + snr_sigma * mad

 prime_hits = []
 plist = primes_upto(Pmax)
 for p in plist:
 f_target = math.log(p) / (2 * math.pi)
 j = np.argmin(np.abs(freqs - f_target))
 pw = power[j]
 prime_hits.append((p, freqs[j], pw, pw >= thresh))

 keep = [p for (p, f, pw, ok) in prime_hits if ok]
 miss = [p for (p, f, pw, ok) in prime_hits if not ok]

 print(f"[fft] threshold = {thresh:.3e}")
 print(f"[fft] primes detected: {keep}")
 print(f"[fft] primes missed: {miss}")

if __name__ == "__main__":
 import sys
 path = sys.argv[1] if len(sys.argv)>1 else "eigs_merged.npy"
 lams = np.load(path)
 lams = np.sort(lams[np.isfinite(lams)])
 print(f"[info] loaded {len(lams)} eigenvalues from {path}")
 fit_rescaled(lams)
 fft_prime_check(lams)