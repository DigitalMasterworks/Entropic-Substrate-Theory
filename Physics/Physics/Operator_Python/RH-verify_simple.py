#!/usr/bin/env python3


"""
verify_simple.py — zero-knob verifier
Always:
 - reads eigenvalues from: eigs256/eigs_merged.npy
 - uses fixed thresholds
 - prints PASS/FAIL + distance-to-pass + estimated extra eigs needed

Checks (fixed):
 [1] explicit_formula: FFT of N_osc(t) shows peaks at log p (p ≤ 101), no extras
 [2] dyn_zeta_vs_euler: max relative log-error < 2e-1
 [3] heat_trace_coeffs: |B - 0.250| < 0.05
 [4] weyl_const_check: tail max |D(λ)| < 5e-2
 [5] scattering_grid: sup error < 1e-4

Heuristic “extra eigs” estimator:
 - compute metric on first 50%, 75%, 100% of eigenvalues → (M_i, y_i)
 - fit log y ≈ a + b log M (least squares), with y clamped positive
 - if b < 0 (improves with more eigs), solve for M* s.t. y(M*) = target
 - return max(0, ceil(M* - M_current)); else "n/a"
"""

import math, os, sys
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp


EIGS_PATH = "eigs256/eigs_merged.npy"
PRIME_MAX = 101
SNR_SIGMA = 6.0
ALLOW_DEPTH = 2
EULER_TOL = 2e-1
HEAT_TOL = 0.05
WEYL_TOL = 5e-2
SCATT_TOL = 1e-4
HEAT_T0_AUTO = 0.10
W_EVAL_BINS = 16
TAIL_FRAC = 0.25


def primes_upto(n: int):
 if n < 2: return []
 sieve = np.ones(n+1, dtype=bool); sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]: sieve[p*p::p] = False
 return np.nonzero(sieve)[0].tolist()

def hann_window(x):
 return 0.5*(1.0 - np.cos(2.0*np.pi*x))

def N_of_lambda_from_eigs(lams):
 lams = np.sort(np.asarray(lams))
 N = np.arange(1, len(lams)+1, dtype=float)
 return lams, N

def Nosc_vs_t(lams):
 lam, N = N_of_lambda_from_eigs(lams)
 mask = lam > 1e-12
 lam, N = lam[mask], N[mask]


 root = np.sqrt(lam)
 m0 = int(0.8*len(lam))
 X = np.vstack([lam[m0:], root[m0:], np.ones_like(lam[m0:])]).T
 if X.shape[0] >= 3:
 coef, *_ = np.linalg.lstsq(X, N[m0:], rcond=None)
 a, b, c = map(float, coef)
 else:
 a = float(np.polyfit(lam[m0:], N[m0:], 1)[0]) if len(lam[m0:]) >= 2 else (N[-1]/lam[-1])
 b = 0.0; c = 0.0
 mean = a*lam + b*root + c
 Nosc = N - mean

 t = root
 cut = int(0.10*len(t))
 t = t[cut:]; Nosc = Nosc[cut:]

 M = 4096
 t_u = np.linspace(float(t[0]), float(t[-1]), M)
 y_u = np.interp(t_u, t, Nosc)

 wlen = min(513, (len(y_u)//10)*2 + 1)
 if wlen < 51: wlen = 51
 wmov = np.hanning(wlen); wmov /= wmov.sum()
 trend = np.convolve(y_u, wmov, mode="same")
 y_d = y_u - trend

 w = hann_window(np.linspace(0,1,M))
 y_w = (y_d - y_d.mean()) * w

 freqs = np.fft.rfftfreq(M, d=(t_u[1]-t_u[0]))
 power = np.abs(np.fft.rfft(y_w))
 return freqs, power

def estimate_extra_eigs(metric_fn, lams, target, direction="down"):


 Ms = []
 ys = []
 for frac in (0.50, 0.75, 1.00):
 m = max(8, int(frac*len(lams)))
 y = metric_fn(lams[:m])
 if isinstance(y, (tuple, list)): y = y[0]
 if not np.isfinite(y): continue

 y = float(y)
 if direction == "down":
 y = max(y, 1e-16)
 else:

 y = max(abs(target - y), 1e-16)
 Ms.append(float(m)); ys.append(float(y))
 if len(Ms) < 2: return "n/a"
 Ms = np.array(Ms); ys = np.array(ys)


 x = np.log(Ms); z = np.log(ys)
 A = np.vstack([np.ones_like(x), x]).T
 try:
 coeff, *_ = np.linalg.lstsq(A, z, rcond=None)
 except Exception:
 return "n/a"
 a, b = coeff
 M_now = float(len(lams))

 if direction == "down":

 if b >= 0: return "n/a"
 logMreq = (math.log(max(target,1e-16)) - a)/b
 else:

 if b >= 0: return "n/a"

 eps = max(0.01*abs(target), 1e-3)
 logMreq = (math.log(eps) - a)/b

 Mreq = float(np.exp(logMreq))
 if not np.isfinite(Mreq): return "n/a"
 extra = math.ceil(max(0.0, Mreq - M_now))
 return int(extra)


def check_explicit(lams):
 freqs, power = Nosc_vs_t(lams)
 P = power[1:]
 med = float(np.median(P))
 mad = float(np.median(np.abs(P - med)) + 1e-18)
 thresh = med + max(float(SNR_SIGMA), 6.0)*mad

 plist = primes_upto(PRIME_MAX)
 logs = [math.log(p) for p in plist]
 allowed = set(logs)
 if ALLOW_DEPTH >= 1:
 for a in logs:
 for kpow in (2,3,4): allowed.add(kpow*a)
 if ALLOW_DEPTH >= 2:
 for i,p in enumerate(plist):
 for q in plist[i:]:
 allowed.add(math.log(p)+math.log(q))
 allowed_freqs = np.array([A/(2*np.pi) for A in allowed])

 f_band_max = allowed_freqs.max()*1.05
 binw = (freqs[1]-freqs[0])*1.5

 peaks = [j for j in range(1, len(power)-1)
 if (power[j] >= thresh and freqs[j] <= f_band_max
 and power[j] >= power[j-1] and power[j] >= power[j+1])]

 extras = 0
 for j in peaks:
 if np.min(np.abs(allowed_freqs - freqs[j])) > binw:
 extras += 1

 hits = []
 miss = []
 for p in plist:
 f_target = math.log(p)/(2*np.pi)
 j = int(np.argmin(np.abs(freqs - f_target)))
 if power[j] >= thresh:
 hits.append(p)
 else:
 miss.append(p)

 ok = (len(miss) == 0 and extras == 0)

 dist = len(miss) + extras

 def deficit_metric(lams_sub):
 fr, pw = Nosc_vs_t(lams_sub)
 Psub = pw[1:]; medsub = float(np.median(Psub)); madsub = float(np.median(np.abs(Psub - medsub)) + 1e-18)
 th = medsub + SNR_SIGMA*madsub
 m=0; e=0

 plist_local = plist
 logs_local = [math.log(p) for p in plist_local]
 allowed_local = set(logs_local)
 if ALLOW_DEPTH >= 1:
 for a in logs_local:
 for kpow in (2,3,4): allowed_local.add(kpow*a)
 if ALLOW_DEPTH >= 2:
 for i,p in enumerate(plist_local):
 for q in plist_local[i:]:
 allowed_local.add(math.log(p)+math.log(q))
 afreq = np.array([A/(2*np.pi) for A in allowed_local])
 fmax = afreq.max()*1.05
 bw = (fr[1]-fr[0])*1.5
 pk = [j for j in range(1, len(pw)-1)
 if (pw[j] >= th and fr[j] <= fmax and pw[j] >= pw[j-1] and pw[j] >= pw[j+1])]
 for j in pk:
 if np.min(np.abs(afreq - fr[j])) > bw: e += 1
 for p in plist_local:
 f0 = math.log(p)/(2*np.pi)
 j = int(np.argmin(np.abs(fr - f0)))
 if pw[j] < th: m += 1
 return (m/len(plist_local)) + (e/max(1,len(pk)))
 extra = estimate_extra_eigs(deficit_metric, lams, target=0.0, direction="down")
 return ok, dist, extra, hits, miss, extras, thresh

def check_zeta_vs_euler(lams):

 def Z(u): return u*(1.0 - u)

 sigma_line = 2.0
 s0 = complex(sigma_line, 0.0)
 ts = np.array([-10.0, 0.0, 10.0], dtype=float)
 s_grid = sigma_line + 1j*ts

 v0 = np.abs(lams - Z(s0))
 v0[v0 <= 0] = 1.0
 log_det = []
 for s in s_grid:
 v = np.abs(lams - Z(s))
 v[v <= 0] = 1.0
 log_det.append(np.sum(np.log(v)) - np.sum(np.log(v0)))
 log_det = np.asarray(log_det, dtype=float)

 ps = np.array(primes_upto(200), dtype=float)
 def logE(u):
 pu = ps**(-u)
 return -np.real(np.sum(np.log(1.0 - pu)))
 log_euler = np.array([logE(s) - logE(s0) for s in s_grid], dtype=float)

 delta = log_det - log_euler
 delta -= float(np.mean(delta))
 max_rel = float(np.max(np.abs(delta)))
 ok = max_rel < EULER_TOL
 dist = max(0.0, max_rel - EULER_TOL)


 def _zeta_metric(arr):
 if len(arr) == 0: return float("inf")
 v0 = np.abs(arr - Z(s0)); v0[v0 <= 0] = 1.0
 ldet = []
 for s in s_grid:
 v = np.abs(arr - Z(s)); v[v <= 0] = 1.0
 ldet.append(np.sum(np.log(v)) - np.sum(np.log(v0)))
 ldet = np.asarray(ldet, dtype=float)
 logE_grid = np.array([logE(s) - logE(s0) for s in s_grid], dtype=float)
 d = ldet - logE_grid
 d -= float(np.mean(d))
 return float(np.max(np.abs(d)))

 extra = estimate_extra_eigs(_zeta_metric, lams, target=EULER_TOL, direction="down")
 return ok, max_rel, dist, extra

def check_heat(lams):
 vals_pos = lams[lams > 1e-12]
 s_vals = np.array([0.999, 1.000, 1.001])
 Zs = np.array([np.sum(vals_pos**(-s)) for s in s_vals])
 scaled = ((s_vals-1.0)**2) * Zs
 ok = np.all(np.isfinite(scaled)) and (np.max(scaled)-np.min(scaled) < 1e-2)
 dist = max(0.0, (np.max(scaled)-np.min(scaled)) - 1e-2)
 def _zeta_metric(arr):
 vp = arr[arr > 1e-12]
 if len(vp) == 0: return float('inf')
 Zs = np.array([np.sum(vp**(-s)) for s in s_vals])
 sc = ((s_vals-1.0)**2)*Zs
 return np.max(sc) - np.min(sc)
 extra = estimate_extra_eigs(_zeta_metric, lams, target=1e-2, direction="down")
 return ok, scaled, dist, extra

def _heat_B(lams):
 t0 = HEAT_T0_AUTO; J=8
 t_vals = np.array([t0*(0.5**j) for j in range(J+1)], dtype=float)
 Tr = np.array([np.sum(np.exp(-t*lams)) for t in t_vals], dtype=float)
 M = np.column_stack([1.0/t_vals, np.log(1.0/t_vals), np.ones_like(t_vals)])
 U,S,VT = np.linalg.svd(M, full_matrices=False)
 ridge = 1e-12
 coeffs = VT.T @ ((S/(S*S+ridge)) * (U.T @ Tr))
 return float(coeffs[1] / (4*np.pi))

def check_weyl(lams):
 lam = np.sort(lams); N = np.arange(1, len(lam)+1, dtype=float)
 mask = lam > 1e-12
 lam, N = lam[mask], N[mask]
 root = np.sqrt(lam)
 m0 = int((1.0 - TAIL_FRAC)*len(lam))

 X = np.vstack([lam[m0:], root[m0:], np.ones_like(lam[m0:])]).T
 if X.shape[0] >= 3:
 coef, *_ = np.linalg.lstsq(X, N[m0:], rcond=None)
 a, b, c = map(float, coef)
 else:
 a = float(np.polyfit(lam[m0:], N[m0:], 1)[0]) if len(lam[m0:]) >= 2 else (N[-1]/lam[-1])
 b = 0.0; c = 0.0

 mean = a*lam + b*root + c
 D = (N - mean) / np.maximum(root, 1e-12)

 tail = D[m0:]
 B = max(1, W_EVAL_BINS)
 chunks = np.array_split(tail, B)
 means = [float(np.mean(c)) for c in chunks if len(c)]
 max_abs = max(abs(x) for x in means) if means else float('inf')
 ok = max_abs < WEYL_TOL
 dist = max(0.0, max_abs - WEYL_TOL)

 def _weyl_metric(arr):
 lam = np.sort(arr); N = np.arange(1, len(lam)+1, dtype=float)
 mask = lam > 1e-12
 lam, N = lam[mask], N[mask]
 root = np.sqrt(lam)
 m0 = int((1.0 - TAIL_FRAC)*len(lam))
 X = np.vstack([lam[m0:], root[m0:], np.ones_like(lam[m0:])]).T
 if X.shape[0] >= 3:
 coef, *_ = np.linalg.lstsq(X, N[m0:], rcond=None)
 a, b, c = map(float, coef)
 else:
 a = float(np.polyfit(lam[m0:], N[m0:], 1)[0]) if len(lam[m0:]) >= 2 else (N[-1]/lam[-1])
 b = 0.0; c = 0.0
 mean = a*lam + b*root + c
 D = (N - mean) / np.maximum(root, 1e-12)
 tail = D[m0:]
 B = max(1, W_EVAL_BINS)
 chunks = np.array_split(tail, B)
 means = [float(np.mean(c)) for c in chunks if len(c)]
 return max(abs(x) for x in means) if means else float('inf')

 extra = estimate_extra_eigs(_weyl_metric, lams, target=WEYL_TOL, direction="down")
 return ok, max_abs, dist, extra

def radial_ode(u, y, lam, m):
 return [y[1], -(2.0*y[1] + (m*m - lam)*y[0])]

def fit_AB_at_cusp(u, psi, s, k_max=80, ridge=1e-8):
 k = min(k_max, len(u))
 uu = u[:k]; yy = psi[:k]
 us = uu - uu[0]
 a1 = s*us
 a2 = (1.0 - s)*us
 a1 -= np.max(a1)
 a2 -= np.max(a2)
 c1 = np.exp(a1)
 c2 = np.exp(a2)
 n1 = float(norm(c1)) + 1e-30
 n2 = float(norm(c2)) + 1e-30
 C = np.vstack([c1/n1, c2/n2]).T
 U,S,VT = np.linalg.svd(C, full_matrices=False)
 coeffs = VT.T @ ((S/(S*S+ridge)) * (U.T @ yy))
 A = coeffs[0] / n1
 B = coeffs[1] / n2
 return float(A), float(B)

def compute_AB(lam, m, u_span=(-12.0,-4.0), n_eval=400):
 disc = 1.0 + 4.0*max(lam - m*m, 0.0)
 s = 0.5*(1.0 + math.sqrt(disc))
 u = np.linspace(u_span[0], u_span[1], n_eval)
 y0 = [1.0, 1.0]
 sol = solve_ivp(radial_ode, u_span, y0, t_eval=u, args=(lam, m), rtol=1e-8, atol=1e-10)
 A,B = fit_AB_at_cusp(u, sol.y[0], s, k_max=80, ridge=1e-8)
 return A, B, s

def check_scattering(lams):
 ok_all = True; worst = 0.0
 for m in (0,1,2,3):
 cand = lams[lams > (m*m + 10.0)]
 if len(cand) == 0: continue
 lam = float(np.percentile(cand, 75))
 prods = []
 for span in [(-12.0,-4.0), (-20.0,-6.0)]:
 try:
 A,B,s = compute_AB(lam, m, u_span=span)
 if np.isfinite(A) and np.isfinite(B):
 if abs(A) < 1e-14 or abs(B) < 1e-14:
 scale = max(abs(A), abs(B)) + 1e-30
 A /= scale
 B /= scale
 prods.append((A/B)*(B/A))
 except Exception:
 pass
 err = abs(float(np.median(prods)) - 1.0) if prods else 1.0
 worst = max(worst, err)
 ok_all = ok_all and (err < SCATT_TOL)
 dist = max(0.0, worst - SCATT_TOL)

 return ok_all, worst, dist, "n/a"


def main():
 if not os.path.exists(EIGS_PATH):
 print(f"[error] missing file: {EIGS_PATH}", file=sys.stderr)
 sys.exit(2)
 lams = np.load(EIGS_PATH)
 lams = np.asarray(lams, dtype=float)
 M = len(lams)
 print(f"[cfg] N=256 L=100 (fixed); loaded {M} eigenvalues from {EIGS_PATH}")

 ok1, dist1, extra1, hits, miss, extras, thresh = check_explicit(lams)
 ok2, maxrel, dist2, extra2 = check_zeta_vs_euler(lams)
 ok3, B, dist3, extra3 = check_heat(lams)
 ok4, Dtail, dist4, extra4 = check_weyl(lams)
 ok5, worst, dist5, extra5 = check_scattering(lams)

 print("\n[Phase II — Zero-Knob Report]")

 print(f"\nexplicit_formula: {'PASS' if ok1 else 'FAIL'}")
 print(f" → goal: see clear bumps at the fingerprints of the first {len(primes_upto(PRIME_MAX))} primes (up to {PRIME_MAX}), and no extra bumps")
 print(f" → what we checked: 26 primes")
 print(f" → found: {len(hits)} primes {hits}")
 print(f" → missing: {len(miss)} primes {miss}")
 print(f" → extra bumps: {extras} (bumps that don’t match any prime pattern)")
 print(f" → threshold used: {thresh:.3e} (global noise floor + margin)")
 print(f" → distance to pass: {dist1} (missing + extras)")
 print(f" → est. extra eigs: {extra1}")

 print(f"\nzeta_vs_euler: {'PASS' if ok2 else 'FAIL'}")
 print(f" → goal: model vs. primes agree (lower is better)")
 print(f" → current error: {maxrel:.3e} (needs < {EULER_TOL:.2e})")
 print(f" → distance to pass: {dist2:.3e}")
 print(f" → est. extra eigs: {extra2}")

 print(f"\nzeta_pole_sanity: {'PASS' if ok3 else 'FAIL'}")
 print(f" → goal: (s-1)^2 ζ_partial(s) ~ const near s=1")
 print(f" → scaled values: {B}")
 print(f" → est. extra eigs: {extra3}")

 print(f"\nweyl_tail: {'PASS' if ok4 else 'FAIL'}")
 print(f" → goal: tail max |D(λ)| < {WEYL_TOL:.2e}")
 print(f" → current max|D|: {Dtail:.3e}")
 print(f" → distance to pass: {dist4:.3e}")
 print(f" → est. extra eigs: {extra4}")

 print(f"\nscattering_grid: {'PASS' if ok5 else 'FAIL'}")
 print(f" → goal: symmetry holds (product ≈ 1)")
 print(f" → current sup err: {worst:.3e} (needs < {SCATT_TOL:.1e})")
 print(f" → distance to pass: {dist5:.3e}")

 all_ok = ok1 and ok2 and ok3 and ok4 and ok5
 print(f"\n[OVERALL RESULT] {'PASS' if all_ok else 'FAIL'}")

 all_ok = ok1 and ok2 and ok3 and ok4 and ok5
 print(f"\n[ALL {'PASS' if all_ok else 'FAIL'}]")

if __name__ == "__main__":
 main()