
import numpy as np, cmath
from mpmath import pi, gamma, zeta, log as mlog


T_MIN, T_MAX, T_POINTS = -5.0, 5.0, 41
KMAX_LIST = (8000, 12000, 16000)
FIT_A0, FIT_B0 = 0.81373, 0.00
DO_TAIL_FIT = True
A_GRID = 7
B_GRID = 7
A_SPAN, B_LO, B_HI = 0.03, -0.30, 0.60


LAMS = np.load("eigs_merged.npy")
LAMS = np.sort(LAMS[np.isfinite(LAMS)])

def chi(s, a, b):

 return (pi**(-a*s)) * (gamma(s/2)**b) * (gamma((1-s)/2)**b)

def clog(x): return complex(mlog(x))
def log_zeta(s): return clog(zeta(s))

def logdet_spec_vec(ts, kmax, anchor=0.5+0j):
 """vectorized complex logdet( H - s(1-s) )-logdet at anchor, for s=1/2+it over ts."""
 lam = LAMS[:kmax]
 s = 0.5 + 1j*ts
 z = s*(1-s)
 z0 = anchor*(1-anchor)

 out = np.zeros_like(s, dtype=complex)
 for l in lam:
 out += np.log(l - z) - np.log(l - z0)
 return out

def unwrap_along_t(arr):
 """unwrap Im(log) along t to remove ±2π jumps."""
 out = arr.copy()
 im = out.imag.copy()
 for i in range(1, len(im)):
 delta = im[i] - im[i-1]
 im[i] -= 2*np.pi * np.round(delta/(2*np.pi))
 return out.real + 1j*im

def second_diff(arr):
 return arr[2:] - 2*arr[1:-1] + arr[:-2]

def fit_quadratic_t(ts, y):
 """fit a + b t + c t^2; return fitted values"""
 T = np.vstack([np.ones_like(ts), ts, ts*ts]).T
 coeff, *_ = np.linalg.lstsq(T, y, rcond=None)
 return T @ coeff

def precompute_logs(ts):
 """precompute spectral logdet (by kmax) and orbit log ζ (shared), unwrapped."""
 s = 0.5 + 1j*ts
 lz = np.array([log_zeta(si) for si in s], dtype=complex)
 spec = []
 for kmax in KMAX_LIST:
 ldet = logdet_spec_vec(ts, kmax=kmax)
 if DO_TAIL_FIT:

 re = ldet.real
 re -= fit_quadratic_t(ts, re)
 ldet = (re + 1j*ldet.imag)
 spec.append(unwrap_along_t(ldet))
 return np.array(spec), unwrap_along_t(lz)

def build_logF(ts, spec_logs, lz, a, b):
 """logF = logdet - log chi - log ζ, for each kmax config; unwrap per config"""
 s = 0.5 + 1j*ts
 lchi = np.array([clog(chi(si, a, b)) for si in s], dtype=complex)
 logs = []
 for ldet in spec_logs:
 vals = ldet - lchi - lz
 logs.append(unwrap_along_t(vals))
 return np.stack(logs, axis=0)

def score_delta2_mean(logF):
 """L2 score of zero-centered Δ2 mean across t (lower is better)."""
 d2 = np.stack([second_diff(row) for row in logF], axis=0)
 d2_mean = d2.mean(axis=0)
 d2_mean -= d2_mean.mean()
 return float(np.sum(np.abs(d2_mean)**2)), d2_mean, d2

def fit_chi_params(ts, spec_logs, lz, a0=FIT_A0, b0=FIT_B0):
 a_vals = np.linspace(a0 - A_SPAN, a0 + A_SPAN, A_GRID)
 b_vals = np.linspace(B_LO, B_HI, B_GRID)
 best = None
 for a in a_vals:
 for b in b_vals:
 logF = build_logF(ts, spec_logs, lz, a, b)
 score, _, _ = score_delta2_mean(logF)
 if (best is None) or (score < best[0]):
 best = (score, a, b)
 return best

def report(ts, d2_mean, d2_all, logF):

 d2m = d2_mean - d2_mean.mean()
 spread = np.max(np.abs(d2_all - d2m), axis=0)


 mags = logF.real
 gm = mags.mean(axis=1, keepdims=True)
 mag_centered = np.exp(mags - gm)
 mag_mean = mag_centered.mean(axis=0)
 mag_lo = mag_centered.min(axis=0)
 mag_hi = mag_centered.max(axis=0)

 print("t\t|Δ2 log F| mean\tspread\t\t|F|/geom-mean (lo.. hi)")
 for t, m, s, mf, lo, hi in zip(ts[1:-1], np.abs(d2m), np.abs(spread), mag_mean, mag_lo, mag_hi):
 print(f"{t:+.2f}\t{m:.3e}\t{s:.3e}\t{mf:.3f} ({lo:.3f}.. {hi:.3f})")

if __name__ == "__main__":
 ts = np.linspace(T_MIN, T_MAX, T_POINTS)

 spec_logs, lz = precompute_logs(ts)

 score, a_fit, b_fit = fit_chi_params(ts, spec_logs, lz)
 print(f"[chi-fit] best score={score:.3e} a={a_fit:.5f} b={b_fit:.5f}")

 logF = build_logF(ts, spec_logs, lz, a_fit, b_fit)
 score, d2_mean, d2_all = score_delta2_mean(logF)
 report(ts, d2_mean, d2_all, logF)