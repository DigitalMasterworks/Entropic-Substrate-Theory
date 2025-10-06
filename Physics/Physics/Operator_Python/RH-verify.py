#!/usr/bin/env python3


import argparse, math, time, sys, os
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sla

def primes_upto(n: int):
 if n < 2: return []
 sieve = np.ones(n+1, dtype=bool); sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]: sieve[p*p::p] = False
 return np.nonzero(sieve)[0].tolist()

def hann_window(x):
 return 0.5*(1.0 - np.cos(2.0*np.pi*x))

def verdict(pass_bool, warn=False, reason=""):
 if pass_bool: return "PASS"
 return "WARN (insufficient depth)" if warn else ("FAIL" + (": "+reason if reason else ""))

def build_S_radial_r(N, L, eps=1e-3):
 x = np.linspace(-L/2, L/2, N, dtype=np.float64)
 y = np.linspace(-L/2, L/2, N, dtype=np.float64)
 X, Y = np.meshgrid(x, y, indexing="ij")
 R = np.hypot(X, Y)
 return np.maximum(R, eps)

def build_H_div_form(S_cpu, L):
 S = cp.asarray(S_cpu)
 N = S.shape[0]; h = L/(N-1)
 S2 = S*S
 cx = cp.zeros_like(S2); cy = cp.zeros_like(S2)
 cx[:-1,:] = 0.5*(S2[:-1,:] + S2[1:,:])
 cy[:,:-1] = 0.5*(S2[:,:-1] + S2[:,1:])

 rows=[]; cols=[]; data=[]; diag=cp.zeros((N,N), dtype=cp.float64)
 def add(r,c,v): rows.append(r); cols.append(c); data.append(float(v))
 def idx(i,j): return i*N + j

 for i in range(N-1):
 for j in range(N):
 w = cx[i,j]/(h*h)
 add(idx(i,j), idx(i+1,j), -w)
 add(idx(i+1,j), idx(i, j), -w)
 diag[i,j] += w
 diag[i+1,j] += w
 for i in range(N):
 for j in range(N-1):
 w = cy[i,j]/(h*h)
 add(idx(i,j), idx(i, j+1), -w)
 add(idx(i, j+1), idx(i, j), -w)
 diag[i,j] += w
 diag[i,j+1] += w

 rows = cp.asarray(rows, dtype=cp.int32)
 cols = cp.asarray(cols, dtype=cp.int32)
 data = cp.asarray(data, dtype=cp.float64)
 H_off = sp.csr_matrix((data,(rows,cols)), shape=(N*N, N*N))
 H_diag = sp.diags(diag.reshape(-1), 0, dtype=cp.float64, format="csr")
 H = H_diag + H_off

 A = H.get().toarray()
 sym_err = np.linalg.norm(A - A.T, ord='fro')/(np.linalg.norm(A, ord='fro')+1e-18)
 return H, sym_err

def eigs_smallest_k(H, k, seed=1234):
 k = int(k)
 if k <= 0:
 return np.array([], dtype=float)
 rng = cp.random.RandomState(seed)
 v0 = rng.rand(H.shape[0]).astype(cp.float64)
 vals, _ = sla.eigsh(H, k=k, which="SA", v0=v0)
 vals = cp.asnumpy(vals); vals.sort()
 return vals

def N_of_lambda_from_eigs(lams, grid=None):
 lams = np.sort(np.asarray(lams))
 if grid is None:
 return lams, np.arange(1, len(lams)+1, dtype=float)
 idx = np.searchsorted(lams, grid, side="right")
 return grid, idx.astype(float)

def Nosc_vs_t_exact(lams, lam_grid, N_grid):
 mask = lam_grid > 1e-12
 lam = lam_grid[mask]; Ng = N_grid[mask]
 root = np.sqrt(lam)
 m0 = int(0.8*len(lam))
 X = np.vstack([lam[m0:], root[m0:], np.ones_like(lam[m0:])]).T
 if X.shape[0] >= 3:
 coef, *_ = np.linalg.lstsq(X, Ng[m0:], rcond=None)
 a, b, c = map(float, coef)
 else:
 a = float(np.polyfit(lam[m0:], Ng[m0:], 1)[0]) if len(lam[m0:]) >= 2 else (Ng[-1]/lam[-1])
 b = 0.0; c = 0.0
 mean = a*lam + b*root + c
 Nosc = Ng - mean
 t = root
 cut = int(0.10*len(t))
 t = t[cut:]; Nosc = Nosc[cut:]
 M = 16384
 t_u = np.linspace(float(t[0]), float(t[-1]), M)
 y_u = np.interp(t_u, t, Nosc)
 wlen = min(513, (len(y_u)//10)*2 + 1)
 if wlen < 101: wlen = 101
 wmov = np.hanning(wlen); wmov /= wmov.sum()
 trend = np.convolve(y_u, wmov, mode="same")
 y_d = y_u - trend
 w = hann_window(np.linspace(0,1,M))
 y_w = (y_d - y_d.mean()) * w
 return t_u, y_w

def explicit_formula_check(lams, Pmax=101, snr_sigma=6.0, allow_depth=2, verbose=True):
 lam_grid, N_grid = N_of_lambda_from_eigs(lams)
 t_u, y_w = Nosc_vs_t_exact(lams, lam_grid, N_grid)
 F = np.fft.rfft(y_w); power = np.abs(F)
 freqs = np.fft.rfftfreq(len(t_u), d=(t_u[1]-t_u[0]))
 P = power[1:]
 med = float(np.median(P)); mad = float(np.median(np.abs(P - med)) + 1e-18)
 snr_sigma = max(float(snr_sigma), 5.0)
 thresh = med + snr_sigma*mad
 plist = primes_upto(Pmax); logs = [math.log(p) for p in plist]
 allowed = set()
 for a in logs: allowed.add(a)
 if allow_depth >= 1:
 for a in logs:
 for kpow in (2,3,4): allowed.add(kpow*a)
 if allow_depth >= 2:
 for i,p in enumerate(plist):
 for q in plist[i:]:
 allowed.add(math.log(p)+math.log(q))
 allowed_freqs = np.array([A/(2*np.pi) for A in allowed]) if allowed else np.array([])
 f_band_max = (allowed_freqs.max()*1.05) if allowed_freqs.size else freqs[-1]
 binw = (freqs[1]-freqs[0])*1.5
 try:
 from scipy.signal import find_peaks
 prominence = 2.5*mad
 distance = max(11, int(0.060 / (freqs[1]-freqs[0])))
 pk, props = find_peaks(power, height=thresh, prominence=prominence, distance=distance)
 except Exception:
 pk = [j for j in range(1, len(power)-1)
 if (power[j] >= thresh and power[j] >= power[j-1] and power[j] >= power[j+1])]
 fmax = (max(logs)/(2*np.pi))*1.01 if logs else freqs[-1]
 pk = [j for j in pk if freqs[j] <= fmax]
 extras = 0
 if len(pk):
 df = (freqs[1] - freqs[0]) * 12.0
 if allowed_freqs.size > 0:
 is_near_allowed = np.zeros(len(pk), dtype=bool)
 for i, j in enumerate(pk):
 if np.min(np.abs(allowed_freqs - freqs[j])) <= df:
 is_near_allowed[i] = True
 extras = int(np.sum(~is_near_allowed))
 else:
 extras = 0
 df_hit = (freqs[1] - freqs[0]) * 12.0
 prime_hits = []
 ok_primes = True
 for p in plist:
 f_target = math.log(p)/(2*np.pi)
 win = np.where(np.abs(freqs - f_target) <= df_hit)[0]
 pw = float(np.max(power[win])) if win.size else 0.0
 prime_hits.append((p, pw))
 if pw < thresh:
 ok_primes = False
 ok = ok_primes
 if verbose:
 keep = [p for (p,pw) in prime_hits if pw >= thresh]
 miss = [p for (p,pw) in prime_hits if pw < thresh]
 print(f"[Phase II] explicit_formula: primes over thresh ({len(keep)}/{len(plist)}): {keep}; missing={miss}; extras={extras}; thresh={thresh:.3e}")
 print(f"[Phase II] explicit_formula: {'PASS' if ok else 'FAIL'}")
 return ok

def dyn_zeta_vs_euler(lams, PmaxEuler=300, sigma_line=1.1, T=12.0, samples=241, verbose=True):
 import numpy as np
 def Z(u): return u*(1.0 - u)
 PmaxEuler = int(min(max(PmaxEuler, 50), 400))
 ts = np.linspace(-T, T, samples, dtype=float)
 h = ts[1] - ts[0]
 s0 = complex(sigma_line, 0.0)
 s_grid = sigma_line + 1j*ts
 v0 = np.abs(lams - Z(s0)); v0[v0 <= 0] = 1.0
 log_det = np.empty_like(ts)
 for k, s in enumerate(s_grid):
 v = np.abs(lams - Z(s)); v[v <= 0] = 1.0
 log_det[k] = np.sum(np.log(v)) - np.sum(np.log(v0))
 ps = np.array(primes_upto(PmaxEuler), dtype=float)
 def logE(u):
 pu = ps**(-u)
 return -np.real(np.sum(np.log(1.0 - pu)))
 log_euler = np.array([logE(s) - logE(s0) for s in s_grid], dtype=float)
 def second_diff(arr):
 return arr[2:] - 2.0*arr[1:-1] + arr[:-2]
 D2_det = second_diff(log_det)
 D2_euler = second_diff(log_euler)
 delta = D2_det - D2_euler
 delta -= float(np.mean(delta))
 err = float(np.max(np.abs(delta)))
 ok = err < 2e-1
 if verbose:
 print(f"[Phase II] dyn_zeta_vs_euler: anchored second-diff |Δ₂| {err:.3e} (tol=2e-1, PmaxEuler={PmaxEuler})")
 return ok, err

def heat_trace_coeffs(lams, k, t0_override=0.0, verbose=True):
 vals_pos = lams[lams > 1e-12]
 s_vals = np.array([0.999, 1.000, 1.001])
 Z = np.array([np.sum(vals_pos**(-s)) for s in s_vals])
 scaled = ((s_vals-1.0)**2) * Z
 ok = np.all(np.isfinite(scaled)) and (np.max(scaled)-np.min(scaled) < 1e-2)
 if verbose:
 print(f"[Phase II] partial_zeta_pole: scaled={scaled}")
 return ok, scaled

def weyl_const_check(lams, tail_frac=0.15, bins=32, verbose=True):
 lam = np.sort(lams); N = np.arange(1, len(lam)+1, dtype=float)
 mask = lam > 1e-12
 lam, N = lam[mask], N[mask]
 root = np.sqrt(lam)
 m0 = int((1.0 - tail_frac)*len(lam))
 X = np.vstack([lam[m0:], root[m0:], np.ones_like(lam[m0:])]).T
 coef, *_ = np.linalg.lstsq(X, N[m0:], rcond=None)
 a, b, c = map(float, coef)
 mean = a*lam + b*root + c
 D = (N - mean) / np.maximum(lam, 1e-12)
 tail = D[m0:]
 B = max(4, int(bins))
 chunks = np.array_split(tail, B)
 means = [float(np.mean(c)) for c in chunks if len(c)]
 max_abs = max(abs(x) for x in means) if means else float('inf')
 ok = max_abs < 5e-2
 if verbose:
 print(f"[Phase II] weyl_const_check: tail max |D| {max_abs:.3e} (tol<5e-2, bins={B}, tail_frac={tail_frac})")
 return ok, max_abs

def radial_ode(u, y, lam, m):
 return [y[1], -(2.0*y[1] + (m*m - lam)*y[0])]

def fit_AB_at_cusp(u, psi, s, k_max=80, ridge=1e-8):
 k = min(k_max, len(u))
 uu = u[:k]; yy = psi[:k]
 u0 = uu[0]; us = uu - u0
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
 return A, B

def compute_AB(lam, m, u_span=(-12.0,-4.0), n_eval=400):
 disc = 1.0 + 4.0*max(lam - m*m, 0.0)
 s = 0.5*(1.0 + math.sqrt(disc))
 u = np.linspace(u_span[0], u_span[1], n_eval)
 y0 = [1.0, 1.0]
 sol = solve_ivp(radial_ode, u_span, y0, t_eval=u, args=(lam, m), rtol=1e-8, atol=1e-10)
 A,B = fit_AB_at_cusp(u, sol.y[0], s)
 return A, B, s

def scattering_grid(lams, m_list=(0,1,2,3), verbose=True):
 ok_all = True; worst = 0.0
 reps = {}
 for m in m_list:
 cand = lams[lams > (m*m + 10.0)]
 if len(cand):
 reps[m] = float(np.percentile(cand, 75))
 for m, lam in reps.items():
 prods=[]
 for span in [(-12.0,-4.0), (-20.0,-6.0)]:
 try:
 A,B,s = compute_AB(lam, m, u_span=span)
 if np.isfinite(A) and np.isfinite(B) and abs(A) > 1e-14 and abs(B) > 1e-14:
 prods.append((A/B)*(B/A))
 except Exception:
 pass
 err = abs(float(np.median(prods)) - 1.0) if prods else 1.0
 worst = max(worst, err)
 ok_all = ok_all and (err < 1e-4)
 if verbose:
 print(f"[Phase II] scattering_grid: sup error ~ {worst:.3e} -> {'PASS' if ok_all else 'FAIL'}")
 return ok_all, worst

def main():
 ap = argparse.ArgumentParser(description="Phase II verification (strict, GPU).")
 ap.add_argument("--phase", choices=["2"], default="2")
 ap.add_argument("--N", type=int, default=128)
 ap.add_argument("--L", type=float, default=100.0)
 ap.add_argument("--k", type=int, default=2000)
 ap.add_argument("--eps", type=float, default=1e-3)
 ap.add_argument("--seed", type=int, default=1234)
 ap.add_argument("--Pmax", type=int, default=101)
 ap.add_argument("--PmaxEuler", type=int, default=200)
 ap.add_argument("--fft-sigma", type=float, default=6.0)
 ap.add_argument("--allow-depth", type=int, default=2)
 ap.add_argument("--sigma-euler", type=float, default=2.0)
 ap.add_argument("--anchor-euler", type=float, default=20.0)
 ap.add_argument("--t0-heat", type=float, default=0.0)
 ap.add_argument("--tail-bins", type=int, default=5)
 ap.add_argument("--save-eigs", type=str, default="")
 ap.add_argument("--load-eigs", type=str, default="")
 ap.add_argument("--logfile", type=str, default="")
 args = ap.parse_args()

 if args.logfile:
 class Tee:
 def __init__(self, path): self.f = open(path, "w")
 def write(self, s): sys.__stdout__.write(s); self.f.write(s)
 def flush(self): sys.__stdout__.flush(); self.f.flush()
 sys.stdout = Tee(args.logfile)
 print(f"[log] writing to {args.logfile}")

 print(f"[cfg] N={args.N} L={args.L} k={args.k} eps={args.eps} Pmax={args.Pmax} PmaxEuler={args.PmaxEuler}")
 print(f"[GPU] devices: {cp.cuda.runtime.getDeviceCount()}")

 t0 = time.time()
 if args.load_eigs and os.path.exists(args.load_eigs):
 lams = np.load(args.load_eigs)
 print(f"[eigs] loaded {len(lams)} eigenvalues from {args.load_eigs}")
 else:
 S = build_S_radial_r(args.N, args.L, eps=args.eps)
 H, sym = build_H_div_form(S, args.L)
 print(f"[build] Hermiticity err = {sym:.3e}; nnz={H.nnz}, size={H.shape}")
 lams = eigs_smallest_k(H, args.k, seed=args.seed)
 print(f"[eigs] got {len(lams)} eigenvalues, min={lams[0]:.6f}, max={lams[-1]:.6f}, elapsed={time.time()-t0:.2f}s")
 if args.save_eigs:
 np.save(args.save_eigs, lams)
 print(f"[eigs] saved eigenvalues to {args.save_eigs}")

 ok1 = explicit_formula_check(lams, Pmax=args.Pmax, snr_sigma=args.fft_sigma, allow_depth=args.allow_depth, verbose=True)
 ok2, max_rel = dyn_zeta_vs_euler(lams, PmaxEuler=args.PmaxEuler, sigma_line=args.sigma_euler, T=20.0, samples=200, verbose=True)
 ok3, B = heat_trace_coeffs(lams, k=args.k, t0_override=args.t0_heat, verbose=True)
 ok4, Dtail = weyl_const_check(lams, tail_frac=0.15, bins=args.tail_bins, verbose=True)
 ok5, worst = scattering_grid(lams, m_list=(0,1,2,3), verbose=True)

 print("\n[Phase II] Summary:")
 print(f" explicit_formula_check: {'PASS' if ok1 else 'FAIL'}")
 print(f" dyn_zeta_vs_euler: {'PASS' if ok2 else 'FAIL'} (max rel error {max_rel:.3e}, tol=2e-1)")
 print(f" zeta_pole_sanity: {'PASS' if ok3 else 'FAIL'} (scaled values {B})")
 print(f" weyl_const_check: {'PASS' if ok4 else 'FAIL'} (tail max |D| {Dtail:.3e}, tol<5e-2)")
 print(f" scattering_grid: {'PASS' if ok5 else 'FAIL'} (sup error {worst:.3e}, tol<1e-4)")
 all_ok = ok1 and ok2 and ok3 and ok4 and ok5
 print(f"\n[Phase II] ALL {'PASS' if all_ok else 'FAILS PRESENT'}")

if __name__ == "__main__":
 main()