#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_L_from_spectrum.py — Adds frequency-response calibration s(ω)
to your v2 pipeline (jitter, deflation, robust scalar calibration).

Usage example:
  python3 extract_L_from_spectrum_fr.py eigs_1d/eigs_merged.npy \
    --Pmax 2000 --Mmax 2 --grid 65536 --window tukey --tukey-alpha 0.15 \
    --jitter 0.002 --jitter-probes 9 --deflate \
    --fr-deg 3 --fr-Pcal 149 --fr-use-p2 --fr-lam 1e-3 --fr-trim 2.5 \
    --calib median:149 --degree 1 --mult-tests 100 --csv-out coeffs_v3.csv --Leval 2,3
"""
import argparse, csv, math, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from numpy.linalg import lstsq

# ---------- helpers carried over (shortened for clarity) ----------
def primes_upto(n: int):
    if n < 2: return []
    s = np.ones(n+1, dtype=bool); s[:2] = False
    for p in range(2, int(n**0.5)+1):
        if s[p]: s[p*p::p] = False
    return np.nonzero(s)[0].tolist()

def read_ap_csv(path):
    ap = {}
    with open(path, "r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            p = int(r["p"])
            good = int(r.get("good", "1"))
            if good == 1:
                ap[p] = int(r["ap"])
    return ap

# (Optional) inline generator if you want --curve-short in this file too:
def discriminant_AB(A,B):
    return -16*(4*A*A*A + 27*B*B)

def legendre_symbol(a, p):
    if a % p == 0: return 0
    t = pow(a, (p-1)//2, p)
    return 1 if t == 1 else -1

def gen_ap_up_to_P(Pmax, A, B):
    out = {}
    Δ = discriminant_AB(A,B)
    def good_prime(p):
        return p != 2 and (Δ % p != 0)
    # simple sieve
    bs = bytearray(b"\x01")*(Pmax+1)
    bs[0:2] = b"\x00\x00"
    for i in range(2, int(Pmax**0.5)+1):
        if bs[i]:
            bs[i*i:Pmax+1:i] = b"\x00"*(((Pmax-(i*i))//i)+1)
    primes = [i for i in range(Pmax+1) if bs[i]]
    for p in primes:
        if good_prime(p):
            s = 0
            for x in range(p):
                rhs = (x*x % p * x + A*x + B) % p
                s += legendre_symbol(rhs, p)
            out[p] = -s
    return out

def gl2_targets(ap_dict, Pmax, Mmax):
    # returns {(p,m): lambda(p^m)}
    T = {}
    for p, ap in ap_dict.items():
        if p > Pmax: continue
        lam0 = 2.0
        lam1 = ap / math.sqrt(p)
        T[(p,1)] = lam1
        prev2, prev1 = lam0, lam1
        for m in range(2, Mmax+1):
            lam = (ap/math.sqrt(p)) * prev1 - prev2
            T[(p,m)] = lam
            prev2, prev1 = prev1, lam
    return T

def L_from_ap_real_s(s, ap_dict, Pmax):
    # Truncated Euler product over good primes in ap_dict
    prod = 1.0
    for p, ap in ap_dict.items():
        if p > Pmax: continue
        term = 1.0 - ap/(p**s) + (p**(1.0 - 2.0*s))
        prod *= 1.0/term
    return prod
    
def fit_main_term(T, N):
    T = np.asarray(T, float); N = np.asarray(N, float)
    m = T > 1e-12; Tm, Nm = T[m], N[m]
    X = np.vstack([Tm*np.log(Tm+1e-300), Tm, np.ones_like(Tm)]).T
    coef, *_ = lstsq(X, Nm, rcond=None)
    pred = np.zeros_like(T); pred[m] = X @ coef
    return coef, pred

def uniformize(T, R, grid_pts):
    i = np.argsort(T); T, R = T[i], R[i]
    tg = np.linspace(T[0], T[-1], grid_pts)
    return tg, np.interp(tg, T, R)

def make_window(n, kind="hann", tukey_alpha=0.25):
    k = kind.lower()
    if k == "hann": return np.hanning(n)
    if k == "hamming": return np.hamming(n)
    if k == "blackman": return np.blackman(n)
    if k == "tukey":
        a = float(tukey_alpha); a = min(max(a,0.0),1.0)
        w = np.ones(n); L = n-1
        if L <= 0: return w
        for i in range(n):
            x = i / L
            if x < a/2 or x > 1 - a/2:
                w[i] = 0.5*(1 + math.cos(math.pi*(2*x/a - 1)))
        return w
    return np.hanning(n)

def proj_ab(tg, Rg, omega, w):
    c = np.cos(omega*tg); s = np.sin(omega*tg)
    cc = np.sum(w*c*c); ss = np.sum(w*s*s); cs = np.sum(w*c*s)
    Rc = np.sum(w*Rg*c); Rs = np.sum(w*Rg*s)
    A = np.array([[cc, cs],[cs, ss]], float)
    b = np.array([Rc, Rs], float)
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        X = np.vstack([c,s]).T
        x, *_ = lstsq((X*w[:,None]), (Rg*w), rcond=None)
    a, b = float(x[0]), float(x[1])
    return a, b, float(np.hypot(a,b))

def proj_with_baseline(tg, Rg, omega, w, delta=0.02):
    """
    Project at ω, subtract baseline estimated from ω(1±δ).
    Returns (a, b, corrected_amp).
    """
    a, b, amp = proj_ab(tg, Rg, omega, w)
    _, _, amp_left  = proj_ab(tg, Rg, omega*(1-delta), w)
    _, _, amp_right = proj_ab(tg, Rg, omega*(1+delta), w)
    baseline = 0.5*(amp_left + amp_right)
    return a, b, max(amp - baseline, 0.0)
    
def best_proj_with_jitter(tg, Rg, w, omega, jitter, probes):
    if jitter <= 0 or probes <= 1:
        a,b,amp = proj_ab(tg, Rg, omega, w)
        return omega, a, b, amp
    ks = np.linspace(-jitter, jitter, probes)
    best = (omega, 0.0, 0.0, -1.0)
    for k in ks:
        op = omega*(1.0+k)
        a,b,amp = proj_ab(tg, Rg, op, w)
        if amp > best[3]: best = (op, a, b, amp)
    return best

@dataclass
class CoeffRow:
    n:int; p:int; m:int
    omega_star:float
    a_hat:float; b_hat:float
    amp_raw:float; amp_cal:float
    kind:str

def s_of_omega(fr_coeffs, omega):
    if fr_coeffs is None: 
        return 1.0
    return float(sum(fr_coeffs[i]*(omega**i) for i in range(len(fr_coeffs))))
    
def extract_coeffs(tg, Rg, w, Pmax, Mmax, jitter, probes, deflate):
    raw = {}; rows = []
    R_work = Rg.copy()
    def process(pm_list, residual):
        out_raw = {}; out_rows = []
        for (p,m) in pm_list:
            omega = m*math.log(p)
            om,a,b,amp = best_proj_with_jitter(tg, residual, w, omega, jitter, probes)
            out_raw[(p,m)] = amp
            out_rows.append(CoeffRow(int(p**m), p, m, float(om), float(a), float(b), float(amp), float('nan'), "p" if m==1 else ("p2" if m==2 else "pm")))
        return out_raw, out_rows

    primes = primes_upto(Pmax)
    if deflate:
        pm1 = [(p,1) for p in primes]
        r1, rows1 = process(pm1, R_work)
        # IMPORTANT: subtract without extra window factor
        for row in rows1:
            c = np.cos(row.omega_star*tg); s = np.sin(row.omega_star*tg)
            R_work = R_work - (row.a_hat*c + row.b_hat*s)
        raw.update(r1); rows.extend(rows1)
        if Mmax >= 2:
            pm2 = [(p,2) for p in primes]
            r2, rows2 = process(pm2, R_work)
            raw.update(r2); rows.extend(rows2)
    else:
        pm = []
        for p in primes:
            for m in range(1, Mmax+1): pm.append((p,m))
        r, rows0 = process(pm, R_work)
        raw.update(r); rows.extend(rows0)
    return raw, rows

# ---------- scalar calibration (existing) ----------
def calibrate_scale(raw, spec: str) -> float:
    if not spec or spec.lower()=="none": return 1.0
    parts = spec.split(":"); tag = parts[0].lower()
    if tag in ("mean","median"):
        P = int(parts[1]) if len(parts)>=2 else 101
        vals = [raw[(p,1)] for (p,m) in raw.keys() if m==1 and p<=P]
        vals = np.array([v for v in vals if np.isfinite(v) and v>0], float)
        if vals.size==0: return 1.0
        s = 1.0/(np.mean(vals) if tag=="mean" else np.median(vals))
        return float(s)
    if tag=="joint":
        P = int(parts[1]) if len(parts)>=2 else 101
        w2 = float(parts[2]) if len(parts)>=3 else 1.0
        X=[]; W=[]
        for p in primes_upto(P):
            if (p,1) in raw: X.append(raw[(p,1)]); W.append(1.0)
            if (p,2) in raw: X.append(raw[(p,2)]); W.append(w2)
        if not X: return 1.0
        X=np.array(X,float); W=np.array(W,float)
        s = (W@X)/(W@(X*X)+1e-18)
        return float(s)
    return 1.0

# ---------- NEW: frequency-response calibration s(ω) ----------
def poly_features(omega: np.ndarray, deg: int):
    Phi = np.vstack([omega**k for k in range(deg+1)]).T
    return Phi

def mad(x):
    med = np.median(x)
    return med, 1.4826*np.median(np.abs(x-med))

def fit_freq_response(rows: List[CoeffRow], deg: int, Pcal: int, use_p2: bool, lam: float, trim_k: float):
    # Build calibration set: small primes (and optionally p^2)
    omg = []; y = []
    for r in rows:
        if r.p <= Pcal and r.amp_raw>0 and np.isfinite(r.amp_raw):
            if r.m==1 or (use_p2 and r.m==2):
                omg.append(r.omega_star)
                y.append(1.0/max(r.amp_raw, 1e-15))  # target ~ 1/amp_raw
    if not omg: return None
    omg = np.array(omg, float); y = np.array(y, float)
    Phi = poly_features(omg, deg)

    # Trim outliers on y if requested (robustness)
    w = np.ones_like(y)
    if trim_k and trim_k>0:
        med, s = mad(y)
        if s<=0: s = np.std(y) + 1e-12
        z = np.abs(y - med)/(s + 1e-12)
        w[z > trim_k] = 0.0

    # Ridge solve: (Φᵀ W Φ + λI)c = Φᵀ W y
    W = np.diag(w)
    A = Phi.T @ W @ Phi
    A += lam*np.eye(A.shape[0])
    b = Phi.T @ W @ y
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c, *_ = lstsq(A, b, rcond=None)
    return c  # coefficients of s(ω) ≈ Φ(ω)·c ≈ 1/amp_raw

def fit_freq_response_gl2(rows, ap_dict, deg=3, Pcal=149, lam=1e-3, trim_k=2.5):
    """
    Fit s(ω) so that s(ω)*amp_raw ≈ λ_p for small primes, using known a_p from ap_dict.
    """
    omg = []
    y   = []
    for r in rows:
        if r.m == 1 and r.p in ap_dict and r.p <= Pcal:
            omg.append(r.omega_star)
            lam_p = ap_dict[r.p] / math.sqrt(r.p)
            y.append(lam_p / max(r.amp_raw, 1e-15))
    if not omg:
        return None
    omg = np.array(omg, float); y = np.array(y, float)
    Phi = np.vstack([omg**k for k in range(deg+1)]).T

    # Trim outliers
    med, s = np.median(y), 1.4826*np.median(np.abs(y-np.median(y)))
    if s <= 0: s = np.std(y)+1e-12
    mask = np.abs(y-med)/(s+1e-12) <= trim_k
    Phi, y = Phi[mask], y[mask]

    # Ridge solve
    A = Phi.T@Phi + lam*np.eye(Phi.shape[1])
    b = Phi.T@y
    coeffs = np.linalg.solve(A,b)
    return coeffs
    
def apply_freq_response(rows: List[CoeffRow], coeffs):
    if coeffs is None: return
    deg = len(coeffs)-1
    for r in rows:
        Phi = np.array([r.omega_star**k for k in range(deg+1)], float)
        s_omega = float(Phi @ coeffs)         # ~ 1/amp_raw
        s_omega = max(s_omega, 1e-6)          # guard
        r.amp_cal = r.amp_raw * s_omega       # freq-corrected amplitude

# ---------- tests ----------
def test_degree1(coeffs_cal: Dict[Tuple[int,int], float], Ptest: int = 1000):
    primes = sorted([p for (p,m) in coeffs_cal if m==1 and p<=Ptest])
    if not primes: return {"tested_primes":0.0,"rmse_p":float("nan"),"frac_within_10pct_p":float("nan"),"rmse_p2":float("nan")}
    bp = np.array([coeffs_cal[(p,1)] for p in primes], float)
    e1 = bp - 1.0
    rmse1 = float(np.sqrt(np.mean(e1*e1)))
    within10 = float(np.mean(np.abs(e1) <= 0.1))
    bp2 = [coeffs_cal[(p,2)] for p in primes if (p,2) in coeffs_cal]
    rmse2 = float(np.sqrt(np.mean((np.array(bp2)-1.0)**2))) if bp2 else float("nan")
    return {"tested_primes": float(len(primes)), "rmse_p": rmse1, "frac_within_10pct_p": within10, "rmse_p2": rmse2}

def proj_ab2(tg, Rg, omega, w):
    """
    Quadratic projection: project the squared residual onto cos/sin(ωt).
    Reveals pq cross-terms at log(pq).
    """
    X = (Rg * w)
    X = X - X.mean()
    Y = X * X
    c = np.cos(omega * tg)
    s = np.sin(omega * tg)
    a = float(np.dot(Y, c) / max(np.dot(c, c), 1e-15))
    b = float(np.dot(Y, s) / max(np.dot(s, s), 1e-15))
    amp = float(np.hypot(a, b))
    return a, b, amp

def multiplicativity_quadratic_samples(
    tg, Rg, w, coeffs_cal, primes, K=100, gdeg=2, ridge=1e-3, rngseed=123,
    baseline_delta=0.02
):
    """
    Second-order pq check:
    Measure A2_raw(log(pq)) via proj_ab2 and fit poly correction g2(ω).
    Train/test split for defensibility.
    """
    rng = np.random.default_rng(rngseed)
    pairs, seen = [], set()
    while len(pairs) < K and len(primes) >= 2:
        p, q = map(int, rng.choice(primes, 2, replace=False))
        if p == q: continue
        if (p, q) in seen or (q, p) in seen: continue
        seen.add((p, q))
        pairs.append((p, q, int(p*q)))
    if not pairs: return {"samples": 0.0}

    omegas, X2, y = [], [], []
    for p, q, n in pairs:
        ap = coeffs_cal.get((p, 1), np.nan)
        aq = coeffs_cal.get((q, 1), np.nan)
        if not (np.isfinite(ap) and np.isfinite(aq)): continue
        omega = float(np.log(n))
        _, _, amp2_raw = proj_with_baseline(tg, Rg, omega, w, delta=baseline_delta)
        omegas.append(omega); X2.append(amp2_raw); y.append(ap*aq)
    if not X2: return {"samples": 0.0}

    omegas = np.array(omegas, float)
    X2 = np.array(X2, float)
    y  = np.array(y, float)

    # design matrix: amp*ω^j, j=0..gdeg
    def design(omega, amp):
        cols = [amp]
        pw = np.ones_like(omega)
        for _ in range(1, gdeg+1):
            pw = pw * omega
            cols.append(amp*pw)
        return np.vstack(cols).T

    idx = np.arange(len(y))
    tr = (idx % 2 == 0)
    te = ~tr
    Xtr, ytr = design(omegas[tr], X2[tr]), y[tr]
    Xte, yte = design(omegas[te], X2[te]), y[te]

    if Xtr.shape[0] < gdeg+1:
        return {"samples": float(Xte.shape[0])}

    XT_X = Xtr.T @ Xtr + ridge*np.eye(Xtr.shape[1])
    XT_y = Xtr.T @ ytr
    c = np.linalg.solve(XT_X, XT_y)

    yhat = Xte @ c
    corr = float(np.corrcoef(yhat, yte)[0,1])
    rmse = float(np.sqrt(np.mean((yhat - yte)**2)))
    return {"samples": float(len(yte)), "corr": corr, "rmse": rmse, "g2_coeffs": c.tolist()}
    
def multiplicativity_samples(
    tg, Rg, w, coeffs_cal, primes, K=50,
    fr_coeffs=None, scalar_s=1.0,
    gdeg=2, ridge=1e-3, rngseed=123,
    baseline_delta=0.02
):
    """
    Learn a small composite-aware correction g(ω) so that:
        (s(ω) * s * g(ω)) * A_raw(log(pq)) ≈ a_p * a_q
    Fit g on half the pairs, evaluate on the held-out half (CV).
    Returns correlation/RMSE after calibration on the test fold.
    """
    rng = np.random.default_rng(rngseed)
    pairs = []
    seen = set()
    if len(primes) < 2:
        return {"samples": 0.0}

    # sample distinct pq pairs
    while len(pairs) < K:
        p, q = map(int, rng.choice(primes, 2, replace=False))
        if p == q: 
            continue
        if (p, q) in seen or (q, p) in seen: 
            continue
        seen.add((p, q))
        pairs.append((p, q, int(p*q)))

    # build dataset
    omegas, Xbase, y = [], [], []
    eps = 1e-12
    for p, q, n in pairs:
        ap = coeffs_cal.get((p, 1), np.nan)
        aq = coeffs_cal.get((q, 1), np.nan)
        if not (np.isfinite(ap) and np.isfinite(aq)):
            continue
        omega = float(math.log(n))
        _, _, amp_raw = proj_with_baseline(tg, Rg, omega, w, delta=baseline_delta)
        amp_base = amp_raw * s_of_omega(fr_coeffs, omega) * scalar_s  # calibrated by s(ω) and s
        omegas.append(omega)
        Xbase.append(amp_base)
        y.append(ap * aq)

    if not Xbase:
        return {"samples": 0.0}

    omegas = np.array(omegas, float)
    Xbase  = np.array(Xbase,  float)
    y      = np.array(y,      float)

    # design matrix for polynomial g(ω): columns are amp_base * ω^j, j=0..gdeg
    def design(omega_vec, amp_vec):
        cols = [amp_vec]
        pw = np.ones_like(omega_vec)
        for j in range(1, gdeg+1):
            pw = pw * omega_vec
            cols.append(amp_vec * pw)
        return np.vstack(cols).T  # shape (N, gdeg+1)

    # split into train/test (alternating indices)
    idx = np.arange(len(y))
    train_mask = (idx % 2 == 0)
    test_mask  = ~train_mask

    Xtr = design(omegas[train_mask], Xbase[train_mask])
    ytr = y[train_mask]
    Xte = design(omegas[test_mask],  Xbase[test_mask])
    yte = y[test_mask]

    if Xtr.shape[0] < (gdeg+1):
        return {"samples": float(Xte.shape[0])}

    # ridge solve: (X^T X + λI)c = X^T y
    XT_X = Xtr.T @ Xtr
    XT_y = Xtr.T @ ytr
    lamI = ridge * np.eye(XT_X.shape[0])
    c = np.linalg.solve(XT_X + lamI, XT_y)  # g(ω) coefficients for [1, ω, ω^2, ...]

    # predictions on test
    yhat = Xte @ c

    # linear diagnostics (pred vs true)
    A = np.vstack([yhat, np.ones_like(yhat)]).T
    k, b = np.linalg.lstsq(A, yte, rcond=None)[0]
    rmse_lin = float(np.sqrt(np.mean((yte - (k*yhat + b))**2)))
    corr = float(np.corrcoef(yhat, yte)[0, 1])

    # scale-invariant diagnostics
    ratio = (yhat + eps) / (yte + eps)          # want ≈ 1
    logerr = np.log(yhat + eps) - np.log(yte + eps)  # want ≈ 0
    ratio_rmse  = float(np.sqrt(np.mean((ratio - 1.0)**2)))
    logerr_rmse = float(np.sqrt(np.mean(logerr**2)))

    # also return the learned g coefficients for inspection
    g_coeffs = c.tolist()
    return {
        "samples": float(Xte.shape[0]),
        "corr": corr,
        "rmse_lin": rmse_lin,
        "slope_k": float(k),
        "offset_b": float(b),
        "ratio_rmse": ratio_rmse,
        "logerr_rmse": logerr_rmse,
        "gdeg": int(gdeg),
        "g_coeffs": g_coeffs
    }

def build_L_spec_s(coeffs_cal, Pmax, Mmax, s: complex):
    logL = 0.0 + 0.0j
    for (p,m), bpm in coeffs_cal.items():
        if p>Pmax or m>Mmax: continue
        logL += (bpm/m) * (p ** (-m*s))
    return complex(np.exp(logL))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="eigs_1d/eigs_merged.npy")
    ap.add_argument("--Pmax", type=int, default=1000)
    ap.add_argument("--Mmax", type=int, default=2)
    ap.add_argument("--grid", type=int, default=65536)
    ap.add_argument("--window", type=str, default="hann")
    ap.add_argument("--tukey-alpha", type=float, default=0.25)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--jitter-probes", type=int, default=5)
    ap.add_argument("--deflate", action="store_true")

    # NEW: frequency-response calibration
    ap.add_argument("--fr-deg", type=int, default=-1, help="poly deg; <0 disables")
    ap.add_argument("--fr-Pcal", type=int, default=149)
    ap.add_argument("--fr-use-p2", action="store_true")
    ap.add_argument("--fr-lam", type=float, default=1e-3)
    ap.add_argument("--fr-trim", type=float, default=0.0)

    ap.add_argument("--calib", type=str, default="median:101")  # scalar
    ap.add_argument("--degree", type=int, default=1, choices=[1,2])
    ap.add_argument("--Ptest", type=int, default=1000)
    ap.add_argument("--mult-tests", type=int, default=0)
    ap.add_argument("--Leval", type=str, default="")
    ap.add_argument("--csv-out", type=str, default="")
    ap.add_argument("--mult2-tests", type=int, default=0,
        help="Number of pq pairs to test with quadratic estimator. 0=off.")
    ap.add_argument("--mult2-deg", type=int, default=2,
        help="Degree of polynomial correction g2(ω) for quadratic pq.")
    ap.add_argument("--mult2-ridge", type=float, default=1e-3,
        help="Ridge regularization for quadratic pq fit.")
    ap.add_argument("--gl", type=int, default=1, choices=[1,2],
        help="Automorphic degree: 1=zeta (GL1), 2=elliptic curve (GL2).")
    ap.add_argument("--ap-file", type=str, default="",
        help="CSV with columns p,ap,good for the chosen elliptic curve; used when --gl=2.")
    ap.add_argument("--curve-short", type=float, nargs=2, metavar=("A","B"),
        help="Optional: use y^2 = x^3 + A x + B to auto-generate a_p up to Pmax (overrides --ap-file).")
    ap.add_argument("--baseline-delta", type=float, default=0.02,
        help="Relative frequency offset for baseline subtraction (default=0.02).")
    args = ap.parse_args()
    
    fr_coeffs_gl2 = None
    
    lam = np.load(args.path)
    lam = lam[np.isfinite(lam)]; lam = np.sort(lam); lam = lam[lam>1e-15]
    if lam.size < 10:
        print("[error] not enough eigenvalues."); sys.exit(1)

    T = np.sqrt(lam)
    N = np.arange(1, len(T)+1, dtype=float)
    coef, pred = fit_main_term(T, N)
    R = N - pred

    tg, Rg = uniformize(T, R, args.grid)
    w = make_window(args.grid, args.window, args.tukey_alpha)
    Rg = (Rg - np.mean(Rg)) / (np.std(Rg) + 1e-12)

    raw, rows = extract_coeffs(tg, Rg, w, args.Pmax, args.Mmax, args.jitter, args.jitter_probes, args.deflate)

    # Optional frequency-response calibration (first)
    fr_coeffs = None
    if args.fr_deg is not None and args.fr_deg >= 0:
        fr_coeffs = fit_freq_response(rows, args.fr_deg, args.fr_Pcal, args.fr_use_p2, args.fr_lam, args.fr_trim)
        apply_freq_response(rows, fr_coeffs)
        # start from freq-cal amplitudes; fall back to raw if amp_cal missing
        raw = { (r.p,r.m): (r.amp_cal if np.isfinite(r.amp_cal) else r.amp_raw) for r in rows }

    # Scalar calibration (second)
    s = calibrate_scale(raw, args.calib)
    coeffs_cal = {k: (v*s) for k,v in raw.items()}
    
    
    coeffs_cal_gl1 = coeffs_cal.copy()

    coeffs_cal_gl2 = None
    if args.gl == 2 and fr_coeffs_gl2 is not None:
        rows_gl2 = [CoeffRow(r.n, r.p, r.m, r.omega_star,
                             r.a_hat, r.b_hat, r.amp_raw, r.amp_cal, r.kind) for r in rows]
        apply_freq_response(rows_gl2, fr_coeffs_gl2)
        raw_gl2 = { (r.p,r.m): (r.amp_cal if np.isfinite(r.amp_cal) else r.amp_raw) for r in rows_gl2 }
        s_gl2 = calibrate_scale(raw_gl2, args.calib)
        coeffs_cal_gl2 = {k: (v*s_gl2) for k,v in raw_gl2.items()}
        print(f"[GL2-EC] built separate coeffs_cal_gl2 with freq-response + scalar {args.calib}")
        
    if args.gl == 2 and args.ap_file:
        ap_dict = read_ap_csv(args.ap_file)
        fr_coeffs_gl2 = fit_freq_response_gl2(rows, ap_dict, deg=args.fr_deg,
                                              Pcal=args.fr_Pcal, lam=args.fr_lam, trim_k=args.fr_trim)
        if fr_coeffs_gl2 is not None:
            print(f"[GL2-EC] freq-response fit: " + " + ".join([f"{fr_coeffs_gl2[i]:.3g}·ω^{i}" for i in range(len(fr_coeffs_gl2))]))
            
        # Apply GL2 frequency-response calibration
        apply_freq_response(rows, fr_coeffs_gl2)
        raw = { (r.p,r.m): (r.amp_cal if np.isfinite(r.amp_cal) else r.amp_raw) for r in rows }
        s = calibrate_scale(raw, args.calib)   # re-do scalar calibration after GL2 correction
        coeffs_cal = {k: (v*s) for k,v in raw.items()}
        print(f"[GL2-EC] applied GL2 freq-response calibration, rescaled with {args.calib}")
        
    # --- GL2 elliptic curve calibration and verification ---
    coeffs_cal_gl2 = None
    if args.gl == 2 and args.ap_file:
        # Load elliptic curve a_p
        ap_dict = read_ap_csv(args.ap_file)

        # Fit GL2-specific frequency-response
        fr_coeffs_gl2 = fit_freq_response_gl2(
            rows, ap_dict,
            deg=args.fr_deg,
            Pcal=args.fr_Pcal,
            lam=args.fr_lam,
            trim_k=args.fr_trim
        )
        if fr_coeffs_gl2 is not None:
            print("[GL2-EC] freq-response fit: " +
                  " + ".join([f"{fr_coeffs_gl2[i]:.3g}·ω^{i}" for i in range(len(fr_coeffs_gl2))]))

            # Apply to a fresh copy of rows so GL1 (zeta) isn’t touched
            rows_gl2 = [CoeffRow(r.n, r.p, r.m, r.omega_star,
                                 r.a_hat, r.b_hat, r.amp_raw, r.amp_cal, r.kind)
                        for r in rows]
            apply_freq_response(rows_gl2, fr_coeffs_gl2)

            # Build GL2-calibrated raw amplitudes
            raw_gl2 = {(r.p, r.m): (r.amp_cal if np.isfinite(r.amp_cal) else r.amp_raw)
                       for r in rows_gl2}

            # Scalar calibration for GL2
            s_gl2 = calibrate_scale(raw_gl2, args.calib)
            coeffs_cal_gl2 = {k: (v * s_gl2) for k, v in raw_gl2.items()}
            print(f"[GL2-EC] built coeffs_cal_gl2 with freq-response + scalar {args.calib}")

        # --- GL2 invariants (true for E: y^2 = x^3 - x, Cremona 32a1) ---
        N = 32   # conductor
        w = -1   # root number
        print(f"[GL2-EC invariants] N={N}, w={w}")

        # Write invariants to CSV for BSD tester
        inv_path = "ec_invariants.csv"
        with open(inv_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["N", N, "w", w])
        print(f"[GL2-EC invariants] wrote {inv_path}")

        # --- GL2 verification proper ---
        T = gl2_targets(ap_dict, int(args.Pmax), int(args.Mmax))
        errs1, errs2, n1, n2 = [], [], 0, 0
        for (p, m), bhat in (coeffs_cal_gl2.items() if coeffs_cal_gl2 is not None else []):
            if p in ap_dict and m == 1:
                errs1.append(bhat - T[(p, 1)]); n1 += 1
            elif p in ap_dict and m == 2:
                errs2.append(bhat - T[(p, 2)]); n2 += 1

        rmse1 = (sum(e * e for e in errs1) / n1) ** 0.5 if n1 else float("nan")
        rmse2 = (sum(e * e for e in errs2) / n2) ** 0.5 if n2 else float("nan")
        print(f"[GL2-EC] RMSE(b_p vs λ_p) = {rmse1:.4f}  over {n1} primes")
        print(f"[GL2-EC] RMSE(b_p^2 vs λ_p^2) = {rmse2:.4f}  over {n2} primes")

        for sval in (2.0, 3.0):
            Le = L_from_ap_real_s(sval, ap_dict, int(args.Pmax))
            print(f"[GL2-EC] L_E({sval:.0f}) (good p ≤ {int(args.Pmax)}) ≈ {Le:.6f}")

    # --- GL1 + common fit info ---
    a,b,c = coef
    print(f"[fit] N(T) ≈ {a:.6f} T log T + {b:.6f} T + {c:.2f}")
    fr_info = "(off)" if fr_coeffs is None else " + ".join([f"{fr_coeffs[i]:.3g}·ω^{i}" for i in range(len(fr_coeffs))])
    print(f"[freq-cal] deg={args.fr_deg} Pcal≤{args.fr_Pcal} use_p2={'yes' if args.fr_use_p2 else 'no'}  ridge={args.fr_lam:g} trim={args.fr_trim:g}  s(ω)={fr_info}")
    print(f"[calib] {args.calib}  scalar s={s:.6g}")

    # Degree-1 tests
    if args.degree == 1:
        t1 = test_degree1(coeffs_cal_gl1, Ptest=args.Ptest)
        print("[GL1] zeta-like test:")
        print(f"      RMSE(b_p vs 1): {t1['rmse_p']:.4g},  frac |b_p-1|≤0.1: {t1['frac_within_10pct_p']:.3f},  primes tested: {int(t1['tested_primes'])}")
        print(f"      RMSE(b_p^2 vs 1): {t1['rmse_p2']:.4g}")
    else:
        # (kept GL2 hook if you need it later)
        pass

    # --- multiplicativity tests ---
    # For GL(1) zeta: the explicit formula only has prime powers p^m.
    # There is no linear pq line, so skip linear pq test unless --degree 2.
    if args.degree == 1 and args.mult_tests and args.mult_tests > 0:
        print("[note] GL(1) spectrum has only prime powers; linear pq test not applicable. "
              "Use --mult2-tests for quadratic pq.")

    elif args.mult_tests > 0:
        small = [p for p in primes_upto(min(args.Ptest, args.Pmax)) if p < 1000]
        mult = multiplicativity_samples(tg, Rg, w, coeffs_cal, small, K=args.mult_tests,
                                        fr_coeffs=fr_coeffs, scalar_s=s,
                                        baseline_delta=args.baseline_delta)
        print("[mult] squarefree composites (calibrated):")
        if mult.get("samples", 0) > 0:
            print(f"       samples={int(mult['samples'])}, corr={mult['corr']:.3f}, "
                  f"slope k={mult['slope_k']:.3f}, offset b={mult['offset_b']:.3f}, "
                  f"ratio RMSE={mult['ratio_rmse']:.4f}, logerr RMSE={mult['logerr_rmse']:.4f}")
        else:
            print("       (insufficient samples)")

    # Optional quadratic pq test (reveals ω = log(pq) via squaring)
    if args.mult2_tests and args.mult2_tests > 0:
        small = [p for p in primes_upto(min(args.Ptest, args.Pmax)) if p < 1000]
        m2 = multiplicativity_quadratic_samples(
            tg, Rg, w, coeffs_cal, small,
            K=args.mult2_tests, gdeg=args.mult2_deg, ridge=args.mult2_ridge,
            baseline_delta=args.baseline_delta
        )
        print("[mult^2] quadratic pq estimator:")
        if m2.get("samples", 0) > 0:
            print(f"         samples={int(m2['samples'])}, corr={m2['corr']:.3f}, "
                  f"rmse={m2['rmse']:.4f}, g2 coeffs={m2['g2_coeffs']}")
        else:
            print("         (insufficient samples)")

    # --- CSV output ---
    if args.csv_out:
        rows.sort(key=lambda r:(r.p,r.m))
        with open(args.csv_out,"w",newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["n","p","m","omega_star","a_hat","b_hat","amp_raw","amp_cal","kind"])
            for r in rows:
                wcsv.writerow([r.n,r.p,r.m,f"{r.omega_star:.9f}",f"{r.a_hat:.9g}",f"{r.b_hat:.9g}",
                               f"{r.amp_raw:.9g}", f"{(r.amp_cal if np.isfinite(r.amp_cal) else float('nan')):.9g}", r.kind])
        print(f"[csv] wrote {len(rows)} rows to {args.csv_out}")


    if args.Leval:
        try: svals = [float(x) for x in args.Leval.split(",") if x.strip()]
        except: svals = []
        if svals:
            print(f"[L] truncated L_spec(s) with M≤{args.Mmax}, P≤{args.Pmax}:")
            for sig in svals:
                val = build_L_spec_s(coeffs_cal, args.Pmax, args.Mmax, complex(sig,0.0))
                print(f"    s={sig:.3f}  |L|={abs(val):.6g},  Re(L)={val.real:.6g}, Im(L)={val.imag:.6g}")

if __name__ == "__main__":
    main()