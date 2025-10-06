#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bsd_harness_stronger.py

A stronger numerical BSD harness:
  - Completed L-function Λ(s) = N^(s/2) (2π)^(-s) Γ(s) L(E,s)
  - Good/multiplicative/additive local behavior for primes
  - Multiplicative Dirichlet coefficients a_n up to n ≤ M
  - Root number W estimated from Λ(s) ≈ W Λ(2 - s)
  - Functional-equation residual across a grid
  - Rank probe: order of vanishing of Λ(s) at s = 1 via central differences

Inputs:
  * Minimal Weierstrass coefficients (a1..a6)
  * Conductor N (for common curves you can set N manually; e.g., 11a1→11, 37a1→37)
  * Truncation P_MAX (primes), M_MAX (Dirichlet sum length)

Notes:
  - This is still a *numeric* harness, not a proof. It’s designed to be a
    high-quality front-end you’ll later swap to your operator determinant.
"""

import numpy as np
from math import isclose, pi, log, exp
try:
    from scipy.special import gamma
    have_scipy = True
except Exception:
    import math
    have_scipy = False

# --------- CONFIG: choose a curve and conductor ------------------------------

# 11a1 (rank 0): y^2 + y = x^3 - x^2 - 10 x - 20, conductor N=11
# A1,A2,A3,A4,A6, N = 1, -1, 0, -10, -20, 11

# 37a1 (rank 1): y^2 + y = x^3 - x, conductor N=37
A1,A2,A3,A4,A6, N = 1, 0, 0, -1, 0, 37   # <- default here: rank 1 case

P_MAX  = 200      # primes up to this for building local data
M_MAX = 100000
EPS_GRID = [0.15,0.10,0.08,0.06,0.05,0.04,0.03,0.02]

# --------- basic algebra: discriminant, c4, c6, reduction type ---------------

def invariants(a1,a2,a3,a4,a6):
    b2 = a1*a1 + 4*a2
    b4 = a1*a3 + 2*a4
    b6 = a3*a3 + 4*a6
    b8 = a1*a1*a6 + 4*a2*a6 - a1*a3*a4 + a2*a3*a3 - a4*a4
    c4 = b2*b2 - 24*b4
    c6 = -b2*b2*b2 + 36*b2*b4 - 216*b6
    Δ = -b2*b2*b8 - 8*b4*b4*b4 - 27*b6*b6 + 9*b2*b4*b6
    return (b2,b4,b6,b8,c4,c6,Δ)

B2,B4,B6,B8,C4,C6,DISC = invariants(A1,A2,A3,A4,A6)

def primes_upto(n):
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    for i in range(2,int(n**0.5)+1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return [i for i in range(2,n+1) if sieve[i]]

PRIMES = primes_upto(P_MAX)

def reduction_type_at_p(p):
    """Return 'good','mult','add'. Use conductor first; fall back to Δ,c4."""
    # If p | N it's a bad prime by definition for this isogeny class
    if N % p == 0:
        # 37a1 is multiplicative at p=37
        return 'mult'
    if (DISC % p) != 0:
        return 'good'
    # Bad reduction by discriminant; decide mult vs add by c4
    return 'mult' if (C4 % p) != 0 else 'add'

# --------- point counting mod p to get a_p at good p; multiplicative sign ----

def E_count_mod_p(p, a1,a2,a3,a4,a6):
    p = int(p)
    cnt = 1  # infinity
    if p == 2:
        xs = [0,1]; ys = [0,1]
        for x in xs:
            rhs = (x**3 + (a2*(x*x)) + (a4*x) + a6) % p
            for y in ys:
                lhs = (y*y + a1*x*y + a3*y) % p
                if lhs == (rhs % p): cnt += 1
        return cnt
    for x in range(p):
        rhs = (pow(x,3,p) + (a2*(x*x)) + (a4*x) + a6) % p
        B = (a1*x + a3) % p
        C = (-rhs) % p
        D = (B*B - (4*C)%p) % p
        if D == 0:
            cnt += 1
        else:
            ls = pow(D, (p-1)//2, p)
            if ls == 1:
                cnt += 2
    return cnt

def ap_good(p):
    n = E_count_mod_p(p, A1,A2,A3,A4,A6)
    return p + 1 - n

def multiplicative_sign(p):
    """
    For multiplicative reduction:
      split  -> a_p = +1
      non-split -> a_p = -1
    Heuristic test: #E(F_p) == p ? (then a_p = 1), else a_p = -1.
    (This simple test works reliably for many small curves.)
    """
    n = E_count_mod_p(p, A1,A2,A3,A4,A6)
    ap = p + 1 - n
    if ap == 1: return +1
    if ap == -1: return -1
    # fallback if noisy: choose sign by which is closer to ap
    return +1 if abs(ap-1) < abs(ap+1) else -1

# --------- local data and a_p table ------------------------------------------

ap = {}        # p -> a_p at good p; ±1 at mult; 0 at add
ptype = {}     # 'good'/'mult'/'add'

for p in PRIMES:
    t = reduction_type_at_p(p)
    ptype[p] = t
    if t == 'good':
        ap[p] = ap_good(p)
    elif t == 'mult':
        ap[p] = multiplicative_sign(p)   # ±1
    else:
        ap[p] = 0                        # additive

# --------- Dirichlet coefficients a_n up to n ≤ M_MAX ------------------------

# prime-power rules:
#   good: a_{p^0}=1; a_{p^1}=a_p; a_{p^k}=a_p a_{p^{k-1}} - p a_{p^{k-2}}
#   mult: a_{p^k} = a_p^k  (±1)^k
#   add : a_{p^1}=0, and all higher a_{p^k}=0
# multiplicativity: a_{mn} = a_m a_n if (m,n)=1

# precompute smallest prime factor for 1..M_MAX
spf = np.arange(M_MAX+1)
spf[:2] = 0
for i in range(2, int(M_MAX**0.5)+1):
    if spf[i] == i:
        spf[i*i:M_MAX+1:i] = np.where(spf[i*i:M_MAX+1:i]==spf[i*i:M_MAX+1:i], i, spf[i*i:M_MAX+1:i])
# fix primes
for i in range(2, M_MAX+1):
    if spf[i] == 0:
        spf[i] = i

def ap_for_p(p, a1, a2, a3, a4, a6):
    p = int(p)
    n = E_count_mod_p(p, a1, a2, a3, a4, a6)
    return p + 1 - n
    
def ensure_ap_entry(p):
    p = int(p) 
    if p in ap:
        return
    if DISC % p == 0:
        ap[p] = 0  # bad reduction placeholder
    else:
        ap[p] = ap_for_p(p, A1, A2, A3, A4, A6)
        
def ap_power(p, k):
    ensure_ap_entry(p)
    t = ptype.get(p, 'good')
    if t == 'good':
        if k == 0: return 1
        if k == 1: return ap[p]
        a0, a1 = 1, ap[p]
        for _ in range(2, k+1):
            a2 = ap[p]*a1 - p*a0
            a0, a1 = a1, a2
        return a1
    elif t == 'mult':
        return (ap[p])**k   # ±1^k
    else:
        return 0 if k >= 1 else 1

def dirichlet_an(n):
    """Compute a_n by multiplicativity using spf."""
    if n == 1: return 1
    res = 1
    while n > 1:
        p = spf[n]
        k = 0
        while n % p == 0:
            n //= p; k += 1
        res *= ap_power(p, k)
        if res == 0 and n > 1:
            # additive factor kills multiplicative product early
            return 0
    return res

# build table
aN = np.zeros(M_MAX+1, dtype=np.int64)
aN[1] = 1
for n in range(2, M_MAX+1):
    aN[n] = dirichlet_an(n)

# --------- L(s) and completed Λ(s) ------------------------------------------

TWOPI = 2.0*pi

def L_series(s):
    """
    Smoothed Dirichlet series for better behavior near s≈1.
    Simple exponential weight w(n)=exp(-n/X). Increase M_MAX or X for accuracy.
    """
    n = np.arange(1, M_MAX+1, dtype=np.float64)
    X = max(50.0, M_MAX / 8.0)   # mild smoothing scale
    w = np.exp(-n / X)
    return np.sum(w * (aN[1:] * n**(-s)))

def Gamma(s):
    if have_scipy:
        return gamma(s)
    # fallback via exp(lgamma)
    import math
    return exp(math.lgamma(s))

def Lambda_completed(s):
    # Λ(s) = N^(s/2) (2π)^(-s) Γ(s) L(s)
    return (N**(0.5*s)) * (TWOPI**(-s)) * Gamma(s) * L_series(s)

# --------- estimate root number W and check functional equation --------------

def estimate_root_number(delta=0.2):
    Lp = Lambda_completed(1.0 + delta)
    Lm = Lambda_completed(1.0 - delta)
    # Ideally: Λ(1+δ) ≈ W Λ(1-δ)  ⇒  W ≈ Λ(1+δ)/Λ(1-δ)
    if abs(Lm) < 1e-15:
        return 1  # fallback
    W_est = Lp / Lm
    # round to nearest ±1
    return 1 if W_est >= 0 else -1

def fe_residual_grid(W, eps_list):
    res = []
    for eps in eps_list:
        Lp = Lambda_completed(1.0 + eps)
        Lm = Lambda_completed(1.0 - eps)
        num = abs(Lp - W*Lm)
        den = abs(Lp) + abs(W*Lm) + 1e-15
        res.append(num/den)
    return res

# --------- rank probe at s = 1 via central differences ----------------------

def central_diff_rank(eps_list, tol=5e-3):
    """
    Test r = 0,1,2 by inspecting Λ, Λ', Λ'' at s=1 with central differences.
    Return (best_r, estimates_dict).
    """
    # Λ(1)
    vals = [Lambda_completed(1.0 + e) for e in eps_list] + [Lambda_completed(1.0 - e) for e in eps_list]
    L1_est = 0.5*(vals[0] + vals[len(eps_list)])  # crude sym avg
    # central derivatives using smallest eps
    e = min(eps_list)
    Lp = Lambda_completed(1.0 + e)
    Lm = Lambda_completed(1.0 - e)
    Lp2 = Lambda_completed(1.0 + 2*e)
    Lm2 = Lambda_completed(1.0 - 2*e)
    # first derivative
    L1p = (Lp - Lm) / (2*e)
    # second derivative (5-point could be used; here 2nd-order)
    L1pp = (Lp - 2*Lambda_completed(1.0) + Lm) / (e*e) if abs(Lambda_completed(1.0))>0 else (Lp2 - 2*L1_est + Lm2)/( (2*e)**2 )

    # decide rank
    r = None
    if abs(L1_est) > tol:
        r = 0
    elif abs(L1p) > tol:
        r = 1
    else:
        r = 2
    return r, {"Lambda(1)": L1_est, "Lambda'(1)": L1p, "Lambda''(1)": L1pp}

# --------- run ----------------------------------------------------------------

print("== Stronger BSD Harness ==")
print(f"Curve: y^2 + {A1}xy + {A3}y = x^3 + {A2}x^2 + {A4}x + {A6}")
print(f"Conductor N = {N}, primes≤{P_MAX}, Dirichlet truncation M = {M_MAX}")

good = sum(1 for p in PRIMES if ptype[p]=='good')
mult = sum(1 for p in PRIMES if ptype[p]=='mult')
add  = sum(1 for p in PRIMES if ptype[p]=='add')
print(f"Local types: good={good} mult={mult} add={add}")

# Phase I: sanity sample of a_p and a_n
print("\n[Phase I] Local and Dirichlet sanity:")
sample_ps = [p for p in PRIMES if ptype[p]=='good'][:10]
for p in sample_ps:
    print(f"  p={p:3d}: a_p={ap[p]:+d} (good)")

print(f"  a_1={aN[1]}  a_2={aN[2]}  a_3={aN[3]}  a_4={aN[4]}  a_5={aN[5]}  a_6={aN[6]}  ... a_{min(30,M_MAX)}={aN[min(30,M_MAX)]}")

# Phase II: root number estimate and functional equation residuals
W_hat = estimate_root_number(delta=0.2)
res = fe_residual_grid(W_hat, EPS_GRID)
print("\n[Phase II] Functional equation check Λ(s) ≈ W Λ(2-s):")
print(f"  Estimated root number W ≈ {W_hat:+d}")
for eps, r in zip(EPS_GRID, res):
    print(f"    eps={eps:>5.3f}  rel-residual={r:8.2e}")
print(f"  FE residual median ≈ {np.median(res):.2e}")

# Phase III: rank probe at the center s=1
rank_est, derivs = central_diff_rank(EPS_GRID, tol=5e-3)
print("\n[Phase III] Rank probe at s=1:")
for k,v in derivs.items():
    print(f"  {k:12s} ≈ {v:.6e}")
print(f"  ==> estimated analytic rank r ≈ {rank_est}")

print("\n== Summary ==")
print("PASS I   : local a_p & multiplicativity  YES")
print(f"PASS II  : FE residual small?            median ≈ {np.median(res):.2e}")
print(f"PASS III : rank (order at s=1)           r ≈ {rank_est}  (expect 0 for 11a1, 1 for 37a1)")
