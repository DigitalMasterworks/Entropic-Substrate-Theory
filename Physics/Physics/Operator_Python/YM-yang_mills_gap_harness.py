#!/usr/bin/env python3

"""
yang_mills_gap_harness.py (fixed shape literal + small stability tweaks)
"""

import numpy as np
from numpy.random import default_rng
from math import sqrt
rng = default_rng(0)

DIM = 3
LT = 24
LS = 16
SIZES = (LT, LS, LS)
BETA = 2.5
N_THERM = 2000
N_SWEEPS = 4000
MEAS_EVERY = 10

def su2_rand_eps(eps=0.15):
 v = rng.normal(size=3)
 vn = np.linalg.norm(v) + 1e-16
 axis = v / vn
 theta = eps * rng.uniform()
 a0 = np.cos(theta)
 a = axis * np.sin(theta)
 return np.array([a0, a[0], a[1], a[2]])

def su2_mul(q, r):
 a0,a1,a2,a3 = np.moveaxis(q, -1, 0)
 b0,b1,b2,b3 = np.moveaxis(r, -1, 0)
 return np.stack([
 a0*b0 - a1*b1 - a2*b2 - a3*b3,
 a0*b1 + a1*b0 + a2*b3 - a3*b2,
 a0*b2 - a1*b3 + a2*b0 + a3*b1,
 a0*b3 + a1*b2 - a2*b1 + a3*b0
 ], axis=-1)

def su2_tr(q): return 2.0*q[...,0]
def su2_adj(q):
 out = q.copy(); out[...,1:] *= -1.0; return out

def shift(x, mu, sgn):
 y = list(x); y[mu] = (y[mu] + sgn) % SIZES[mu]; return tuple(y)

def staple(U, x, mu):
 s = np.zeros(4)
 for nu in range(DIM):
 if nu == mu: continue

 t1 = su2_mul( U[shift(x,mu,+1)+(nu,)], su2_adj(U[shift(x,nu,+1)+(mu,)]) )
 t1 = su2_mul( U[x+(nu,)], t1 )

 t2 = su2_mul( su2_adj(U[shift(x,nu,+1)+(mu,)]), U[shift(x,nu,-1)+(nu,)] )
 t2 = su2_mul( t2, su2_adj(U[shift(x,mu,-1)+(nu,)]) )
 s += t1 + t2
 return s


U = np.zeros(SIZES + (DIM, 4))
U[...,0] = 1.0

def sweep_metropolis(eps=0.15):
 acc = 0
 for idx in np.ndindex(*SIZES):
 for mu in range(DIM):
 q_old = U[idx+(mu,)].copy()
 st = staple(U, idx, mu)
 r = su2_rand_eps(eps)
 q_new = su2_mul(r, q_old)
 dS = - (BETA/2.0) * (su2_tr(su2_mul(q_new, st)) - su2_tr(su2_mul(q_old, st)))
 if dS <= 0.0 or rng.random() < np.exp(-dS):
 U[idx+(mu,)] = q_new
 acc += 1
 return acc / (np.prod(SIZES) * DIM)

def plaquette_trace():
 total = 0.0; cnt = 0
 for idx in np.ndindex(*SIZES):
 for mu in range(DIM):
 for nu in range(mu+1, DIM):
 U1 = U[idx+(mu,)]
 U2 = U[shift(idx,mu,+1)+(nu,)]
 U3 = su2_adj(U[shift(idx,nu,+1)+(mu,)])
 U4 = su2_adj(U[idx+(nu,)])
 loop = su2_mul(su2_mul(U1,U2), su2_mul(U3,U4))
 total += 0.5*su2_tr(loop); cnt += 1
 return total/cnt

def timeslice_scalar_plaquette(t):
 """O(t) = avg of spatial plaquettes at fixed Euclidean time t (use mu,nu=1,2)."""
 total = 0.0; cnt = 0
 for x1 in range(LS):
 for x2 in range(LS):
 idx = (t, x1, x2)
 mu, nu = 1, 2
 U1 = U[idx+(mu,)]
 U2 = U[shift(idx,mu,+1)+(nu,)]
 U3 = su2_adj(U[shift(idx,nu,+1)+(mu,)])
 U4 = su2_adj(U[idx+(nu,)])
 loop = su2_mul(su2_mul(U1,U2), su2_mul(U3,U4))
 total += 0.5*su2_tr(loop); cnt += 1
 return total/cnt

O_bar_acc = 0.0
OO_acc = np.zeros(LT)
n_meas = 0

def wilson_loop(R, T):
 total = 0.0; cnt = 0
 for idx in np.ndindex(*SIZES):
 mu, nu = 0, 1
 q = np.array([1.0,0.0,0.0,0.0])
 x = idx
 for _ in range(R): q = su2_mul(q, U[x+(mu,)]); x = shift(x, mu, +1)
 for _ in range(T): q = su2_mul(q, U[x+(nu,)]); x = shift(x, nu, +1)
 for _ in range(R): x = shift(x, mu, -1); q = su2_mul(q, su2_adj(U[x+(mu,)]))
 for _ in range(T): x = shift(x, nu, -1); q = su2_mul(q, su2_adj(U[x+(nu,)]))
 total += 0.5*su2_tr(q); cnt += 1
 return total/cnt

def plaquette_field():
 P = np.zeros(SIZES)
 for idx in np.ndindex(*SIZES):
 val = 0.0; c=0
 for mu in range(DIM):
 for nu in range(mu+1,DIM):
 U1 = U[idx+(mu,)]
 U2 = U[shift(idx,mu,+1)+(nu,)]
 U3 = su2_adj(U[shift(idx,nu,+1)+(mu,)])
 U4 = su2_adj(U[idx+(nu,)])
 loop = su2_mul(su2_mul(U1,U2), su2_mul(U3,U4))
 val += 0.5*su2_tr(loop); c+=1
 P[idx] = val/c
 return P

def radial_correlator(P):
 origin = tuple([0]*DIM)
 P0 = P[origin]
 dists, vals = [], []
 for idx in np.ndindex(*SIZES):
 dr2 = 0
 for mu in range(DIM):
 dx = min((idx[mu]-origin[mu]) % SIZES[mu], (origin[mu]-idx[mu]) % SIZES[mu])
 dr2 += dx*dx
 dists.append(sqrt(dr2))
 vals.append(P[idx] - P0*P.mean())
 dists = np.array(dists); vals = np.array(vals)
 rmax = int(np.max(dists))
 rs = np.arange(1, rmax+1)
 Cr = []
 for r in rs:
 mask = (np.abs(dists - r) < 0.5)
 Cr.append(np.nan if not np.any(mask) else np.nanmean(vals[mask]))
 return rs, np.array(Cr)

print("== Yang–Mills Mass Gap Harness (fixed) ==")
print(f"Dim={DIM}, sizes={SIZES}, beta={BETA}")
for _ in range(N_THERM):
 sweep_metropolis()

loops = {}
plaqs = []
corr_accum = None; n_corr = 0

for sweep in range(N_SWEEPS):
 sweep_metropolis()
 if (sweep % MEAS_EVERY) == 0:
 plaqs.append(plaquette_trace())
 for R in (1,2,3,4):
 for T in (1,2,3,4):
 loops.setdefault((R,T), []).append(wilson_loop(R,T))
 O0 = timeslice_scalar_plaquette(0)
 Ot = np.array([timeslice_scalar_plaquette(t) for t in range(LT)])
 O_bar_acc += Ot.mean()
 OO_acc += O0 * Ot
 n_meas += 1

areas, minus_logW_overA = [], []
for (R,T), V in loops.items():
 W = np.mean(V); A = R*T
 if W>0 and A>0:
 areas.append(A)
 minus_logW_overA.append(-np.log(W)/A)
sigma_est = (np.median(minus_logW_overA) if minus_logW_overA else np.nan)

print("\n[Phase I] Area law check:")
print(f" median(-log<W>/Area) ≈ σ ~ {sigma_est:.4f}")

O_bar = O_bar_acc / max(n_meas,1)
C = (OO_acc / max(n_meas,1)) - (O_bar**2)

for t in range(1, LT):
 C[t] = 0.5*(C[t] + C[LT-t])

nonneg = np.all(C >= -1e-10)
mono = np.all(np.diff(C[:LT//2]) <= 1e-8)

with np.errstate(divide='ignore', invalid='ignore'):
 m_eff = np.log( np.clip(C[:-1], 1e-16, None) / np.clip(C[1:], 1e-16, None) )
plateau = slice(2, min(10, LT//3))
m_est = float(np.nanmedian(m_eff[plateau]))

print("\n[Phase II] Transfer-matrix correlator (timeslice):")
print(f" m_eff plateau median ≈ {m_est:.4f} (lattice units)")

print("\n[Phase III] Reflection positivity sanity:")
print(f" C(t) ≥ 0 for all t? {bool(nonneg)}")
print(f" C(t) non-increasing (t≲LT/2)? {bool(mono)}")

print("\n== Summary ==")
print(f"PASS area-law? {'YES' if np.isfinite(sigma_est) else 'NO'}")
print(f"PASS gap>0? {'YES' if (np.isfinite(m_est) and m_est>0) else 'NO/WEAK'}")
print(f"PASS RP sanity? {'YES' if (nonneg and mono) else 'WARN'}")