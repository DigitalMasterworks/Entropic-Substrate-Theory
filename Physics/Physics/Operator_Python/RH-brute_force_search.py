#!/usr/bin/env python3
"""
sweep_candidates.py — test candidate 1D substrate functions S(x)
and compare Weyl-law growth.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from numpy.linalg import lstsq
import math, time


def build_S_candidate(x, form="linear", alpha=1.0, beta=0.0, c=1.0):
 r = np.abs(x)
 if form == "linear":
 return r
 if form == "log":
 return r * np.log1p(r)
 if form == "exp":
 return np.exp(r/c)
 if form == "linear_exp":
 return r * np.exp(r/c)
 if form == "power_log":
 return (r**alpha) * (np.log1p(r)**beta)
 raise ValueError(f"unknown form={form}")


def build_H_1d(S, L):
 N = len(S)
 h = L/(N-1)
 S2 = S*S
 c = 0.5*(S2[:-1] + S2[1:]) / (h*h)

 main = np.zeros(N)
 lower = np.zeros(N-1)
 upper = np.zeros(N-1)

 for i in range(N-1):
 w = c[i]
 main[i] += w
 main[i+1] += w
 lower[i] -= w
 upper[i] -= w

 H = diags([main, lower, upper], [0,-1,1], shape=(N,N), format="csr")
 return H


def fit_weyl(vals):
 lam = np.sort(vals[vals > 1e-12])
 Ntot = len(lam)
 N_of_lam = np.arange(1, Ntot+1, dtype=float)

 root = np.sqrt(lam)
 m0 = int(0.8*Ntot)
 X = np.vstack([root[m0:], np.log(lam[m0:]+1), np.ones_like(root[m0:])]).T

 coef, *_ = lstsq(X, N_of_lam[m0:], rcond=None)
 a,b,c = coef
 return a,b,c

def run_candidate(form, N=400, L=100, k=200, **kwargs):
 x = np.linspace(-L/2, L/2, N)
 S = build_S_candidate(x, form=form, **kwargs)
 H = build_H_1d(S, L)
 vals = eigsh(H, k=k, which="SM", return_eigenvectors=False)
 a,b,c = fit_weyl(vals)
 print(f"[{form}] Weyl fit: N(λ) ≈ {a:.4f} sqrt(λ) + {b:.4f} log(λ) + {c:.2f}")

if __name__ == "__main__":
 t0 = time.time()
 run_candidate("linear")
 run_candidate("log")
 run_candidate("exp", c=10.0)
 run_candidate("linear_exp", c=10.0)
 run_candidate("power_log", alpha=1.0, beta=1.0)
 run_candidate("power_log", alpha=1.0, beta=0.5)
 run_candidate("power_log", alpha=0.5, beta=1.0)
 print(f"done in {time.time()-t0:.2f}s")