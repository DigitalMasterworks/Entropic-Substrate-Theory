#!/usr/bin/env python3
"""
verify_entropy_riemann_1d.py — test 1D divergence-form operator
H = -d/dx( S(x)^2 d/dx ) with Dirichlet boundary
"""

import argparse, time, math
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def build_S_1d(N, L, eps=1e-6):
 x = np.linspace(-L/2, L/2, N, dtype=np.float64)
 R = np.abs(x)
 return np.maximum(R, eps)

def build_H_div_form_1d(N, L, eps=1e-6):
 S = build_S_1d(N, L, eps)
 S2 = S*S
 h = L/(N-1)


 c = 0.5*(S2[:-1] + S2[1:]) / (h*h)

 main = np.zeros(N, dtype=np.float64)
 lower = np.zeros(N-1, dtype=np.float64)
 upper = np.zeros(N-1, dtype=np.float64)

 for i in range(N-1):
 w = c[i]
 main[i] += w
 main[i+1] += w
 lower[i] -= w
 upper[i] -= w

 H = diags([main, lower, upper], [0, -1, 1], shape=(N,N), format="csr")
 return H


def verify_weyl(vals, verbose=True):
 lam = np.sort(vals[vals > 1e-12])
 Ntot = len(lam)
 N_of_lam = np.arange(1, Ntot+1, dtype=float)


 root = np.sqrt(lam)
 m0 = int(0.8*Ntot)
 X = np.vstack([root[m0:], np.ones_like(root[m0:])]).T
 y = N_of_lam[m0:]
 a, b = np.linalg.lstsq(X, y, rcond=None)[0]

 if verbose:
 print(f"[weyl] fit N(λ) ≈ {a:.6f} sqrt(λ) + {b:.3f}")
 print(f"[weyl] first/last counts: N({lam[0]:.3f})={N_of_lam[0]}, N({lam[-1]:.3f})={N_of_lam[-1]}")
 return a, b

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--N", type=int, default=400)
 ap.add_argument("--L", type=float, default=100.0)
 ap.add_argument("--k", type=int, default=200)
 args = ap.parse_args()

 print(f"[cfg] N={args.N}, L={args.L}, k={args.k}")
 t0 = time.time()
 H = build_H_div_form_1d(args.N, args.L)
 vals = eigsh(H, k=args.k, which="SM", return_eigenvectors=False)
 elapsed = time.time() - t0
 print(f"[eigs] got {len(vals)} eigenvalues in {elapsed:.2f}s")
 verify_weyl(vals, verbose=True)

if __name__ == "__main__":
 main()