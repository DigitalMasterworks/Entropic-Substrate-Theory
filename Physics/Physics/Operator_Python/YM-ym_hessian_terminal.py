#!/usr/bin/env python3


import math, sys
try:
 import numpy as np
except Exception:
 np = None
try:
 import scipy.sparse as sp
 import scipy.sparse.linalg as spla
except Exception:
 sp = None
 spla = None
from fractions import Fraction







def build_su2_hessian_approx(n):
 """
 Build a toy SU(2) Hessian with DOF = 3 * (
 This is a linearized proxy that includes the group multiplicity (3) and nearest-neighbor link couplings.
 For production, replace with exact second derivatives from Wilson plaquette terms.
 """
 if np is None:
 raise RuntimeError("numpy required")

 nlinks = 2 * n * n
 dim = 3 * nlinks
 H = np.zeros((dim, dim), dtype=float)


 def link_index(i, j, dir):
 return (i % n) * n * 2 + (j % n) * 2 + dir
 for i in range(n):
 for j in range(n):
 for d in (0,1):
 li = link_index(i,j,d)
 for a in range(3):
 idx = 3*li + a
 H[idx, idx] += 4.0

 if d == 0:
 ni, nj, nd = i, j+1, 1
 else:
 ni, nj, nd = i+1, j, 0
 lj = link_index(ni, nj, nd)
 for a in range(3):
 H[3*li + a, 3*lj + a] += -1.0
 H[3*lj + a, 3*li + a] += -1.0
 return H

def lowest_nonzero_eig_dense(H):
 eps = 1e-12
 w, _ = np.linalg.eigh(H)
 w = np.real(w)
 w.sort()
 nz = [v for v in w if v > eps]
 return float(nz[0]) if nz else None

def rationalize_float(x, max_den=10**12):

 return Fraction(x).limit_denominator(max_den)

def print_hessian_lean(H, name="H_approx"):
 """
 Print a sparse Lean matrix constructor (row list of (i,j,val)) and a theorem that lambda_min >= q
 Here we only print a rationalized lambda_min as a Rat; full matrix printing is possible but very long.
 """
 lam = lowest_nonzero_eig_dense(H)
 if lam is None:
 print("No nonzero eigenvalue found.")
 return
 frac = rationalize_float(lam)
 print("-- Lean snippet (terminal):")
 print(f"def lambda_min_n: Rat:= {frac.numerator} / {frac.denominator}")
 print(f"theorem lambda_min_n_def: lambda_min_n = {frac.numerator} / {frac.denominator}:= by rfl")
 print(f"-- numeric lambda_min ≈ {lam}")

def main():
 sizes = [2,3,4]
 for n in sizes:
 print(f"\nBuilding toy SU(2) Hessian (approx) for n={n}...")
 H = build_su2_hessian_approx(n)
 lam = lowest_nonzero_eig_dense(H)
 print(f"n={n}, dim={H.shape[0]}, lambda_min ≈ {lam}")
 frac = rationalize_float(lam, max_den=10**9)
 print(f"Rational approx: {frac.numerator}/{frac.denominator} ≈ {float(frac)}")
 print_hessian_lean(H, name=f"H_n_{n}")

if __name__ == '__main__':
 main()