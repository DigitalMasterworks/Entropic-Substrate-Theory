#!/usr/bin/env python3
"""
ym_wilson_hessian_fd.py

Terminal-only SU(2) Wilson-plaquette Hessian assembler (finite-difference at U=I).

Usage example:
 python3 ym_wilson_hessian_fd.py --n 3 --beta 1.0 --eps 1e-6 --dps 80 --lean-matrix

Options:
 --n lattice linear size (n x n torus), default 3
 --beta Wilson coupling prefactor (only scales Hessian), default 1.0
 --eps finite-difference step size, default 1e-6
 --dps mpmath working precision (digits), default 80
 --lean-matrix if present, print a rationalized Lean Matrix literal for the Hessian (may be large)
 --max-den max denominator for rationalization in Lean printing (default 1e12)

Notes:
 - This constructs U = exp(i * (theta · sigma) / 2) for each link.
 - Action S = sum_p Re Tr(I - U_p). At identity S=0 and Hessian is positive semidefinite.
 - Gauge zero-modes will exist; we identify tiny eigenvalues and ignore them when printing smallest NONZERO eigenvalue.
 - For formal import to Lean: use --lean-matrix or pipe H through rationalizer.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import math
import sys

try:
 import mpmath as mp
except Exception:
 mp = None

try:
 import numpy as np
except Exception:
 np = None

try:
 import scipy.linalg as sla
except Exception:
 sla = None




def pauli_matrices(mp_ctx):
 j = mp_ctx.j if mp_ctx is not None else 1j

 if mp_ctx is not None:
 sigma1 = mp_ctx.matrix([[0, 1], [1, 0]])
 sigma2 = mp_ctx.matrix([[0, -j], [j, 0]])
 sigma3 = mp_ctx.matrix([[1, 0], [0, -1]])
 else:
 sigma1 = np.array([[0+0j, 1+0j],[1+0j, 0+0j]])
 sigma2 = np.array([[0+0j, -1j],[1j, 0+0j]])
 sigma3 = np.array([[1+0j, 0+0j],[0+0j, -1+0j]])
 return sigma1, sigma2, sigma3

def su2_exp_matrix(theta_vec, mp_ctx):
 """
 Given theta_vec = (t1,t2,t3) as mpmath mpf or floats, return 2x2 SU(2) matrix
 U = exp(i * (theta · sigma) / 2).
 """
 if mp_ctx is None:

 t1, t2, t3 = float(theta_vec[0]), float(theta_vec[1]), float(theta_vec[2])
 sigma1 = np.array([[0,1],[1,0]], dtype=complex)
 sigma2 = np.array([[0,-1j],[1j,0]], dtype=complex)
 sigma3 = np.array([[1,0],[0,-1]], dtype=complex)
 A = 1j * (t1 * sigma1 + t2 * sigma2 + t3 * sigma3) / 2.0
 return sla.expm(A)
 else:
 mp = mp_ctx
 t1, t2, t3 = mp.mpf(theta_vec[0]), mp.mpf(theta_vec[1]), mp.mpf(theta_vec[2])
 sigma1, sigma2, sigma3 = pauli_matrices(mp)
 A = mp.j * (t1 * sigma1 + t2 * sigma2 + t3 * sigma3) / mp.mpf(2)
 return mp.expm(A)




def build_index_maps(n):
 """
 For n x n lattice:
 - links are enumerated: for each site (i,j), two directed links: dir 0 = x (to i,j+1), dir 1 = y (to i+1,j)
 - return mapping: link_idx(i,j,dir) -> index in [0, nlinks-1]
 """
 nlinks = 2 * n * n
 def link_idx(i, j, d):
 return ((i % n) * n + (j % n)) * 2 + d
 return link_idx, nlinks




def action_from_theta_vec(theta_vec, n, mp_ctx, beta=1.0):
 """
 theta_vec is length 3 * nlinks (list or numpy array); interpreted as 3 components per link.
 Returns S = beta * sum_p Re Tr(I - U_p)
 """
 if mp_ctx is None and np is None:
 raise RuntimeError("Need numpy or mpmath available.")
 link_idx, nlinks = build_index_maps(n)

 Us = [None] * nlinks
 for i in range(n):
 for j in range(n):
 for d in (0,1):
 idx = link_idx(i,j,d)
 base = 3*idx
 t = theta_vec[base:base+3]
 Us[idx] = su2_exp_matrix(t, mp_ctx)

 sum_val = mp_ctx.mpf(0) if mp_ctx is not None else 0.0
 for i in range(n):
 for j in range(n):


 ux = Us[link_idx(i,j,0)]
 uy = Us[link_idx(i,j,1)]

 ux_right = Us[link_idx(i, j+1, 0)]
 uy_up = Us[link_idx(i+1, j, 1)]

 if mp_ctx is None:

 Up = ux @ uy_up @ np.linalg.inv(ux_right) @ np.linalg.inv(uy)
 tr = np.trace(np.eye(2, dtype=complex) - Up)
 sum_val += float(np.real(tr))
 else:

 try:
 inv_ux_right = mp_ctx.inverse(ux_right)
 inv_uy = mp_ctx.inverse(uy)
 except AttributeError:


 inv_ux_right = mp_ctx.transpose(mp_ctx.conj(ux_right))
 inv_uy = mp_ctx.transpose(mp_ctx.conj(uy))

 Up = ux * uy_up * inv_ux_right * inv_uy
 I = mp_ctx.matrix([[1,0],[0,1]])
 tr = (I - Up)[0,0] + (I - Up)[1,1]

 sum_val += mp_ctx.re(tr)
 return beta * sum_val




def assemble_hessian_fd(n, eps=1e-6, dps=80, beta=1.0, mp_dps=None):
 """
 Assemble Hessian H (real symmetric) of S at theta=0 using central finite differences.
 H_ij = ∂^2 S / ∂x_i ∂x_j at 0 approximated by:
 H_ij ≈ ( S(+ei*eps + ej*eps) - S(+ei*eps - ej*eps) - S(-ei*eps + ej*eps) + S(-ei*eps - ej*eps) )
 / (4 eps^2)
 For i == j we use second central difference:
 H_ii ≈ ( S(+2eps*ei) - 2 S(0) + S(-2eps*ei) ) / (4 eps^2) (or standard formula with eps)
 Return H as numpy 2D float array.
 """
 if mp is None:
 mp_ctx = None
 else:
 mp_ctx = mp
 if mp_dps is not None:
 mp_ctx.mp.dps = mp_dps
 else:
 mp_ctx.mp.dps = dps
 link_idx, nlinks = build_index_maps(n)
 dim = 3 * nlinks

 zero_theta = [0]*dim

 S0 = action_from_theta_vec(zero_theta, n, mp_ctx, beta=beta)
 H = [[0.0]*dim for _ in range(dim)]

 def theta_with(idx, val):
 v = [0.0]*dim
 v[idx] = val
 return v

 for i in range(dim):
 for j in range(i, dim):
 if i == j:


 th_p = [0.0]*dim
 th_m = [0.0]*dim
 th_p[i] = eps
 th_m[i] = -eps
 Sp = action_from_theta_vec(th_p, n, mp_ctx, beta=beta)
 Sm = action_from_theta_vec(th_m, n, mp_ctx, beta=beta)

 Hij = (Sp - 2*S0 + Sm) / (eps*eps)
 else:

 th_pp = [0.0]*dim; th_pm = [0.0]*dim; th_mp = [0.0]*dim; th_mm = [0.0]*dim
 th_pp[i] = eps; th_pp[j] = eps
 th_pm[i] = eps; th_pm[j] = -eps
 th_mp[i] = -eps; th_mp[j] = eps
 th_mm[i] = -eps; th_mm[j] = -eps
 S_pp = action_from_theta_vec(th_pp, n, mp_ctx, beta=beta)
 S_pm = action_from_theta_vec(th_pm, n, mp_ctx, beta=beta)
 S_mp = action_from_theta_vec(th_mp, n, mp_ctx, beta=beta)
 S_mm = action_from_theta_vec(th_mm, n, mp_ctx, beta=beta)
 Hij = (S_pp - S_pm - S_mp + S_mm) / (4 * eps * eps)

 try:
 val = float(Hij)
 except Exception:

 val = float(mp.nstr(Hij, 20))
 H[i][j] = val
 H[j][i] = val


 if (i+1) % 10 == 0 or i == dim-1:
 print(f" assembled row {i+1}/{dim}", file=sys.stdout)
 sys.stdout.flush()
 import numpy as _np
 return _np.array(H, dtype=float)




def smallest_nonzero_eigen(H, zero_thresh=1e-8):
 import numpy as _np
 w = _np.linalg.eigvalsh(H)
 w = _np.real(w)
 w.sort()
 nz = [v for v in w if v > zero_thresh]
 if not nz:
 return None, w
 return float(nz[0]), w

def rationalize_float(x, max_den=10**12):
 return Fraction(x).limit_denominator(int(max_den))

def print_lean_matrix_from_numpy(H, name="H_n", max_den=10**12):
 n, m = H.shape
 print(f"-- Lean matrix {name}: size {n}x{m}")
 print(f"def {name}: Matrix (Fin {n}) (Fin {m}) Rat:=")
 print(" Array.mk [")
 for i in range(n):
 row_entries = []
 for j in range(m):
 r = rationalize_float(float(H[i,j]), max_den=max_den)
 row_entries.append(f"{r.numerator} / {r.denominator}")
 print(" [" + ", ".join(row_entries) + "],")
 print(" ]")




def main(argv=None):
 parser = argparse.ArgumentParser()
 parser.add_argument("--n", type=int, default=3, help="lattice size n (n x n torus)")
 parser.add_argument("--beta", type=float, default=1.0, help="Wilson beta prefactor")
 parser.add_argument("--eps", type=float, default=1e-6, help="finite difference step")
 parser.add_argument("--dps", type=int, default=80, help="mpmath digits precision")
 parser.add_argument("--mp-dps", type=int, default=None, help="explicit mpmath dps (overrides dps if set)")
 parser.add_argument("--lean-matrix", action="store_true", help="print rationalized Lean matrix for Hessian")
 parser.add_argument("--max-den", type=float, default=1e12, help="max denominator for rationalization")
 args = parser.parse_args(argv)

 if mp is None and np is None:
 print("Please install at least one of mpmath or numpy (mpmath recommended)", file=sys.stderr)
 sys.exit(2)

 if mp is not None:
 mp.mp.dps = args.dps if args.mp_dps is None else args.mp_dps

 print(f"Building Hessian for n={args.n}, beta={args.beta}, eps={args.eps}, dps={mp.mp.dps if mp is not None else 'N/A'}")
 H = assemble_hessian_fd(args.n, eps=args.eps, dps=args.dps, beta=args.beta, mp_dps=args.mp_dps)
 print("Assembled Hessian matrix (float) with shape:", H.shape)

 lam, spectrum = smallest_nonzero_eigen(H, zero_thresh=1e-8)
 print("Smallest nonzero eigenvalue (float) =", lam)
 if lam is not None:
 frac = rationalize_float(lam, max_den=args.max_den)
 print(f"Rational approx: {frac.numerator}/{frac.denominator} ≈ {float(frac)}")
 print("-- Lean snippet (terminal):")
 print(f"def lambda_min_n: Rat:= {frac.numerator} / {frac.denominator}")
 print(f"theorem lambda_min_n_def: lambda_min_n = {frac.numerator} / {frac.denominator}:= by rfl")
 else:
 print("No nonzero eigenvalue found above threshold (possible all zero modes).")

 if args.lean_matrix:
 print_lean_matrix_from_numpy(H, name=f"H_wilson_n{args.n}", max_den=args.max_den)

if __name__ == "__main__":
 main()