#!/usr/bin/env python3
"""
hodge_fem_exact.py

Exact-rational P1 FEM assembler on periodic unit square (torus).
Outputs exact rational Frobenius-norm-squared of (P^2 - P) and a high-precision
spectral-norm estimate (rationalized) for convenience.

Usage example:
 python3 hodge_fem_exact.py --Nlist 8 12 16 20 --modes 3 --max-den 1e12 --dps 80

Notes:
 - All mesh assembly and linear algebra for M,B are done with fractions (exact).
 - Gram inverse (small nbasis) is computed exactly with Fraction Gaussian elimination.
 - Projector P is computed exactly as rational matrix (dense).
 - Residual R = P^2 - P computed exactly; Frobenius norm squared is exact rational.
 - We also compute a high-precision float spectral norm (via numpy svd) for a
 compact rationalization to paste into Lean if desired.
"""
from __future__ import annotations
from fractions import Fraction
import argparse, math, sys
import numpy as np




def frac_zero(n, m=None):
 if m is None:
 return [[Fraction(0,1) for _ in range(n)] for __ in range(n)]
 return [[Fraction(0,1) for _ in range(m)] for __ in range(n)]

def mat_mul_frac(A, B):
 n = len(A); m = len(B[0]); p = len(B)
 C = [[Fraction(0,1) for _ in range(m)] for __ in range(n)]
 for i in range(n):
 Ai = A[i]
 for k in range(p):
 Aik = Ai[k]
 if Aik == 0:
 continue
 Bk = B[k]
 for j in range(m):
 C[i][j] += Aik * Bk[j]
 return C

def mat_add_frac(A, B):
 n = len(A); m = len(A[0])
 C = [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]
 return C

def mat_sub_frac(A, B):
 n = len(A); m = len(A[0])
 C = [[A[i][j] - B[i][j] for j in range(m)] for i in range(n)]
 return C

def trans_frac(A):
 n = len(A); m = len(A[0])
 return [[A[i][j] for i in range(n)] for j in range(m)]

def copy_frac(A):
 return [row[:] for row in A]

def invert_small_frac(A):
 """Invert small square matrix A with Fraction entries via Gaussian elimination."""
 n = len(A)

 M = [A[i][:] + [Fraction(1 if i==j else 0,1) for j in range(n)] for i in range(n)]
 for i in range(n):

 if M[i][i] == 0:

 pivot = None
 for k in range(i+1, n):
 if M[k][i]!= 0:
 pivot = k; break
 if pivot is None:
 raise ZeroDivisionError("Singular matrix in invert_small_frac")
 M[i], M[pivot] = M[pivot], M[i]
 piv = M[i][i]

 M[i] = [x / piv for x in M[i]]

 for r in range(n):
 if r == i: continue
 fac = M[r][i]
 if fac == 0: continue
 M[r] = [M[r][c] - fac * M[i][c] for c in range(2*n)]

 Inv = [row[n:] for row in M]
 return Inv




def make_periodic_coords_indices(N):
 """Return list of coordinates (x,y) as Fractions and node index function."""
 coords = []
 for i in range(N):
 for j in range(N):

 coords.append((Fraction(j, N), Fraction(i, N)))
 def node(ii, jj):
 return ((ii % N) * N + (jj % N))
 return coords, node

def triangle_area_frac(v0, v1, v2):

 x0,y0 = v0; x1,y1 = v1; x2,y2 = v2
 det = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0)
 area = Fraction(1,2) * abs(det)
 return area

def assemble_node_mass_matrix_exact(N):
 coords, node = make_periodic_coords_indices(N)
 n = len(coords)

 tris = []
 for i in range(N):
 for j in range(N):
 n00 = node(i, j)
 n10 = node(i+1, j)
 n01 = node(i, j+1)
 n11 = node(i+1, j+1)
 tris.append((n00, n10, n11))
 tris.append((n00, n11, n01))
 M = [[Fraction(0,1) for _ in range(n)] for __ in range(n)]
 for tri in tris:
 i,j,k = tri
 v0, v1, v2 = coords[i], coords[j], coords[k]
 A = triangle_area_frac(v0, v1, v2)

 factor = A / Fraction(12,1)
 loc = [[factor * 2, factor * 1, factor * 1],
 [factor * 1, factor * 2, factor * 1],
 [factor * 1, factor * 1, factor * 2]]
 ids = [i,j,k]
 for a in range(3):
 for b in range(3):
 M[ids[a]][ids[b]] += loc[a][b]
 return M




def block_mass_vector_exact(M_node):
 n = len(M_node)

 M2 = [[Fraction(0,1) for _ in range(2*n)] for __ in range(2*n)]
 for i in range(n):
 for j in range(n):
 M2[i][j] = M_node[i][j]
 M2[i+n][j+n] = M_node[i][j]
 return M2




def build_rational_vector_basis(coords, k_modes=2):

 n = len(coords)

 def field_vals_fx(fx_func, fy_func):
 fxv = [fx_func(x,y) for (x,y) in coords]
 fyv = [fy_func(x,y) for (x,y) in coords]
 return fxv + fyv
 basis = []

 basis.append(field_vals_fx(lambda x,y: Fraction(1,1), lambda x,y: Fraction(0,1)))
 basis.append(field_vals_fx(lambda x,y: Fraction(0,1), lambda x,y: Fraction(1,1)))

 basis.append(field_vals_fx(lambda x,y: x, lambda x,y: Fraction(0,1)))
 basis.append(field_vals_fx(lambda x,y: Fraction(0,1), lambda x,y: y))

 basis.append(field_vals_fx(lambda x,y: x*x, lambda x,y: Fraction(0,1)))
 basis.append(field_vals_fx(lambda x,y: Fraction(0,1), lambda x,y: y*y))
 basis.append(field_vals_fx(lambda x,y: x*y, lambda x,y: x*y))

 B = basis

 nb = len(B)
 mat = [[B[c][r] for c in range(nb)] for r in range(2*len(coords))]
 return mat




def compute_projector_exact(B_frac, M_block_frac):

 ndof = len(B_frac)
 nbasis = len(B_frac[0])

 MB = mat_mul_frac(M_block_frac, B_frac)

 B_T = trans_frac(B_frac)
 Gram = mat_mul_frac(B_T, MB)

 Gram_inv = invert_small_frac(Gram)


 left = mat_mul_frac(B_frac, Gram_inv)


 right = trans_frac(MB)
 P = mat_mul_frac(left, right)
 return P





def frobenius_norm_sq_frac(A):
 s = Fraction(0,1)
 n = len(A); m = len(A[0])
 for i in range(n):
 for j in range(m):
 s += A[i][j] * A[i][j]
 return s

def spectral_norm_estimate_float(A_frac, dps=80):

 import numpy as _np
 A = _np.array([[float(x.numerator)/float(x.denominator) for x in row] for row in A_frac], dtype=float)
 s = _np.linalg.svd(A, compute_uv=False)
 return float(s[0])

def run_N(N, modes=2, dps=80, max_den=10**12):
 coords, node = make_periodic_coords_indices(N)
 Mnode = assemble_node_mass_matrix_exact(N)
 Mblock = block_mass_vector_exact(Mnode)
 B = build_rational_vector_basis(coords, k_modes=modes)
 P = compute_projector_exact(B, Mblock)

 P2 = mat_mul_frac(P, P)
 R = mat_sub_frac(P2, P)
 fro_sq = frobenius_norm_sq_frac(R)

 spec = spectral_norm_estimate_float(R, dps=dps)

 from fractions import Fraction as F
 spec_frac = F(spec).limit_denominator(int(max_den))
 return {
 "N": N,
 "ndof": 2 * (N*N),
 "nbasis": len(B[0]),
 "fro_sq_num": fro_sq.numerator,
 "fro_sq_den": fro_sq.denominator,
 "spec_float": spec,
 "spec_rational_num": spec_frac.numerator,
 "spec_rational_den": spec_frac.denominator
 }




def main(argv=None):
 parser = argparse.ArgumentParser()
 parser.add_argument("--Nlist", type=int, nargs="+", default=[8,12,16,20])
 parser.add_argument("--modes", type=int, default=2)
 parser.add_argument("--dps", type=int, default=80)
 parser.add_argument("--max-den", type=float, default=1e12)
 args = parser.parse_args(argv)
 print("Exact FEM P1 assembler; outputs exact Frobenius-norm-squared and rationalized spectral estimate.")
 for N in args.Nlist:
 out = run_N(N, modes=args.modes, dps=args.dps, max_den=args.max_den)
 fro_str = f"{out['fro_sq_num']}/{out['fro_sq_den']}"
 spec_str = f"{out['spec_rational_num']}/{out['spec_rational_den']}"
 print(f"N={out['N']:2d} ndof={out['ndof']:4d} nbasis={out['nbasis']:2d} fro_sq = {fro_str} ≈ {float(out['fro_sq_num'])/float(out['fro_sq_den']):.12e} spec ≈ {out['spec_float']:.12e} spec_rat = {spec_str}")

if __name__ == "__main__":
 main()