#!/usr/bin/env python3
"""
hodge_fem_sampler.py

Terminal-only FEM sampler for Hodge projector evidence on the torus (periodic unit square).

Usage examples:
 python3 hodge_fem_sampler.py --Nlist 8 12 16 --modes 3 --rationalize --max-den 1000000000000

What it does:
 - Builds a uniform NxN triangular mesh on [0,1]^2 with periodic identifications.
 - Constructs P1 nodal mass matrix (assembled from triangle local mass matrices).
 - Builds a vector-field mass matrix M = diag(M_node, M_node) for (vx,vy) nodal vector fields.
 - Constructs a small geometry-tied basis B (samples of analytic 1-forms / low-frequency modes)
 expressed in nodal values (two components per node).
 - Forms projector P = B (B^T M B)^{-1} B^T M and computes ||P^2 - P||_2.
 - Prints N vs residual, and optionally rationalizes residuals to rationals (useful for Lean).
"""
from __future__ import annotations
import argparse
import math
from fractions import Fraction
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla




def make_periodic_grid_triangles(N):
 """
 Create node coordinates and triangle connectivity for NxN periodic grid.
 Returns:
 coords: (n_nodes, 2) array of node coordinates
 triangles: list of triples (i,j,k) indexing nodes
 Periodic wrapping is handled in indexing.
 """
 xs = np.linspace(0, 1, N, endpoint=False)
 coords = []
 for i in range(N):
 for j in range(N):
 coords.append((xs[j], xs[i]))
 coords = np.array(coords)
 def node(i, j):
 return ((i % N) * N + (j % N))
 tris = []
 for i in range(N):
 for j in range(N):
 n00 = node(i, j)
 n10 = node(i+1, j)
 n01 = node(i, j+1)
 n11 = node(i+1, j+1)

 tris.append((n00, n10, n11))
 tris.append((n00, n11, n01))
 return coords, tris





def triangle_area(v0, v1, v2):
 return 0.5 * abs((v1[0]-v0[0])*(v2[1]-v0[1]) - (v2[0]-v0[0])*(v1[1]-v0[1]))

def assemble_node_mass_matrix(N, coords, tris):
 n = coords.shape[0]
 rows = []
 cols = []
 data = []
 for tri in tris:
 i,j,k = tri
 v0, v1, v2 = coords[i], coords[j], coords[k]
 A = triangle_area(v0,v1,v2)
 loc = np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float) * (A / 12.0)
 ids = [i,j,k]
 for a in range(3):
 for b in range(3):
 rows.append(ids[a])
 cols.append(ids[b])
 data.append(loc[a,b])
 M = sp.coo_matrix((data, (rows, cols)), shape=(n,n)).tocsr()
 return M







def build_vector_basis(coords, k_modes=3):
 n = coords.shape[0]
 def node_field(fx, fy):

 fxv = fx(coords[:,0], coords[:,1])
 fyv = fy(coords[:,0], coords[:,1])
 return np.concatenate([fxv, fyv])
 basis = []

 basis.append(node_field(lambda x,y: np.ones_like(x), lambda x,y: np.zeros_like(x)))
 basis.append(node_field(lambda x,y: np.zeros_like(x), lambda x,y: np.ones_like(x)))

 for k in range(1, k_modes+1):
 basis.append(node_field(lambda x,y,kk=k: np.cos(2*np.pi*kk*x), lambda x,y,kk=k: np.zeros_like(x)))
 basis.append(node_field(lambda x,y,kk=k: np.zeros_like(x), lambda x,y,kk=k: np.cos(2*np.pi*kk*y)))
 basis.append(node_field(lambda x,y,kk=k: np.sin(2*np.pi*kk*x), lambda x,y,kk=k: np.sin(2*np.pi*kk*y)))
 B = np.column_stack(basis)
 return B




def block_mass_for_vector_field(M_node):
 n = M_node.shape[0]
 zero = sp.csr_matrix((n,n))
 top = sp.hstack([M_node, zero])
 bot = sp.hstack([zero, M_node])
 M_block = sp.vstack([top, bot]).tocsr()
 return M_block




def projector_from_basis(B, M_block):




 nbasis = B.shape[1]
 MB = M_block.dot(B)
 Gram = B.T.dot(MB)

 Gram_inv = np.linalg.inv(Gram)

 P_left = B.dot(Gram_inv)
 P = P_left.dot(MB.T)
 return P

def spectral_norm(mat):

 s = np.linalg.svd(mat, compute_uv=False)
 return float(s[0])




def run_for_N(N, modes=3, rationalize=False, max_den=10**12):
 coords, tris = make_periodic_grid_triangles(N)
 M_node = assemble_node_mass_matrix(N, coords, tris)
 M_block = block_mass_for_vector_field(M_node)
 B = build_vector_basis(coords, k_modes=modes)
 P = projector_from_basis(B, M_block)

 P2_minus_P = P.dot(P) - P
 err = spectral_norm(P2_minus_P)
 if rationalize:
 frac = Fraction(err).limit_denominator(int(max_den))
 return err, frac
 return err, None

def main():
 parser = argparse.ArgumentParser()
 parser.add_argument("--Nlist", type=int, nargs="+", default=[8,12,16,20], help="mesh N sizes")
 parser.add_argument("--modes", type=int, default=3, help="number of low-frequency modes")
 parser.add_argument("--rationalize", action="store_true", help="rationalize residuals for Lean")
 parser.add_argument("--max-den", type=float, default=1e12, help="max denominator for rationalization")
 args = parser.parse_args()

 print("Model: FEM P1 nodal basis on periodic torus; projector computed via L2 mass matrix.")
 print("N | ndof | nbasis | ||P^2 - P||_2 ", end="")
 if args.rationalize:
 print("| rationalized")
 else:
 print("")
 for N in args.Nlist:
 err, frac = run_for_N(N, modes=args.modes, rationalize=args.rationalize, max_den=args.max_den)
 ndof = 2 * (N*N)
 nbasis = 2 + 3*args.modes*1
 if args.rationalize:
 print(f"{N:3d} | {ndof:4d} | {nbasis:6d} | {err:.12e} | {frac.numerator}/{frac.denominator}")
 else:
 print(f"{N:3d} | {ndof:4d} | {nbasis:6d} | {err:.12e}")

if __name__ == '__main__':
 main()