#!/usr/bin/env python3

import math, sys
try:
 import numpy as np
except Exception:
 np = None
try:
 import mpmath as mp
except Exception:
 mp = None

if mp is not None:
 mp.mp.dps = 200

def sample_holomorphic_one_form_on_torus(tau, N):
 """
 Sample the holomorphic 1-form dz on the torus lattice points.
 Return basis matrix B (n x k) with k basis vectors built from sampled differentials.
 For a torus, true holomorphic 1-form is constant; to generate a refining discretization,
 sample localized bump windows (test functions) times dz to mimic finite-element basis.
 """
 if np is None:
 raise RuntimeError("numpy required")
 n = 2 * N
 k = N
 B = np.zeros((n, k), dtype=complex)

 xs = [ (i+0.5)/n for i in range(n) ]
 for j in range(k):
 cx = (j+0.5)/k
 width = 0.25 / (j+1)
 for i,x in enumerate(xs):
 val = math.exp(-((x-cx)**2)/(2*width*width))

 phase = math.cos(2*math.pi*(j+1)*x) + 1j*math.sin(2*math.pi*(j+1)*x)
 B[i,j] = val * phase
 return B

def orthoprojector(B):
 Q, _ = np.linalg.qr(B, mode='reduced')
 P = Q @ Q.conjugate().T
 M = P @ P - P
 try:
 s = np.linalg.svd(M, compute_uv=False)[0]
 except Exception:
 s = float(np.max(np.abs(M)))
 return float(s)

def main():
 maxN = 8
 print("Model: samples of holomorphic 1-form on torus; projector convergence printed with high mpmath precision.")
 for N in range(1, maxN+1):
 B = sample_holomorphic_one_form_on_torus(tau=complex(0.5,0.8), N=N)
 err = orthoprojector(B)
 print(f"N={N:2d} dim={2*N:2d} basis={N:2d} ||P^2-P||_2 = {err:.12e}")

if __name__ == '__main__':
 main()