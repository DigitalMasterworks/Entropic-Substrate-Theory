#!/usr/bin/env python3






from fractions import Fraction
from pathlib import Path
import json


R_MAX = 30

nmodes = (2*R_MAX + 1)**2
dim = nmodes * 4





modes = []
for nx in range(-R_MAX, R_MAX+1):
 for ny in range(-R_MAX, R_MAX+1):
 k2 = nx*nx + ny*ny
 modes.append(((nx,ny), k2))

modes_sorted = sorted(modes, key=lambda t: (t[1], t[0]))

k0_idx = None
for i, ((nx,ny),k2) in enumerate(modes_sorted):
 if nx == 0 and ny == 0:
 k0_idx = i
 break
if k0_idx is None:
 raise RuntimeError("zero mode not found; unexpected.")

base_off = 4 * k0_idx



def zero_column(index):
 return {index: Fraction(1,1)}

Qz_cols = []
Qb_cols = []

for j in range(4):
 Qz_cols.append(zero_column(base_off + j))

for j in range(4):
 Qb_cols.append(zero_column(base_off + j))


def dot_sparse(a, b):

 if len(a) < len(b):
 small, large = a, b
 else:
 small, large = b, a
 s = Fraction(0,1)
 for k,v in small.items():
 if k in large:
 s += v * large[k]
 return s

M = [[dot_sparse(Qz_cols[i], Qb_cols[j]) for j in range(4)] for i in range(4)]


is_identity = True
for i in range(4):
 for j in range(4):
 expected = Fraction(1,1) if i==j else Fraction(0,1)
 if M[i][j]!= expected:
 is_identity = False

report = {
 "R_MAX": R_MAX,
 "nmodes": nmodes,
 "dim": dim,
 "k0_idx": k0_idx,
 "M_matrix": [[str(x) for x in row] for row in M],
 "is_identity": is_identity
}

outp = Path("torus_projector_exact_report.json")
outp.write_text(json.dumps(report, indent=2))
print("Wrote", outp.resolve())
print("is_identity:", is_identity)