#!/usr/bin/env python3










from fractions import Fraction
import sys, re
import numpy as np

def parse_lean_array_text(text):
 m = re.search(r'Array\.mk\s*\[\s*(.*)\s*\]\s*$', text, re.DOTALL)
 inner = m.group(1) if m else text
 rows = re.findall(r'\[([^\]]*)\]', inner)
 M = []
 for r in rows:
 entries = [e.strip() for e in r.split(',') if e.strip()!='']
 row = []
 for ent in entries:
 fm = re.match(r'(-?\d+)\s*/\s*(\d+)', ent)
 if fm:
 row.append(Fraction(int(fm.group(1)), int(fm.group(2))))
 else:
 try:
 f = float(ent)
 row.append(Fraction(f).limit_denominator(10**12))
 except:
 raise ValueError("Can't parse entry: "+ent)
 M.append(row)
 return M

def bareiss_det(A):
 n = len(A)
 if n==0:
 return Fraction(1,1)
 M = [row[:] for row in A]
 denom = Fraction(1,1)
 for k in range(n-1):
 if M[k][k] == 0:
 pivot = None
 for i in range(k+1,n):
 if M[i][k]!= 0:
 pivot = i; break
 if pivot is None:
 return Fraction(0,1)
 M[k], M[pivot] = M[pivot], M[k]
 denom = -denom
 for i in range(k+1,n):
 for j in range(k+1,n):
 M[i][j] = (M[i][j]*M[k][k] - M[i][k]*M[k][j]) / denom
 M[i][k] = Fraction(0,1)
 denom = M[k][k]
 return M[-1][-1]

def fraction_rank_and_pivots(A):

 n = len(A)
 m = len(A[0])
 M = [row[:] for row in A]
 pivot_cols = []
 r = 0
 for c in range(m):

 piv = None
 for i in range(r, n):
 if M[i][c]!= 0:
 piv = i; break
 if piv is None:
 continue

 M[r], M[piv] = M[piv], M[r]

 pivval = M[r][c]
 M[r] = [val / pivval for val in M[r]]

 for i in range(n):
 if i==r: continue
 fac = M[i][c]
 if fac!= 0:
 M[i] = [M[i][j] - fac*M[r][j] for j in range(m)]
 pivot_cols.append(c)
 r += 1
 if r==n:
 break
 return r, pivot_cols

def nullspace_basis(A):

 n = len(A)
 m = len(A[0])


 M = [row[:] for row in A]
 pivot_cols = []
 r = 0
 pivrows = {}
 for c in range(m):
 piv = None
 for i in range(r, n):
 if M[i][c]!= 0:
 piv = i; break
 if piv is None:
 continue
 M[r], M[piv] = M[piv], M[r]
 pivval = M[r][c]
 M[r] = [val / pivval for val in M[r]]
 for i in range(n):
 if i==r: continue
 fac = M[i][c]
 if fac!= 0:
 M[i] = [M[i][j] - fac*M[r][j] for j in range(m)]
 pivot_cols.append(c)
 pivrows[c]=r
 r += 1
 if r==n:
 break
 free_cols = [c for c in range(m) if c not in pivot_cols]
 basis = []
 for free in free_cols:
 vec = [Fraction(0,1) for _ in range(m)]
 vec[free] = Fraction(1,1)

 for c, rowi in pivrows.items():
 vec[c] = -M[rowi][free]
 basis.append(vec)
 return basis, pivot_cols

def float_eigen_min_nonzero(A, tol=1e-12):

 import numpy as _np
 n = len(A)
 Af = _np.array([[float(x.numerator)/float(x.denominator) for x in row] for row in A], dtype=float)
 w = _np.linalg.eigvalsh(Af)
 wpos = [x for x in w if x > tol]
 if len(wpos)==0:
 return None, sorted(w)
 return float(min(wpos)), sorted(w)

def rationalize_float(x, max_den=10**12):
 from fractions import Fraction
 return Fraction(x).limit_denominator(int(max_den))

def main(argv):
 if len(argv)<2:
 print("Usage: python3 analyze_h_wilson_n2.py H_wilson_n2_lean.txt")
 return
 fname = argv[1]
 text = open(fname).read()
 A = parse_lean_array_text(text)
 n = len(A)
 print(f"Parsed {n}x{len(A[0])} rational matrix.")

 rank, pivots = fraction_rank_and_pivots(A)
 print(f"Exact rank (fraction-Gauss): {rank}")
 print(f"Pivot columns (0-based): {pivots}")

 basis, pivot_cols = nullspace_basis(A)
 print(f"Nullspace dimension: {len(basis)}")
 for i, v in enumerate(basis):

 nz = [(j, int(v[j].numerator), int(v[j].denominator)) for j in range(len(v)) if v[j]!= 0]
 print(f" nullvec {i}: nonzeros {nz}")

 if len(pivots) >= rank and rank>0:
 idx = pivots[:rank]
 print("Using pivot columns as principal index set:", idx)

 sub = [[A[i][j] for j in idx] for i in idx]
 detsub = bareiss_det(sub)
 print(f"det(H[idx,idx]) = {detsub.numerator}/{detsub.denominator}; ≈ {float(detsub):.12e}")

 print("Leading principal minors (k: det):")
 for k in range(1,n+1):
 sub = [row[:k] for row in A[:k]]
 detk = bareiss_det(sub)
 print(f"{k:2d}: {detk.numerator}/{detk.denominator}; ≈ {float(detk):.12e}")

 minpos, allw = float_eigen_min_nonzero(A, tol=1e-10)
 if minpos is None:
 print("No strictly positive eigenvalue above tol found in float eval; eigenvalues:", allw)
 else:
 print(f"Smallest positive eigenvalue (float) ≈ {minpos:.12e}")
 rat = rationalize_float(minpos, max_den=10**12)
 print(f"Rationalized smallest positive eigenvalue ≈ {rat.numerator}/{rat.denominator}; ≈ {float(rat):.12e}")

if __name__ == "__main__":
 main(sys.argv)