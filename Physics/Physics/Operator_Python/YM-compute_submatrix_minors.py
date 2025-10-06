#!/usr/bin/env python3





from fractions import Fraction
import re, sys

IDX = [0,1,2,3,4,5,9,10,11]

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

if __name__ == "__main__":
 if len(sys.argv) < 2:
 print("Usage: python3 compute_submatrix_minors.py H_wilson_n2_lean.txt")
 sys.exit(1)
 text = open(sys.argv[1]).read()
 H = parse_lean_array_text(text)

 Hsub = [[ H[i][j] for j in IDX ] for i in IDX ]
 n = len(Hsub)
 print(f"Principal index set IDX = {IDX}")
 print(f"Submatrix size: {n}x{n}")
 print("Leading principal minors of H_sub (k: numerator/denominator; decimal approx):")
 for k in range(1, n+1):
 sub = [row[:k] for row in Hsub[:k]]
 detk = bareiss_det(sub)
 print(f"{k:2d}: {detk.numerator}/{detk.denominator}; ≈ {float(detk):.12e}")