#!/usr/bin/env python3
"""
print_leading_minors.py

Parse a Lean `Array.mk [ [...], [...],... ]` matrix literal from a file (or stdin),
convert entries of the form `num / den` to Fraction, compute exact leading principal
determinants det(H[:k,:k]) for k=1..n, and print them as reduced rationals plus decimals.

Usage:
 python3 print_leading_minors.py H_wilson_n2_lean.txt
Or:
 cat H_wilson_n2_lean.txt | python3 print_leading_minors.py
"""
from fractions import Fraction
import sys
import re
from typing import List

def parse_lean_array_text(text: str) -> List[List[Fraction]]:
 """
 Extract rows from a Lean Array.mk literal.
 Looks for `Array.mk [... ]` and then captures `[`...`],` row blocks.
 Each entry is expected as 'NUM / DEN' (with optional spaces).
 """

 m = re.search(r'Array\.mk\s*\[\s*(.*)\s*\]\s*$', text, re.DOTALL)
 if not m:

 inner = text
 else:
 inner = m.group(1)


 row_matches = re.findall(r'\[([^\]]*)\]', inner)
 if not row_matches:
 raise ValueError("No rows found in input. Ensure the file contains the Array.mk [... ] block.")

 matrix = []
 for rowtext in row_matches:

 entries = [e.strip() for e in rowtext.split(',') if e.strip()!= ""]
 row = []
 for ent in entries:

 ent = ent.replace("_", "").strip()
 frac_match = re.match(r'(-?\d+)\s*/\s*(\d+)', ent)
 if frac_match:
 num = int(frac_match.group(1))
 den = int(frac_match.group(2))
 row.append(Fraction(num, den))
 else:

 try:
 if '.' in ent or 'e' in ent.lower():
 f = float(ent)
 row.append(Fraction(f).limit_denominator(10**12))
 else:
 row.append(Fraction(int(ent), 1))
 except Exception as ex:
 raise ValueError(f"Couldn't parse matrix entry: '{ent}'") from ex
 matrix.append(row)

 widths = set(len(r) for r in matrix)
 if len(widths)!= 1:
 raise ValueError(f"Non-rectangular rows found; lengths = {widths}")
 return matrix

def det_fraction_matrix(mat: List[List[Fraction]]) -> Fraction:
 """Exact determinant via fraction Gaussian elimination (LU) with partial pivoting."""
 n = len(mat)

 A = [[Fraction(x) for x in row] for row in mat]
 det = Fraction(1,1)
 sign = 1
 for k in range(n):

 pivot_row = None
 for i in range(k, n):
 if A[i][k]!= 0:
 pivot_row = i
 break
 if pivot_row is None:
 return Fraction(0,1)
 if pivot_row!= k:

 A[k], A[pivot_row] = A[pivot_row], A[k]
 sign *= -1
 pivot = A[k][k]
 det *= pivot


 for i in range(k+1, n):
 if A[i][k] == 0:
 continue
 factor = A[i][k] / pivot

 for j in range(k, n):
 A[i][j] -= factor * A[k][j]
 return det * sign

def main(argv):
 if len(argv) >= 2:
 fname = argv[1]
 with open(fname, 'r') as f:
 text = f.read()
 else:

 text = sys.stdin.read()
 try:
 M = parse_lean_array_text(text)
 except Exception as e:
 print("Error parsing input:", e, file=sys.stderr)
 sys.exit(2)
 n = len(M)
 print(f"Parsed {n}x{len(M[0])} rational matrix.")
 if n!= len(M[0]):
 print("Warning: matrix is not square; leading principal minors defined only up to min(n,m).")
 kmax = n
 print("Leading principal minors (k: rational = numerator/denominator; decimal approx):")
 for k in range(1, kmax+1):
 sub = [row[:k] for row in M[:k]]
 detk = det_fraction_matrix(sub)

 if detk.denominator < 0:
 detk = Fraction(-detk.numerator, -detk.denominator)
 print(f"{k:2d}: {detk.numerator}/{detk.denominator}; ≈ {float(detk):.12e}")
 print("\nDone. Copy these rational determinants into Lean as `def det_k: Rat:= num / den` or as `theorem` = by rfl.")
 print("If you paste the output here I will produce the full Lean positivity proof (Sylvester) for H_wilson_n2.")

if __name__ == "__main__":
 main(sys.argv)