#!/usr/bin/env python3





from fractions import Fraction
import re, sys

NULL_VECS = [


 {0:1, 6:1},
 {1:1, 7:1},
 {2:1, 8:1},
 {0:-1,3:-1,9:1,12:1},
 {1:-1,4:-1,10:1,13:1},
 {2:-1,5:-1,11:1,14:1},
 {3:1,15:1},
 {4:1,16:1},
 {5:1,17:1},
 {0:1,3:1,9:-1,18:1},
 {1:1,4:1,10:-1,19:1},
 {2:1,5:1,11:-1,20:1},
 {9:1,21:1},
 {10:1,22:1},
 {11:1,23:1},
]

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

def mat_vec_mul(M, v):
 n = len(M)
 res = [Fraction(0,1) for _ in range(n)]
 for i in range(n):
 s = Fraction(0,1)
 for j in range(len(v)):
 s += M[i][j] * v[j]
 res[i] = s
 return res

if __name__=="__main__":
 if len(sys.argv) < 2:
 print("Usage: python3 verify_nullvecs.py H_wilson_n2_lean.txt")
 sys.exit(1)
 text = open(sys.argv[1]).read()
 H = parse_lean_array_text(text)
 n = len(H)
 print(f"Parsed {n}x{n} matrix. Verifying {len(NULL_VECS)} null vectors.")
 for idx, sparse in enumerate(NULL_VECS):
 v = [Fraction(0,1) for _ in range(n)]
 for k,val in sparse.items():
 v[k] = Fraction(val,1)
 prod = mat_vec_mul(H, v)
 nz = [(i, p.numerator, p.denominator) for i,p in enumerate(prod) if p!= 0]
 if nz:
 print(f"nullvec {idx}: H*v nonzeros: {nz}")
 else:
 print(f"nullvec {idx}: H*v == 0 (exact)")