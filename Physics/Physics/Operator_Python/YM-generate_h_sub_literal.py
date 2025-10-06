#!/usr/bin/env python3




import sys, re
from pathlib import Path
from fractions import Fraction


IDX = [0,1,2,3,4,5,9,10,11]

def parse_matrix(text):

 text2 = text.replace('[', ' ').replace(']', ' ')
 rows = []
 for line in text2.splitlines():

 toks = re.findall(r'([+-]?\d+\s*/\s*\d+|[+-]?\d+)', line)
 if not toks:
 continue
 row = []
 for t in toks:
 if '/' in t:
 a, b = re.split(r'\s*/\s*', t)
 row.append(Fraction(int(a), int(b)))
 else:
 row.append(Fraction(int(t), 1))

 rows.append(row)
 if not rows:
 raise ValueError("No numeric rows parsed. Ensure input contains bracketed rows like [a, b, c] or plain numeric rows.")

 maxlen = max(len(r) for r in rows)
 rows = [r + [Fraction(0,1)]*(maxlen-len(r)) for r in rows]
 return rows

def emit_submatrix_lean(mat, idx):
 n = len(mat)
 k = len(idx)
 sub = []
 for i in idx:
 if i<0 or i>=n:
 raise IndexError(f"Index {i} out of range for matrix size {n}")
 row = [mat[i][j] for j in idx]
 sub.append(row)
 lines = []
 lines.append("-- AUTO-GENERATED: Myproject/YangMills_Wilson_n2_sub.lean")
 lines.append("import Mathlib.Data.Rat.Basic")
 lines.append("import Mathlib.LinearAlgebra.Matrix")
 lines.append("")
 lines.append("namespace Myproject")
 lines.append("namespace YangMills")
 lines.append("")
 lines.append("def H_wilson_n2_sub: Matrix (Fin 9) (Fin 9) Rat:=")
 lines.append(" Matrix.ofFn (λ (i: Fin 9) (j: Fin 9) =>")
 lines.append(" match i.val, j.val with")
 for i in range(9):
 for j in range(9):
 val = sub[i][j]
 lines.append(f" | {i}, {j} => (({val.numerator}: Rat) / ({val.denominator}: Rat))")
 lines.append(" | _, _ => 0)")
 lines.append("")
 lines.append("end YangMills")
 lines.append("end Myproject")
 return "\n".join(lines)

if __name__ == '__main__':
 if len(sys.argv) < 2:
 print("Usage: python3 generate_h_sub_literal.py H_wilson_n2_lean.txt", file=sys.stderr)
 sys.exit(1)
 p = Path(sys.argv[1])
 if not p.exists():
 print("File not found:", p, file=sys.stderr)
 sys.exit(2)
 text = p.read_text()
 mat = parse_matrix(text)
 out = emit_submatrix_lean(mat, IDX)
 out_path = Path("Myproject") / "YangMills_Wilson_n2_sub.lean"
 out_path.parent.mkdir(parents=True, exist_ok=True)
 out_path.write_text(out)
 print("Wrote", out_path)