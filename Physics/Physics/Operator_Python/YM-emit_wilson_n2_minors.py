#!/usr/bin/env python3




from fractions import Fraction
from pathlib import Path


IDX = [0,1,2,3,4,5,9,10,11]


M_STR_ROWS = [
 "1/1, 0/1, 0/1, -1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1",
 "0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1",
 "0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2",
 "-1/2, 0/1, 0/1, 1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1",
 "0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1",
 "0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1",
 "-1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1",
 "0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1",
 "0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2",
 "1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1",
 "0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1",
 "0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1",
 "0/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1",
 "0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1",
 "1/2, 0/1, 0/1, -1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1",
 "0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1",
 "0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1",
 "0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1, 0/1",
 "0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2, 0/1",
 "0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 1/1, 0/1, 0/1, -1/2",
 "-1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1, 0/1",
 "0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1, 0/1",
 "0/1, 0/1, -1/2, 0/1, 0/1, 0/1, 0/1, 0/1, 1/2, 0/1, 0/1, -1/1, 0/1, 0/1, 1/2, 0/1, 0/1, 0/1, 0/1, 0/1, -1/2, 0/1, 0/1, 1/1"
]


def parse_cell(s: str) -> Fraction:
 s = s.strip()
 if s == "":
 return Fraction(0,1)
 if "/" in s:
 a,b = s.split("/")
 return Fraction(int(a.strip()), int(b.strip()))
 return Fraction(int(s), 1)


M = []
for row in M_STR_ROWS:
 cells = [parse_cell(x) for x in row.split(",")]
 if len(cells)!= 24:
 raise ValueError(f"row length!= 24: got {len(cells)}")
 M.append(cells)


def subdet(M_full, idxk):
 n = len(idxk)

 A = [[M_full[i][j] for j in idxk] for i in idxk]
 det = Fraction(1,1)
 for c in range(n):

 pivot = None
 for r in range(c,n):
 if A[r][c]!= 0:
 pivot = r
 break
 if pivot is None:
 return Fraction(0,1)
 if pivot!= c:
 A[c], A[pivot] = A[pivot], A[c]
 det *= -1
 p = A[c][c]
 det *= p

 for j in range(c+1, n):
 A[c][j] = A[c][j] / p

 for r in range(c+1, n):
 factor = A[r][c]
 if factor == 0:
 continue
 for j in range(c+1, n):
 A[r][j] = A[r][j] - factor * A[c][j]
 A[r][c] = Fraction(0,1)
 return det


def rat_lit(fr: Fraction) -> str:

 if fr.denominator < 0:
 fr = Fraction(-fr.numerator, -fr.denominator)
 if fr.denominator == 1:
 return f"{fr.numerator}"
 return f"({fr.numerator}: Rat) / ({fr.denominator}: Rat)"


def pos_pf(fr: Fraction, name: str) -> str:
 if fr <= 0:

 return ""
 if fr.denominator == 1:
 return f"""theorem {name}_pos: 0 < {name}:= by
 norm_num
"""

 return f"""theorem {name}_pos: 0 < {name}:= by
 show 0 < ({fr.numerator}: Rat) / ({fr.denominator}: Rat)
 apply div_pos <;> norm_num
"""


TEMPLATE = """\
import Mathlib.Algebra.Ring.Rat
import Mathlib.Analysis.Matrix
import Mathlib.Tactic

namespace Myproject
namespace YangMillsReverse

/-- Principal index set used in analysis. -/
def H_wilson_n2_IDX: List Nat:= [{idxs}]

{blocks}

end YangMillsReverse
end Myproject
"""

def main():
 blocks = []
 for k in range(1, len(IDX)+1):
 idxk = IDX[:k]
 d = subdet(M, idxk)
 nm = f"det_{k}"
 lit = rat_lit(d)
 block = f"def {nm}: Rat:= {lit}\n" \
 f"theorem {nm}_def: {nm} = {lit}:= by rfl\n" \
 f"{pos_pf(d, nm)}"
 blocks.append(block)
 out = TEMPLATE.format(idxs=", ".join(map(str, IDX)), blocks="\n".join(blocks))
 outpath = Path("../YangMills/Wilson_n2_minors.lean")
 outpath.write_text(out)
 print(f"Wrote {outpath} with {len(blocks)} minor definitions.")

if __name__ == "__main__":
 main()