
from pathlib import Path

eigs = {
 2: "2",
 3: "2",
 4: "2",
}

TEMPLATE = """\
import Mathlib.Algebra.Ring.Rat

namespace Myproject
namespace YangMillsReverse

{body}

end YangMillsReverse
end Myproject
"""

def block(n, val):
 return f"""\
def lambda_min_{n}: Rat:= {val}
theorem lambda_min_{n}_def: lambda_min_{n} = {val}:= by rfl
"""

Path("../YangMills/Data.lean").write_text(
 TEMPLATE.format(body="\n".join(block(n, v) for n,v in eigs.items()))
)