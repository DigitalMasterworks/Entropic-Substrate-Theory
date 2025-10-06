
import numpy as np, sys
src = sys.argv[1]; dst = sys.argv[2]
e = np.load(src)
mask = np.isfinite(e)
e = np.sort(np.real(e[mask]))
print(f"[sanitize] kept {e.size}/{mask.size}; min={e[0]:.6g}, max={e[-1]:.6g}")
np.save(dst, e)