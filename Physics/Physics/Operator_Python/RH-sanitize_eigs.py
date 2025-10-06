
import numpy as np, sys
src = sys.argv[1] if len(sys.argv)>1 else "eigs_k10000.npy"
dst = sys.argv[2] if len(sys.argv)>2 else src.replace(".npy","_clean.npy")

e = np.load(src)

e = np.real(e)
mask = np.isfinite(e) & (e >= -1e-12)
e = e[mask]
e[e<0] = 0.0
e = np.sort(e)
print(f"[sanitize] kept {e.size} eigenvalues (from {mask.size}); min={e.min():.6g}, max={e.max():.6g}")
np.save(dst, e)
print(f"[sanitize] saved -> {dst}")