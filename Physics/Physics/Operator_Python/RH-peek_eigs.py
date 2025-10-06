
import numpy as np
e = np.load("eigs_k10000.npy")
print("total:", e.size)
print("finite:", np.isfinite(e).sum())
print("nan:", np.isnan(e).sum(), "inf:", np.isinf(e).sum())
print("min/max (finite only):", np.nanmin(e), np.nanmax(e))