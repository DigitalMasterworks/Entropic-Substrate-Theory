#!/usr/bin/env python3








import numpy as np
import matplotlib.pyplot as plt


EIGS_FILE = "eigs_1d/eigs_merged.npy"
cutoff_list = np.geomspace(1e3, 1e5, 20)
s_shift = 0.5


lam = np.load(EIGS_FILE)
lam = np.sort(lam[lam > 0]).astype(float)

def logdet_surrogate(lam, cutoff, s=s_shift):
 """
 Determinant surrogate:
 log det(H - s(1-s)) ~ - d/ds [ ζ_H(s) ]
 Approximate by truncated product over λ_j <= cutoff^2
 with shift s.
 """
 mask = lam <= cutoff**2
 subset = lam[mask]
 if len(subset) == 0:
 return np.nan

 return np.sum(np.log(subset + s*(1-s))).real


vals = []
for R in cutoff_list:
 val = logdet_surrogate(lam, R)
 vals.append(val)
 print(f"cutoff={R:9.2f} logdet ≈ {val:.6g}")


plt.figure(figsize=(6,4))
plt.plot(cutoff_list, vals, "o-")
plt.xscale("log")
plt.xlabel("Cutoff radius R")
plt.ylabel("log determinant surrogate")
plt.title("Cutoff-independence of substrate determinant")
plt.tight_layout()
plt.savefig("determinant_cutoff.png", dpi=150)
print("Saved: determinant_cutoff.png")