import numpy as np

lam = np.load("eigs_1d/eigs_merged.npy")
lam = np.sort(lam[lam > 0])


windows = [
 (1.5, 6.0),
 (2.0, 8.0),
 (1.0, 4.0),
 (3.0, 9.0),
]

L = 100.0
lam_max = lam.max()

for (c1, c2) in windows:
 t = np.geomspace(c1/lam_max, c2/lam_max, 40)
 T = np.array([np.exp(-ti * lam).sum() for ti in t])


 X = np.column_stack([1.0/t, np.log(1.0/t), np.ones_like(t), t])
 coef, *_ = np.linalg.lstsq(X, T, rcond=None)
 A, B, C, D = coef
 B_norm = B / (L**2)

 print(f"\nWindow c1={c1}, c2={c2}")
 print(f" A (1/t term) = {A:.6g}")
 print(f" B (log term) = {B:.6g}")
 print(f" C (const) = {C:.6g}")
 print(f" D (linear) = {D:.6g}")
 print(f" B normalized = {B_norm:.6g} (compare to theory 0.25)")