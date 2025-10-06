
import numpy as np

eigs = np.load("eigs_1d/eigs_merged.npy")
lam = np.sort(eigs[eigs>0]).astype(float)


L0, L1 = np.quantile(lam, 0.10), np.quantile(lam, 0.90)
mask = (lam >= L0) & (lam <= L1)


x = np.linspace(L0, L1, 40000)
N = np.searchsorted(lam, x)


phi1 = x * np.log(x)
phi2 = x
phi3 = np.ones_like(x)


def gs_cols(*cols):
 Q = []
 for v in cols:
 w = v.copy()
 for q in Q:
 w -= (w @ q) / (q @ q) * q
 Q.append(w)
 return Q

q1, q2, q3 = gs_cols(phi1, phi2, phi3)


X = np.column_stack([q1, q2, q3])
theta = np.linalg.lstsq(X, N, rcond=None)[0]
a_ortho = theta[0] / (q1 @ q1) * (q1 @ phi1)

rng = np.random.default_rng(0)
boots = []
for _ in range(200):
 idx = rng.integers(0, len(x), size=len(x))
 Xb, Nb = X[idx], N[idx]
 tb = np.linalg.lstsq(Xb, Nb, rcond=None)[0]
 ab = tb[0] / (q1 @ q1) * (q1 @ phi1)
 boots.append(ab)
boots = np.array(boots)
print(f"a (orthogonalized) = {a_ortho:.6g} ± {boots.std():.3g} (L0={L0:.3g}, L1={L1:.3g})")