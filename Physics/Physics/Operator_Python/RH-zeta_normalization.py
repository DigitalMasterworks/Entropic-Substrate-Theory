

import numpy as np

def zeta_partial(lams, s):
 lp = lams[lams>1e-12]
 return np.sum(lp**(-s))

def fit_Z_near_one(lams, s0=1.0, delta=1e-3):
 svals = np.array([s0-2*delta, s0-delta, s0, s0+delta, s0+2*delta], dtype=float)
 Z = []
 for s in svals:
 Z.append(((s-1.0)**2) * zeta_partial(lams, s))
 Z = np.array(Z, dtype=float)

 x = svals - 1.0
 A = np.vstack([x, np.ones_like(x)]).T
 C1, C0 = np.linalg.lstsq(A, np.log(np.maximum(Z,1e-300)), rcond=None)[0]
 return C0, C1, svals, Z

if __name__ == "__main__":
 lams = np.load("eigs_merged.npy")
 lams = np.sort(lams[np.isfinite(lams)])
 C0,C1,svals,Z = fit_Z_near_one(lams, delta=1e-3)
 print(f"[Z] log Z(s) ~ C0 + C1 (s-1): C0={C0:.6e}, C1={C1:.6e}")
 for s,z in zip(svals, Z):
 print(f" s={s:.6f} Z(s)={(z):.6e}")