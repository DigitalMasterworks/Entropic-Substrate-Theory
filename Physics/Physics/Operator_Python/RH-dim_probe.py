
import numpy as np
from numpy.linalg import lstsq

eigs = np.load("eigs_1d/eigs_merged.npy")
eigs = np.sort(eigs[eigs>0])

lam = np.linspace(eigs[200], eigs[-200], 20000)
N = np.searchsorted(eigs, lam)

def fit_and_aic(X, y, k):
 theta, *_ = lstsq(X, y, rcond=None)
 res = y - X @ theta
 rss = (res**2).sum()
 n = len(y)
 aic = n*np.log(rss/n + 1e-300) + 2*k
 return aic, theta


X1 = np.column_stack([np.sqrt(lam), np.ones_like(lam)])
aic1, th1 = fit_and_aic(X1, N, 2)


X2 = np.column_stack([lam*np.log(lam), lam, np.ones_like(lam)])
aic2, th2 = fit_and_aic(X2, N, 3)

print(f"AIC_1D={aic1:.1f} AIC_2D={aic2:.1f}")
print(f"1D fit: N≈{th1[0]:.4g}√λ + {th1[1]:.4g}")
print(f"2D fit: N≈{th2[0]:.4g}λ log λ + {th2[1]:.4g}λ + {th2[2]:.4g}")
print("Winner:", "1D" if aic1<aic2 else "2D")