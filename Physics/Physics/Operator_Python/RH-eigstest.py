#!/usr/bin/env python3
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

eigs = np.load("eigs_1d/eigs_merged.npy")
lams = np.sort(eigs[np.isfinite(eigs) & (eigs > 0)])

def fit_cutoff(min_lambda):
 mask = lams >= min_lambda
 L = lams[mask]
 if len(L) < 10:
 return None
 N = np.arange(1, len(lams)+1)[mask]

 X = np.vstack([np.log(L), np.ones_like(L)]).T
 y = N / L
 coeffs, *_ = lstsq(X, y, rcond=None)
 alpha, beta = coeffs
 pred = X.dot(coeffs)
 resid = y - pred
 ss_res = np.sum(resid**2)
 ss_tot = np.sum((y - np.mean(y))**2)
 r2 = 1 - ss_res/ss_tot if ss_tot>0 else float('nan')
 return dict(min_lambda=min_lambda, n=len(L), alpha=alpha, beta=beta, r2=r2, Lmin=L[0], Lmax=L[-1])

cutoffs = [10,20,50,100,200,500,1000,5000]
results = []
for c in cutoffs:
 r = fit_cutoff(c)
 if r is not None:
 results.append(r)
 print(f"cutoff {c:6}: n={r['n']:6}, alpha={r['alpha']:.6g}, beta={r['beta']:.6g}, R2={r['r2']:.6g}")


cs = [r['min_lambda'] for r in results]
alphas = [r['alpha'] for r in results]
plt.figure()
plt.plot(cs, alphas, ".-")
plt.xlabel("min_lambda (cutoff)")
plt.ylabel("alpha (fit)")
plt.xscale('log')
plt.grid(True)
plt.savefig("alpha_vs_cutoff.png")
print("Saved alpha_vs_cutoff.png")


gaps = np.diff(lams)
print("eigs count:", len(lams))
print("min, median, max eigenvalue:", lams[0], np.median(lams), lams[-1])
print("min gap, median gap:", gaps.min(), np.median(gaps))

plt.figure(figsize=(6,3))
plt.hist(gaps[gaps>0], bins=200)
plt.yscale('log')
plt.title("gap histogram (log scale)")
plt.xlabel("gap")
plt.tight_layout()
plt.savefig("gap_hist.png")
print("Saved gap_hist.png")