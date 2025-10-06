#!/usr/bin/env python3
"""
fft_prime_bootstrap_ratio.py — bootstrap comparison of prime FFT powers
using *relative prominence* (prime-to-background ratio).
"""

import numpy as np, math, argparse
from numpy.linalg import lstsq

def primes_upto(n: int):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]:
 sieve[p*p::p] = False
 return np.nonzero(sieve)[0].tolist()

def fft_prime_ratios(lams, Pmax=101, grid_pts=8192, window=20):
 lam = np.sort(lams[lams > 1e-12])
 N_of_lam = np.arange(1, len(lam)+1, dtype=float)
 T = np.sqrt(lam)


 X = np.vstack([T*np.log(T), T, np.ones_like(T)]).T
 coef, *_ = lstsq(X, N_of_lam, rcond=None)
 pred = X @ coef
 osc = N_of_lam - pred


 t_uniform = np.linspace(T[0], T[-1], grid_pts)
 osc_uniform = np.interp(t_uniform, T, osc)

 F = np.fft.rfft(osc_uniform - np.mean(osc_uniform))
 freqs = np.fft.rfftfreq(grid_pts, d=(t_uniform[1]-t_uniform[0]))
 power = np.abs(F)

 ratios = {}
 for p in primes_upto(Pmax):
 f_target = math.log(p)/(2*math.pi)
 j = np.argmin(np.abs(freqs - f_target))
 local = np.r_[power[max(1,j-window):j], power[j+1:j+window+1]]
 bg = np.median(local) if local.size else 1.0
 ratios[p] = power[j]/(bg+1e-12)
 return ratios

def bootstrap_nulls(orig, mode="shuffle", trials=50, Pmax=101, grid_pts=8192):
 """Generate null distributions for prime prominence ratios."""
 N = len(orig)
 ratios = {p: [] for p in primes_upto(Pmax)}
 for _ in range(trials):
 if mode == "shuffle":
 null = orig.copy(); np.random.shuffle(null)
 elif mode == "constant":
 null = np.linspace(orig[0], orig[-1], N)
 elif mode == "noisy":
 null = orig + np.random.normal(scale=np.std(orig)*0.05, size=N)
 else:
 raise ValueError("unknown null mode")
 r = fft_prime_ratios(null, Pmax=Pmax, grid_pts=grid_pts)
 for p,v in r.items():
 ratios[p].append(v)
 return {p: (np.mean(v), np.std(v)) for p,v in ratios.items()}

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("path", nargs="?", default="eigs_1d/eigs_merged.npy",
 help="path to eigenvalue file (default eigs_1d/eigs_merged.npy)")
 ap.add_argument("--Pmax", type=int, default=101,
 help="maximum prime to check (default 101)")
 ap.add_argument("--grid", type=int, default=8192,
 help="FFT grid points (default 8192, try 32768 for high res)")
 ap.add_argument("--trials", type=int, default=50,
 help="bootstrap trials (default 50)")
 args = ap.parse_args()

 orig = np.load(args.path)
 orig = np.sort(orig[np.isfinite(orig)])

 true_ratios = fft_prime_ratios(orig, Pmax=args.Pmax, grid_pts=args.grid)
 shuffle_stats = bootstrap_nulls(orig, "shuffle", trials=args.trials,
 Pmax=args.Pmax, grid_pts=args.grid)
 noisy_stats = bootstrap_nulls(orig, "noisy", trials=args.trials,
 Pmax=args.Pmax, grid_pts=args.grid)

 print("Prime\tTrueRatio\tShuffle μ±σ\tZ-score(shuffle)\tNoisy μ±σ\tZ-score(noisy)")
 for p in sorted(true_ratios.keys()):
 true_val = true_ratios[p]
 mu_s, sd_s = shuffle_stats[p]
 mu_n, sd_n = noisy_stats[p]
 z_s = (true_val - mu_s)/(sd_s+1e-9)
 z_n = (true_val - mu_n)/(sd_n+1e-9)
 print(f"{p}\t{true_val:.2f}\t{mu_s:.2f}±{sd_s:.2f}\t{z_s:.2f}\t"
 f"{mu_n:.2f}±{sd_n:.2f}\t{z_n:.2f}")

if __name__ == "__main__":
 main()