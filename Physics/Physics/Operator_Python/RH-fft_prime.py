#!/usr/bin/env python3
"""
fft_prime_power_compare.py — compare raw FFT powers at prime frequencies
across multiple modes (true, shuffled, constant, noisy).
"""

import numpy as np, math
from numpy.linalg import lstsq

def primes_upto(n: int):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]: sieve[p*p::p] = False
 return np.nonzero(sieve)[0].tolist()

def fft_prime_powers(lams, Pmax=101, grid_pts=8192):
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

 prime_powers = {}
 for p in primes_upto(Pmax):
 f_target = math.log(p)/(2*math.pi)
 j = np.argmin(np.abs(freqs - f_target))
 prime_powers[p] = power[j]
 return prime_powers

def print_comparison(results):
 primes = sorted(set().union(*[r.keys() for r in results.values()]))
 print("Prime\t" + "\t".join(results.keys()))
 for p in primes:
 line = [str(p)]
 for mode in results:
 line.append(f"{results[mode].get(p,0):.2f}")
 print("\t".join(line))

def main():
 import sys
 path = sys.argv[1] if len(sys.argv)>1 else "eigs_1d/eigs_merged.npy"
 orig = np.load(path)
 orig = np.sort(orig[np.isfinite(orig)])


 shuf = orig.copy(); np.random.shuffle(shuf)
 fake = np.linspace(orig[0], orig[-1], orig.size)
 noisy = orig + np.random.normal(scale=np.std(orig)*0.05, size=orig.size)

 results = {
 "true": fft_prime_powers(orig),
 "shuffled": fft_prime_powers(shuf),
 "constant": fft_prime_powers(fake),
 "noisy": fft_prime_powers(noisy)
 }

 print_comparison(results)

if __name__ == "__main__":
 main()