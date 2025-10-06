#!/usr/bin/env python3
"""
fft_null_test.py — run FFT prime-checker on shuffled eigenvalues (null control).
"""

import numpy as np, math
from numpy.linalg import lstsq

def primes_upto(n: int):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]: sieve[p*p::p] = False
 return np.nonzero(sieve)[0].tolist()

def fft_prime_check(lams, Pmax=101, grid_pts=8192, snr_sigma=6.0):
 lam = np.sort(lams[lams>1e-12])
 Ntot = len(lam)
 N_of_lam = np.arange(1, Ntot+1, dtype=float)


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

 P = power[1:]
 med = np.median(P)
 mad = np.median(np.abs(P - med)) + 1e-18
 thresh = med + snr_sigma*mad

 prime_hits = []
 plist = primes_upto(Pmax)
 for p in plist:
 f_target = math.log(p)/(2*math.pi)
 j = np.argmin(np.abs(freqs - f_target))
 pw = power[j]
 prime_hits.append((p, freqs[j], pw, pw>=thresh))

 keep = [p for (p,f,pw,ok) in prime_hits if ok]
 miss = [p for (p,f,pw,ok) in prime_hits if not ok]

 print(f"[fft-null] threshold = {thresh:.3e}")
 print(f"[fft-null] primes detected: {keep}")
 print(f"[fft-null] primes missed: {miss}")

if __name__ == "__main__":
 import sys
 path = sys.argv[1] if len(sys.argv)>1 else "eigs_1d/eigs_merged.npy"
 lams = np.load(path)
 np.random.shuffle(lams)
 fft_prime_check(lams)