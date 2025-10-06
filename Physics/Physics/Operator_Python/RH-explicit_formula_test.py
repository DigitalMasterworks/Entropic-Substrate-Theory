#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math

def primes_upto(n):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]:
 sieve[p*p::p] = False
 return np.nonzero(sieve)[0].tolist()

def analyze_eigs(path="eigs_merged.npy", Pmax=101, grid_pts=8192):

 lam = np.load(path)
 lam = np.sort(lam[np.isfinite(lam)])
 Ntot = len(lam)
 print(f"[info] Loaded {Ntot} eigenvalues")


 N_of_lam = np.arange(1, Ntot+1, dtype=float)


 root = np.sqrt(lam)
 m0 = int(0.8*Ntot)
 X = np.vstack([lam[m0:], root[m0:], np.ones_like(lam[m0:])]).T
 y = N_of_lam[m0:] - 0.5*lam[m0:]*np.log(lam[m0:])
 b, c, _ = np.linalg.lstsq(X, y, rcond=None)[0]
 mean = 0.5*lam*np.log(lam) + b*root + c

 osc = N_of_lam - mean


 t = root
 t_uniform = np.linspace(t[1], t[-1], grid_pts)
 osc_uniform = np.interp(t_uniform, t, osc)


 F = np.fft.rfft(osc_uniform - np.mean(osc_uniform))
 freqs = np.fft.rfftfreq(grid_pts, d=(t_uniform[1]-t_uniform[0]))
 power = np.abs(F)


 P = power[1:]
 med = np.median(P)
 mad = np.median(np.abs(P - med)) + 1e-18
 thresh = med + 6.0*mad


 prime_hits = []
 ok_primes = True
 for p in primes_upto(Pmax):
 f_target = math.log(p)/(2*math.pi)
 j = np.argmin(np.abs(freqs - f_target))
 pw = power[j]
 prime_hits.append((p, freqs[j], pw))
 if pw < thresh:
 ok_primes = False

 print(f"[check] Threshold = {thresh:.3e}")
 for p,f,pw in prime_hits:
 flag = "" if pw >= thresh else ""
 print(f" prime {p:3d} freq={f:.4f} power={pw:.2e} {flag}")

 if ok_primes:
 print(f"\n[RESULT] PASS: All primes ≤ {Pmax} detected")
 else:
 print(f"\n[RESULT] FAIL: Missing some primes ≤ {Pmax}")


 plt.figure(figsize=(10,6))
 plt.plot(freqs, power, lw=1)
 for p,f,pw in prime_hits:
 if pw >= thresh:
 plt.axvline(f, color='r', ls='--', alpha=0.5)
 plt.axhline(thresh, color='k', ls=':')
 plt.xlabel("frequency (cycles per t)")
 plt.ylabel("power")
 plt.title("FFT of oscillatory remainder (prime peaks marked)")
 plt.show()

if __name__ == "__main__":
 analyze_eigs("eigs_merged.npy", Pmax=101, grid_pts=8192)