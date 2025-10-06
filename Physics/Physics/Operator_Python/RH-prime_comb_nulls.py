
import numpy as np

eigs = np.load("eigs_1d/eigs_merged.npy")
eigs = np.sort(eigs[eigs > 0])


lam = np.linspace(eigs[200], eigs[-200], 40000)
x = np.log(lam)
N = np.searchsorted(eigs, lam)

trend = np.poly1d(np.polyfit(lam, N, 2))(lam)
f = N - trend
f = (f - f.mean())


def amp_at_omega(omega, x, f):
 z = np.exp(-1j * omega * x)
 return np.vdot(z, f) / np.sqrt(len(x))


def primes_up_to(n):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2, int(n**0.5)+1):
 if sieve[p]: sieve[p*p::p] = False
 return np.flatnonzero(sieve)

pr = primes_up_to(200)
omegas = np.log(pr.astype(float))


A = np.array([abs(amp_at_omega(w, x, f)) for w in omegas])


rng = np.random.default_rng(0)
null_As = []
for _ in range(200):
 shuf = rng.permutation(f)
 null_As.append([abs(amp_at_omega(w, x, shuf)) for w in omegas])
null_As = np.array(null_As)
mu = null_As.mean(axis=0)
sd = null_As.std(axis=0) + 1e-12
z = (A - mu) / sd

print("top 10 primes by z-score:")
idx = np.argsort(z)[-10:][::-1]
for i in idx:
 print(f"p={pr[i]:3d} omega=log p={omegas[i]:.6f} z={z[i]:.2f} A={A[i]:.3g}")
print(f"\nmedian z over tested primes: {np.median(z):.2f}")