
import numpy as np

eigs = np.load("eigs_1d/eigs_merged.npy")
eigs = np.sort(eigs[eigs>0])

lam = np.linspace(eigs[200], eigs[-200], 40000)
x = np.log(lam)
N = np.searchsorted(eigs, lam)
trend = np.poly1d(np.polyfit(lam, N, 2))(lam)
f = (N - trend) - (N - trend).mean()

def amp(omega):
 z = np.exp(-1j*omega*x)
 return np.vdot(z, f)/np.sqrt(len(x))

def primes_up_to(n):
 sieve = np.ones(n+1, dtype=bool); sieve[:2]=False
 for p in range(2,int(n**0.5)+1):
 if sieve[p]: sieve[p*p::p]=False
 return np.flatnonzero(sieve)

pr = primes_up_to(100)
omegas = np.log(pr.astype(float))
A = np.array([abs(amp(w)) for w in omegas])


jA = np.array([max(abs(amp(w*1.01)), abs(amp(w*0.99))) for w in omegas])
ratio = A/(jA+1e-12)
print("median A/jittered ratio:", np.median(ratio))
print("examples:", [(int(p), float(A[i]), float(ratio[i])) for i,p in list(enumerate(pr))[:10]])


pp = []
for p in pr:
 for k in (2,3):
 pp.append((p,k,np.log(p)*k, abs(amp(np.log(p)*k))))
pp.sort(key=lambda t: -t[3])
print("top prime powers (p^k, k=2,3) by amplitude:", pp[:10])