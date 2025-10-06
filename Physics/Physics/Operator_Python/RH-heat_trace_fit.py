
import numpy as np, argparse
from numpy.linalg import lstsq

p = argparse.ArgumentParser()
p.add_argument("--eigs", default="eigs_1d/eigs_merged.npy")
p.add_argument("--tmin", type=float, nargs="+", default=[1e-5,3e-5,1e-4,3e-4])
p.add_argument("--tmax", type=float, nargs="+", default=[5e-4,1e-3,3e-3,1e-2])
p.add_argument("--n", type=int, default=25)
p.add_argument("--trim_lo", type=int, default=0)
p.add_argument("--trim_hi", type=int, default=0)
args = p.parse_args()

eigs = np.load(args.eigs).astype(float)
eigs = np.sort(eigs[(eigs>0)])
if args.trim_lo: eigs = eigs[args.trim_lo:]
if args.trim_hi: eigs = eigs[:-args.trim_hi]
print(f"[load] {args.eigs} count={len(eigs)}")

def fit_window(tmin, tmax, n):
 t = np.logspace(np.log10(tmin), np.log10(tmax), n)
 T = np.array([np.exp(-ti*eigs).sum(dtype=float) for ti in t])

 X = np.column_stack([np.log(1/t)/t, 1.0/t, 1/np.sqrt(t), np.ones_like(t)])
 a,b,d,c = lstsq(X, T, rcond=None)[0]
 rel_err = np.linalg.norm(T - X @ np.array([a,b,d,c]))/np.linalg.norm(T)
 return a,b,d,c,rel_err

for tmin,tmax in zip(args.tmin, args.tmax):
 a,b,d,c,r = fit_window(tmin,tmax,args.n)
 print(f"t∈[{tmin:.1e},{tmax:.1e}] a={a:.6g} b={b:.6g} d={d:.6g} c={c:.6g} rel_err={r:.3g}")