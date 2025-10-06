#!/usr/bin/env python3


import argparse, sys, math
import numpy as np
from pathlib import Path
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def build_S_candidate(x, form="linear", alpha=1.0, beta=0.0, c=10.0):
 r = np.abs(x)
 if form == "linear":
 return r
 if form == "log":
 return r * np.log1p(r)
 if form == "exp":
 return np.exp(r/c)
 if form == "linear_exp":
 return r * np.exp(r/c)
 if form == "power_log":
 return (r**alpha) * (np.log1p(r)**beta)
 raise ValueError(f"unknown form={form}")


def build_H_1d(N, L, form="linear", alpha=1.0, beta=0.0, c=10.0, eps=1e-6):
 x = np.linspace(-L/2, L/2, N, dtype=np.float64)
 S = build_S_candidate(x, form=form, alpha=alpha, beta=beta, c=c)
 S2 = np.maximum(S, eps)**2
 h = L/(N-1)
 c_edges = 0.5*(S2[:-1] + S2[1:]) / (h*h)
 main = np.zeros(N)
 lower = np.zeros(N-1)
 upper = np.zeros(N-1)
 for i in range(N-1):
 w = c_edges[i]
 main[i] += w
 main[i+1] += w
 lower[i] -= w
 upper[i] -= w
 H = diags([main, lower, upper], [0,-1,1], shape=(N,N), format="csr")
 return H


def merge_unique(paths, out_path):
 buf = []
 for p in paths:
 try:
 e = np.load(p)
 e = np.real(e[np.isfinite(e)])
 if e.size:
 buf.append(e)
 except Exception:
 pass
 if not buf:
 return 0
 E = np.sort(np.concatenate(buf))
 tol = 1e-9
 uniq = [E[0]]
 for x in E[1:]:
 if abs(x - uniq[-1]) > tol:
 uniq.append(x)
 E = np.array(uniq, dtype=float)
 np.save(out_path, E)
 return E.size

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--N", type=int, required=True)
 ap.add_argument("--L", type=float, required=True)
 ap.add_argument("--k", type=int, default=64)
 ap.add_argument("--tol", type=float, default=1e-8)
 ap.add_argument("--outdir", default="eigs_1d")
 ap.add_argument("--min-unique", type=int, default=10000)
 ap.add_argument("--merge-every", type=int, default=25)
 ap.add_argument("--form", default="linear")
 ap.add_argument("--alpha", type=float, default=1.0)
 ap.add_argument("--beta", type=float, default=0.0)
 ap.add_argument("--c", type=float, default=10.0)
 ap.add_argument("--mu", required=True, help="comma list or a:b:step range")
 args = ap.parse_args()

 outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
 merged_path = outdir / "eigs_merged.npy"

 H = build_H_1d(args.N, args.L, form=args.form,
 alpha=args.alpha, beta=args.beta, c=args.c)


 mu_list = []
 if ":" in args.mu:
 a,b,cstep = [float(x) for x in args.mu.split(":")]
 mu_list = list(np.arange(a,b,cstep))
 else:
 mu_list = [float(x) for x in args.mu.split(",") if x.strip()]

 processed = 0
 newly_produced = 0

 for mu in mu_list:
 tag = f"{mu:.6f}".replace(".","p")
 fpath = outdir / f"eigs_mu_{tag}.npy"
 if fpath.exists():
 print(f"[skip] {fpath}")
 continue
 print(f"[win] mu={mu}:: computing {args.k} eigs")
 try:
 vals = eigsh(H, k=args.k, sigma=mu, which="LM", tol=args.tol,
 return_eigenvectors=False)
 vals = np.real(vals[np.isfinite(vals)])
 vals.sort()
 np.save(fpath, vals)
 print(f"[win] saved {fpath} ({vals.size})")
 newly_produced += 1
 except Exception as ex:
 print(f"[warn] mu={mu} failed: {ex}")
 processed += 1

 if newly_produced >= args.merge_every:
 count = merge_unique(list(sorted(outdir.glob("eigs_mu_*.npy"))), merged_path)
 print(f"[merge] unique={count} -> {merged_path}")
 newly_produced = 0
 if count >= args.min_unique:
 print(f"[done] reached min-unique={args.min_unique}")
 return


 count = merge_unique(list(sorted(outdir.glob("eigs_mu_*.npy"))), merged_path)
 print(f"[final] unique={count} -> {merged_path}")

if __name__ == "__main__":
 main()