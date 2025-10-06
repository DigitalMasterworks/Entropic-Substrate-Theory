#!/usr/bin/env python3



import argparse, numpy as np, math, sys

def load_tail(path, tail_count):
 lam = np.load(path)
 lam = np.sort(lam[np.isfinite(lam)])
 if lam.size < tail_count + 5:
 tail_count = max(5, lam.size // 4)
 return lam[-tail_count:], lam[-1] if lam.size else (np.array([]), float('nan'))

def plan_mu(lam_tail, k, n_windows, overlap=0.5, pad_gaps=5, growth="linear"):

 gaps = np.diff(lam_tail)
 gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
 if gaps.size == 0:
 raise RuntimeError("Not enough tail eigenvalues to estimate gaps.")
 g = float(np.median(gaps))



 step = g * max(1.0, k * (1.0 - overlap))
 return g, step

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--merged", default="eigs_1d/eigs_merged.npy")
 ap.add_argument("--k", type=int, default=256, help="eigs per window in harvester")
 ap.add_argument("--tail", type=int, default=400, help="how many tail eigenvalues to analyze")
 ap.add_argument("--windows", type=int, default=60, help="how many μ to propose")
 ap.add_argument("--overlap", type=float, default=0.5)
 ap.add_argument("--pad", type=int, default=5, help="start this many median-gaps beyond current max λ")
 args = ap.parse_args()

 try:
 tail, lam_max = load_tail(args.merged, args.tail)
 except Exception as e:
 print(f"[error] failed to read {args.merged}: {e}")
 sys.exit(1)
 if not np.isfinite(lam_max):
 print("[error] merged file empty; harvest something first.")
 sys.exit(1)

 g, step = plan_mu(tail, args.k, args.windows, overlap=args.overlap, pad_gaps=args.pad)

 mu0 = lam_max + args.pad * g
 mus = [mu0 + i*step for i in range(args.windows)]


 mu_str = ",".join(f"{m:.6f}" for m in mus)
 print(f"# tail median gap g ≈ {g:.6g}")
 print(f"# proposed μ step ≈ {step:.6g} (k={args.k}, overlap={args.overlap:.2f})")
 print(f"# start at μ0 ≈ {mu0:.6g} (pad {args.pad}×g beyond current max λ)")
 print(mu_str)

if __name__ == "__main__":
 main()