import numpy as np, math, os, textwrap

OUTDIR="eigs_windows"
MERGED=os.path.join(OUTDIR,"eigs_merged.npy")
TARGET_ADD=1500
K=8
GAP_MIN=30.0

E=np.load(MERGED); E=np.sort(E[np.isfinite(E)])
d=np.diff(E)

cand_mu=[0.5*(E[i]+E[i+1]) for i in range(len(d)) if d[i]>GAP_MIN]


cand_mu=cand_mu[:math.ceil(TARGET_ADD/max(1,K))]

def chunks(lst,n):
 for i in range(0,len(lst),n):
 yield lst[i:i+n]

print(f"# gap-sniper: gaps>{GAP_MIN}, proposing {len(cand_mu)} centers, k={K}")
for idx,chunk in enumerate(chunks(cand_mu,60),1):
 mu_arg=",".join(f"{m:.6f}" for m in chunk)
 print(f"\n# Run gap {idx}")
 print(f"python3 batch_eigs.py --N 128 --L 100 --k {K} "
 f"--mu {mu_arg} --tol 1e-8 --outdir {OUTDIR} "
 f"--min-unique 16000 --merge-every 25")
