#!/usr/bin/env python3


import argparse, sys, math
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

def build_S_radial_r(N, L, eps=1e-3):
 x = np.linspace(-L/2, L/2, N, dtype=np.float64)
 y = np.linspace(-L/2, L/2, N, dtype=np.float64)
 X, Y = np.meshgrid(x, y, indexing="ij")
 R = np.hypot(X, Y)
 return np.maximum(R, eps)

def build_H_div_form_cpu(N, L, eps=1e-3):
 """
 Assemble H = -div(S^2 grad) (5-pt divergence-form, Dirichlet) on CPU.
 Matches verify.py semantics so the spectra align.
 """
 S = build_S_radial_r(N, L, eps=eps)
 S2 = S*S
 h = L/(N-1)


 cx = np.zeros_like(S2); cy = np.zeros_like(S2)
 cx[:-1,:] = 0.5*(S2[:-1,:] + S2[1:,:])
 cy[:,:-1] = 0.5*(S2[:,:-1] + S2[:,1:])

 N2 = N*N
 rows = []
 cols = []
 data = []
 diag = np.zeros((N,N), dtype=np.float64)

 def add(r,c,v):
 rows.append(r); cols.append(c); data.append(v)
 def idx(i,j):
 return i*N + j


 for i in range(N-1):
 for j in range(N):
 w = cx[i,j]/(h*h)
 add(idx(i, j), idx(i+1, j), -w)
 add(idx(i+1, j), idx(i, j), -w)
 diag[i, j] += w
 diag[i+1, j] += w


 for i in range(N):
 for j in range(N-1):
 w = cy[i,j]/(h*h)
 add(idx(i, j), idx(i, j+1), -w)
 add(idx(i, j+1), idx(i, j), -w)
 diag[i, j] += w
 diag[i, j+1] += w

 H_off = csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(N2, N2))
 H_diag = diags(diag.reshape(-1), 0, dtype=np.float64, format="csr")
 return H_diag + H_off

def parse_mu(s):
 if ":" in s:
 a,b,c = [float(x) for x in s.split(":")]
 return ("range", a, b, c)
 vals = [float(x) for x in s.split(",") if x.strip()]
 return ("list", vals)

def merge_unique(paths, out_path):
 buf=[]
 for p in paths:
 try:
 e = np.load(p)
 e = np.real(e[np.isfinite(e)])
 if e.size: buf.append(e)
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
 ap = argparse.ArgumentParser(description="Windowed eigen harvest until minimum unique count is met.")
 ap.add_argument("--N", type=int, required=True)
 ap.add_argument("--L", type=float, required=True)
 ap.add_argument("--k", type=int, default=800, help="eigs per window")
 ap.add_argument("--mu", required=True, help="a:b:step or comma list (initial windows)")
 ap.add_argument("--tol", type=float, default=1e-8)
 ap.add_argument("--outdir", default="eigs_windows")
 ap.add_argument("--min-unique", type=int, default=40000, help="stop when merged unique >= this")
 ap.add_argument("--merge-every", type=int, default=25, help="merge after this many new windows")
 ap.add_argument("--max-mu", type=float, default=1e9, help="hard ceiling for window center growth")
 ap.add_argument("--max-windows", type=int, default=100000, help="safety cap on total windows")
 args = ap.parse_args()

 outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
 merged_path = outdir / "eigs_merged.npy"

 mode, *mu_params = parse_mu(args.mu)
 if mode == "range":
 a,b,c = mu_params
 mu_list = [a + i*c for i in range(max(0, int(math.floor((b - a)/c))+1))]
 else:
 mu_list = mu_params[0]
 a = mu_list[0] if mu_list else 0.0
 c = (mu_list[1] - mu_list[0]) if len(mu_list) >= 2 else 2.0
 b = mu_list[-1] if mu_list else a

 H = build_H_div_form_cpu(args.N, args.L, eps=1e-3)


 lam_min = eigsh(H, k=1, which='SA', return_eigenvectors=False)[0]
 lam_max = eigsh(H, k=1, which='LA', return_eigenvectors=False)[0]
 print(f"[bracket] lam_min={lam_min:.6g} lam_max={lam_max:.6g}")

 def window_file(mu):
 tag = f"{mu:.6g}".replace(".","p")
 return outdir / f"eigs_mu_{tag}.npy"

 processed = 0
 newly_produced = 0


 def run_window(mu):
 nonlocal processed, newly_produced
 p = window_file(mu)
 if p.exists():
 print(f"[skip] {p}")
 return
 print(f"[win] mu={mu}:: computing {args.k} eigs")
 try:
 vals = eigsh(H, k=args.k, sigma=mu, which='LM', tol=args.tol, return_eigenvectors=False)
 vals = np.real(vals[np.isfinite(vals)])
 vals.sort()
 np.save(p, vals)
 print(f"[win] saved {p} ({vals.size})")
 newly_produced += 1
 except Exception as ex:
 print(f"[warn] window mu={mu} failed: {ex}")
 processed += 1


 mu_list = [mu for mu in mu_list if (lam_min < mu < lam_max)]


 for mu in mu_list:
 if processed >= args.max_windows:
 break
 run_window(mu)

 if newly_produced >= args.merge_every:
 count = merge_unique(list(sorted(outdir.glob("eigs_mu_*.npy"))), merged_path)
 print(f"[merge] unique={count} -> {merged_path}")
 newly_produced = 0
 if count >= args.min_unique:
 print(f"[done] reached min-unique={args.min_unique}")
 return


 base = (b if mode == "range" else (mu_list[-1] if mu_list else a))

 current_mu = max(base + c, lam_min + 0.5*abs(c))
 while (processed < args.max_windows) and (current_mu < min(args.max_mu, lam_max)):
 run_window(current_mu)
 if newly_produced >= args.merge_every:
 count = merge_unique(list(sorted(outdir.glob("eigs_mu_*.npy"))), merged_path)
 print(f"[merge] unique={count} -> {merged_path}")
 newly_produced = 0
 if count >= args.min_unique:
 print(f"[done] reached min-unique={args.min_unique}")
 return
 current_mu += c


 count = merge_unique(list(sorted(outdir.glob("eigs_mu_*.npy"))), merged_path)
 print(f"[final] unique={count} -> {merged_path}")
 if count < args.min_unique:
 print(f"[note] stopped without reaching min-unique={args.min_unique} (hit bounds).")

if __name__ == "__main__":
 main()