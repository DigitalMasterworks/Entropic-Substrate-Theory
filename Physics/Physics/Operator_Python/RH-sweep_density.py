#!/usr/bin/env python3
"""
Sweep density fits with R² heatmap.
"""

import os, math, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_eigs(path):
 arr = np.load(path)
 arr = np.array(arr).ravel()
 arr = arr[np.isfinite(arr)]
 arr = arr[arr > 0.0]
 return np.unique(np.sort(arr))

def counting_function(sorted_lams, lam_vals):
 return np.searchsorted(sorted_lams, lam_vals, side="right").astype(float)

def affine_fit(x,y):
 X = np.column_stack([x, np.ones_like(x)])
 coef, *_ = np.linalg.lstsq(X, y, rcond=None)
 a,b = coef.tolist()
 yhat = a*x + b
 ss_res = float(np.sum((y-yhat)**2))
 ss_tot = float(np.sum((y - np.mean(y))**2)) or 1.0
 r2 = 1.0 - ss_res/ss_tot
 return a,b,r2,yhat

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--eigs", required=True)
 ap.add_argument("--out", default="./sweep_out_heatmap")
 args = ap.parse_args()

 os.makedirs(args.out, exist_ok=True)
 lam = load_eigs(args.eigs)
 n = lam.size

 A1 = 1.0/(2*math.pi)

 tail_fracs = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
 bin_counts = [20, 30, 50, 75, 100, 150, 200]

 R2_grid = np.zeros((len(tail_fracs), len(bin_counts)))

 results = []

 for i, tf in enumerate(tail_fracs):
 i0 = int(tf*n)
 lam_tail = lam[i0:]
 N_tail = np.arange(i0+1,n+1,dtype=float)

 s = np.sqrt(lam_tail)
 L = np.log(lam_tail); L[np.isnan(L)|np.isinf(L)] = 0
 X = np.column_stack([s, L, np.ones_like(s)])
 y = N_tail - A1*(s*L)
 coef, *_ = np.linalg.lstsq(X,y,rcond=None)
 B1,C1,D1 = coef.tolist()
 N_model = A1*s*L + B1*s + C1*L + D1
 N_rel = N_tail - N_model
 T = np.sqrt(lam_tail-0.25)
 shape = (T/(2*np.pi))*np.log(T)
 shape[np.isnan(shape)|np.isinf(shape)] = 0

 for j, bins in enumerate(bin_counts):
 edges = np.linspace(T[0],T[-1],bins+1)
 T_centers=[]; N_rel_bin=[]; shape_bin=[]
 for k in range(bins):
 mask=(T>=edges[k])&(T<edges[k+1])
 if not np.any(mask): continue
 T_centers.append(0.5*(edges[k]+edges[k+1]))
 N_rel_bin.append(np.mean(N_rel[mask]))
 shape_bin.append(np.mean(shape[mask]))
 if len(T_centers)<3:
 R2_grid[i,j]=np.nan
 continue
 T_centers=np.array(T_centers)
 N_rel_bin=np.array(N_rel_bin)
 shape_bin=np.array(shape_bin)

 alpha,beta,r2,yhat=affine_fit(shape_bin,N_rel_bin)
 R2_grid[i,j]=r2
 results.append((tf,bins,alpha,beta,r2))


 csv_path=os.path.join(args.out,"sweep_results.csv")
 with open(csv_path,"w") as f:
 f.write("tail_frac,bin_count,alpha,beta,R2\n")
 for tf,bins,alpha,beta,r2 in results:
 f.write(f"{tf},{bins},{alpha},{beta},{r2}\n")


 fig,ax=plt.subplots(figsize=(8,6))
 im=ax.imshow(R2_grid,origin="lower",cmap="viridis",
 extent=(min(bin_counts)-5,max(bin_counts)+5,
 min(tail_fracs)-0.05,max(tail_fracs)+0.05),
 aspect="auto")
 plt.colorbar(im,ax=ax,label="R²")
 ax.set_xticks(bin_counts)
 ax.set_yticks(tail_fracs)
 ax.set_xlabel("bin count")
 ax.set_ylabel("tail fraction")
 ax.set_title("R² heatmap (tail vs bins)")

 for i,tf in enumerate(tail_fracs):
 for j,bins in enumerate(bin_counts):
 r2=R2_grid[i,j]
 if np.isnan(r2): continue
 ax.text(bins,tf,f"{r2:.2f}",ha="center",va="center",color="w",fontsize=7)

 heatfile=os.path.join(args.out,"R2_heatmap.png")
 plt.tight_layout(); plt.savefig(heatfile,dpi=150); plt.close()

 print("[OK] Saved R² heatmap to",heatfile)
 print("[OK] Saved sweep table to",csv_path)

if __name__=="__main__":
 main()