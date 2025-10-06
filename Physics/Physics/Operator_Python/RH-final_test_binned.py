#!/usr/bin/env python3
"""
Final test (binned-fit version): Relative determinant + prime FFT.

- Locks A1 = 1/(2π).
- Uses only the top tail fraction (--tailpct).
- Bins the relative counts before fitting to Riemann shape, giving a cleaner R².
- Outputs side-by-side density/FFT figure and detections list.
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

def primes_up_to(n):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2] = False
 for p in range(2,int(n**0.5)+1):
 if sieve[p]:
 sieve[p*p:n+1:p] = False
 return [i for i in range(2,n+1) if sieve[i]]

def affine_fit(x,y):
 X = np.column_stack([x, np.ones_like(x)])
 coef, *_ = np.linalg.lstsq(X,y,rcond=None)
 a,b = coef.tolist()
 yhat = a*x + b
 ss_res = float(np.sum((y-yhat)**2))
 ss_tot = float(np.sum((y - np.mean(y))**2)) or 1.0
 r2 = 1.0 - ss_res/ss_tot
 return a,b,r2,yhat

def tukey(n, alpha=0.5):
 if alpha<=0: return np.ones(n)
 if alpha>=1: return np.hanning(n)
 w = np.ones(n); edge = int(alpha*(n-1)/2)
 if edge>0:
 h = np.hanning(2*edge)
 w[:edge] = h[:edge]; w[-edge:] = h[-edge:]
 return w

def robust_thr(vals,kappa):
 med = float(np.median(vals))
 mad = float(np.median(np.abs(vals-med))) or 1e-12
 return med + kappa*mad

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--eigs", required=True)
 ap.add_argument("--out", default="./out_final_binned")
 ap.add_argument("--tailpct", type=float, default=0.9)
 ap.add_argument("--bins", type=int, default=100)
 ap.add_argument("--fftpoints", type=int, default=2**16)
 ap.add_argument("--pmax", type=int, default=503)
 ap.add_argument("--kappa", type=float, default=3.0)
 args = ap.parse_args()

 os.makedirs(args.out,exist_ok=True)
 figfile = os.path.join(args.out,"final_density_fft_binned.png")
 txtfile = os.path.join(args.out,"prime_detections_binned.txt")

 lam = load_eigs(args.eigs)
 n = lam.size
 i0 = int(args.tailpct*n)
 lam_tail = lam[i0:]
 N_tail = np.arange(i0+1,n+1,dtype=float)


 A1 = 1.0/(2*math.pi)
 s = np.sqrt(lam_tail); L = np.log(lam_tail)
 L[np.isnan(L)|np.isinf(L)] = 0
 X = np.column_stack([s,L,np.ones_like(s)])
 y = N_tail - A1*(s*L)
 coef, *_ = np.linalg.lstsq(X,y,rcond=None)
 B1,C1,D1 = coef.tolist()
 N_model = A1*s*L + B1*s + C1*L + D1
 N_rel = N_tail - N_model
 T = np.sqrt(lam_tail-0.25)
 shape = (T/(2*np.pi))*np.log(T)
 shape[np.isnan(shape)|np.isinf(shape)] = 0


 bins = np.linspace(T[0],T[-1],args.bins+1)
 T_centers=[]; N_rel_bin=[]; shape_bin=[]
 for k in range(args.bins):
 mask=(T>=bins[k])&(T<bins[k+1])
 if not np.any(mask): continue
 T_centers.append(0.5*(bins[k]+bins[k+1]))
 N_rel_bin.append(np.mean(N_rel[mask]))
 shape_bin.append(np.mean(shape[mask]))
 T_centers=np.array(T_centers); N_rel_bin=np.array(N_rel_bin); shape_bin=np.array(shape_bin)

 alpha,beta,r2,yhat=affine_fit(shape_bin,N_rel_bin)


 trend = alpha*shape + beta
 rem = N_rel - trend; rem -= np.mean(rem)
 Tg=np.linspace(T[0],T[-1],args.fftpoints)
 Rg=np.interp(Tg,T,rem)
 W=tukey(args.fftpoints,0.5)
 F=np.fft.rfft(Rg*W)
 f=np.fft.rfftfreq(args.fftpoints,d=(Tg[1]-Tg[0]))
 P=np.abs(F)**2; P[0]=0
 thr=robust_thr(P[1:],args.kappa)

 primes=primes_up_to(args.pmax)
 detections=[]
 for p in primes:
 fp=math.log(p)/(2*math.pi)
 if fp<=f[1] or fp>=f[-1]: continue
 k=int(np.argmin(np.abs(f-fp)))
 power=float(P[k])
 side=(P[k]>P[k-1] and P[k]>P[k+1])
 detected=(power>=thr and side)
 detections.append((p,fp,power,detected))


 fig,axs=plt.subplots(1,2,figsize=(13,5))
 axs[0].plot(T,N_rel,alpha=0.3,label="raw residuals")
 axs[0].plot(T_centers,N_rel_bin,"o",label="binned avg")
 axs[0].plot(T_centers,yhat,"--",label=f"fit (R²={r2:.3f})")
 axs[0].set_xlabel("T"); axs[0].set_ylabel("Relative counts")
 axs[0].set_title("Zero-density (binned fit)")
 axs[0].legend()

 valid=(f>0)&(P>0)
 axs[1].loglog(f[valid],P[valid],label="FFT power")
 axs[1].axhline(thr,ls="--",alpha=0.4,label="threshold")
 for p,fp,power,det in detections:
 if det:
 axs[1].axvline(fp,color="r",ls=":",alpha=0.3)
 axs[1].scatter([fp],[power],c="g",s=15)
 axs[1].set_xlabel("frequency (log p / 2π)")
 axs[1].set_ylabel("power")
 axs[1].set_title("Prime frequencies (relative remainder)")
 axs[1].legend()
 plt.tight_layout(); plt.savefig(figfile,dpi=150); plt.close()

 with open(txtfile,"w") as g:
 g.write("# p\tfreq\tpower\tdetected\n")
 for p,fp,power,det in detections:
 g.write(f"{p}\t{fp:.6g}\t{power:.6g}\t{int(det)}\n")

 print("[OK] Saved:")
 print(" ",figfile)
 print(" ",txtfile)
 print(f"[INFO] Fit: alpha={alpha:.6g}, beta={beta:.6g}, R²={r2:.3f}")
 print(f"[INFO] Detected {sum(1 for *_,d in detections if d)} primes")

if __name__=="__main__":
 main()