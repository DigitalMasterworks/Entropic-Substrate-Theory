#!/usr/bin/env python3
"""
Sweep FFT robustness: thresholds κ × windows × tail fractions.
Counts prime detections for each combo and saves CSV + heatmaps.
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

def tukey(n, alpha=0.5):
 if alpha<=0: return np.ones(n)
 if alpha>=1: return np.hanning(n)
 w = np.ones(n); edge = int(alpha*(n-1)/2)
 if edge>0:
 h = np.hanning(2*edge)
 w[:edge]=h[:edge]; w[-edge:]=h[-edge:]
 return w

def robust_thr(vals,kappa):
 med=float(np.median(vals))
 mad=float(np.median(np.abs(vals-med))) or 1e-12
 return med + kappa*mad

def primes_up_to(n):
 sieve = np.ones(n+1, dtype=bool)
 sieve[:2]=False
 for p in range(2,int(n**0.5)+1):
 if sieve[p]: sieve[p*p:n+1:p]=False
 return [i for i in range(2,n+1) if sieve[i]]

def run_fft(lam,tail_frac,fftpoints,kappa,window,pmax):
 n=lam.size
 i0=int(tail_frac*n)
 lam_tail=lam[i0:]
 N_tail=np.arange(i0+1,n+1,dtype=float)


 A1=1.0/(2*math.pi)
 s=np.sqrt(lam_tail)
 L=np.log(lam_tail); L[np.isnan(L)|np.isinf(L)]=0
 X=np.column_stack([s,L,np.ones_like(s)])
 y=N_tail - A1*(s*L)
 coef,*_=np.linalg.lstsq(X,y,rcond=None)
 B1,C1,D1=coef.tolist()
 N_model=A1*s*L+B1*s+C1*L+D1
 N_rel=N_tail - N_model
 T=np.sqrt(lam_tail-0.25)
 shape=(T/(2*np.pi))*np.log(T)
 shape[np.isnan(shape)|np.isinf(shape)]=0
 coef, *_ = np.linalg.lstsq(np.column_stack([shape, np.ones_like(shape)]), N_rel, rcond=None)
 alpha, beta = coef.tolist()
 trend = alpha*shape + beta
 rem=N_rel - trend; rem -= np.mean(rem)


 Tg=np.linspace(T[0],T[-1],fftpoints)
 Rg=np.interp(Tg,T,rem)
 if window=="hann": W=np.hanning(fftpoints)
 elif window=="tukey": W=tukey(fftpoints,0.5)
 else: W=np.ones(fftpoints)
 F=np.fft.rfft(Rg*W)
 f=np.fft.rfftfreq(fftpoints,d=(Tg[1]-Tg[0]))
 P=np.abs(F)**2; P[0]=0
 thr=robust_thr(P[1:],kappa)

 primes=primes_up_to(pmax)
 det=0
 for p in primes:
 fp=math.log(p)/(2*math.pi)
 if fp<=f[1] or fp>=f[-1]: continue
 k=int(np.argmin(np.abs(f-fp)))
 power=float(P[k])
 side=(P[k]>P[k-1] and P[k]>P[k+1])
 jitter_ok=True
 detected=(power>=thr and side and jitter_ok)
 if detected: det+=1
 return det

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--eigs",required=True)
 ap.add_argument("--out",default="./sweep_fft_out")
 ap.add_argument("--fftpoints",type=int,default=2**15)
 ap.add_argument("--pmax",type=int,default=503)
 args=ap.parse_args()

 os.makedirs(args.out,exist_ok=True)
 lam=load_eigs(args.eigs)

 tail_fracs=[0.7,0.8,0.9,0.95]
 kappas=[2,3,4,5,6]
 windows=["hann","tukey","rect"]

 results=[]
 for tf in tail_fracs:
 for win in windows:
 for kappa in kappas:
 det=run_fft(lam,tf,args.fftpoints,kappa,win,args.pmax)
 results.append((tf,win,kappa,det))


 csv=os.path.join(args.out,"fft_sweep_results.csv")
 with open(csv,"w") as f:
 f.write("tail_frac,window,kappa,detected\n")
 for tf,win,kappa,det in results:
 f.write(f"{tf},{win},{kappa},{det}\n")


 for tf in tail_fracs:
 grid=np.zeros((len(windows),len(kappas)))
 for i,win in enumerate(windows):
 for j,kappa in enumerate(kappas):
 for tf_,win_,kappa_,det in results:
 if tf_==tf and win_==win and kappa_==kappa:
 grid[i,j]=det
 fig,ax=plt.subplots(figsize=(7,4))
 im=ax.imshow(grid,origin="lower",cmap="viridis",
 extent=(min(kappas)-0.5,max(kappas)+0.5,
 -0.5,len(windows)-0.5),aspect="auto")
 ax.set_xticks(kappas)
 ax.set_yticks(range(len(windows)))
 ax.set_yticklabels(windows)
 plt.colorbar(im,ax=ax,label="# primes detected")
 ax.set_xlabel("κ threshold")
 ax.set_ylabel("window")
 ax.set_title(f"FFT sweep detections (tail={tf})")
 heatfile=os.path.join(args.out,f"fft_heatmap_tail{tf}.png")
 plt.tight_layout(); plt.savefig(heatfile,dpi=150); plt.close()
 print("[OK] saved",heatfile)

 print("[OK] saved csv",csv)

if __name__=="__main__":
 main()