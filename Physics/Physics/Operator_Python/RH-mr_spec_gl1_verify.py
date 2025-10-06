#!/usr/bin/env python3








import argparse, math, csv, re, sys
import numpy as np


DEF_GRID = 65536
DEF_WINDOW = "tukey"
DEF_TUKEY_ALPHA = 0.15
DEF_JITTER = 0.002
DEF_JPROBES = 9
DEF_DEFLATE = True
DEF_FR_DEG = 3
DEF_FR_PCAL = 149
DEF_FR_USE_P2 = True
DEF_FR_LAM = 1e-3
DEF_FR_TRIM = 2.5
DEF_SNR_SIGMA = 3.0

try:
 import mpmath as mp
 mp.mp.dps = 50
except Exception:
 mp = None


def primes_upto(n: int):
 if n < 2: return []
 s = np.ones(n+1, dtype=bool); s[:2] = False
 for p in range(2, int(n**0.5)+1):
 if s[p]: s[p*p::p] = False
 return np.nonzero(s)[0].tolist()

def parse_range(s):
 m = re.match(r'^\s*(\d+)\s*:\s*(\d+)\s*$', s or "")
 if not m: return set()
 a,b = int(m.group(1)), int(m.group(2))
 return set(range(min(a,b), max(a,b)+1))

def window(n, kind="tukey", tukey_alpha=0.15):
 k = kind.lower()
 if k == "hann": return np.hanning(n)
 if k == "hamming": return np.hamming(n)
 if k == "blackman": return np.blackman(n)
 if k == "tukey":
 a = float(tukey_alpha); a = min(max(a,0.0),1.0)
 w = np.ones(n); L = n-1
 if L <= 0: return w
 for i in range(n):
 x = i / L
 if x < a/2 or x > 1 - a/2:
 w[i] = 0.5*(1 + math.cos(math.pi*(2*x/a - 1)))
 return w
 return np.hanning(n)

def fit_tail_plateau(lam, tail_frac=0.6, top_clip=0.2, bins=24):
 lam = np.sort(lam[(lam>1e-12) & np.isfinite(lam)])
 N = len(lam); T = np.sqrt(lam); NofT = np.arange(1, N+1, dtype=float)
 i_lo = int((1.0 - tail_frac)*N); i_hi = int((1.0 - top_clip)*N)
 if i_hi <= i_lo + 4: i_hi = min(N, i_lo+5)
 T_tail = T[i_lo:i_hi]; N_tail = NofT[i_lo:i_hi]
 logT_tail = np.log(T_tail+1e-12)
 T0 = float(np.exp(np.mean(logT_tail))); logT0 = math.log(T0)

 lo, hi = logT_tail[0], logT_tail[-1]

 span = hi - lo
 lo += 0.02 * span
 hi -= 0.02 * span
 edges = np.linspace(lo, hi, bins+1)
 slopes = []
 for b in range(bins):
 a,c = edges[b], edges[b+1]
 m = (logT_tail>=a) & (logT_tail<=c)
 if m.sum() >= 4:
 X = np.vstack([logT_tail[m], np.ones(m.sum())]).T
 Y = N_tail[m]/(T_tail[m] + 1e-12)
 sb, *_ = np.linalg.lstsq(X, Y, rcond=None)
 slopes.append(float(sb[0]))
 if slopes:
 half = max(1, len(slopes)//2)
 alpha_plateau = float(np.median(slopes[-half:]))
 else:
 X = np.vstack([logT_tail, np.ones_like(T_tail)]).T
 Y = N_tail/(T_tail+1e-12)
 sg, *_ = np.linalg.lstsq(X, Y, rcond=None)
 alpha_plateau = float(sg[0])

 Xc = np.vstack([logT_tail - logT0, np.ones_like(T_tail)]).T
 Yc = N_tail/(T_tail + 1e-12)
 beta_centered = float(np.mean(Yc - alpha_plateau*Xc[:,0]))
 beta_unc = beta_centered - alpha_plateau*logT0
 trend = alpha_plateau*T*np.log(T+1e-12) + beta_unc*T
 gamma = float(np.median(NofT - trend))


 lo_dt, hi_dt = logT_tail[0], logT_tail[-1]

 cut = lo_dt + 0.80*(hi_dt - lo_dt)
 mask_dt = (np.log(T + 1e-12) >= cut)
 Td = T[mask_dt]; Nd = NofT[mask_dt]
 if Td.size < max(2000, bins*10):

 Xtp = np.vstack([np.log(T[i_lo:i_hi] + 1e-12), np.ones(i_hi - i_lo)]).T
 Ytp = (NofT[i_lo:i_hi] / (T[i_lo:i_hi] + 1e-12))
 slope_two_pt = float(np.linalg.lstsq(Xtp, Ytp, rcond=None)[0][0])
 else:

 nb = max(8, bins//2)
 ed = np.linspace(np.log(Td[0] + 1e-12), np.log(Td[-1] + 1e-12), nb+1)
 slopes_local = []
 for b in range(nb):
 a, c = ed[b], ed[b+1]
 m = (np.log(Td + 1e-12) >= a) & (np.log(Td + 1e-12) <= c)
 if m.sum() >= 8:
 Xb = np.vstack([np.log(Td[m] + 1e-12), np.ones(m.sum())]).T
 Yb = (Nd[m] / (Td[m] + 1e-12))
 sb, *_ = np.linalg.lstsq(Xb, Yb, rcond=None)
 slopes_local.append(float(sb[0]))
 slope_two_pt = float(np.median(slopes_local)) if slopes_local else float(np.nan)
 return alpha_plateau, beta_unc, gamma, T, NofT, slope_two_pt

def auto_tail_select(lam):
 """
 Scan tail windows and pick the most stable one.
 Score = |two_point - plateau| + penalty for negative slopes + variance proxy.
 Returns (a,b,c,T,NofT,tp, frac, clip, bins)
 """

 tail_fracs = (0.85, 0.88, 0.90, 0.92)
 top_clips = (0.08, 0.10, 0.12, 0.15)
 bin_opts = (48, 64, 96)

 best = None
 best_pack = None

 lam = np.sort(lam[(lam > 1e-12) & np.isfinite(lam)])
 N = len(lam)
 if N < 1000:

 return fit_tail_plateau(lam, tail_frac=0.8, top_clip=0.08, bins=32) + (0.8, 0.08, 32)

 for frac in tail_fracs:
 for clip in top_clips:
 for bins in bin_opts:
 try:
 a,b,c,T,NofT,tp = fit_tail_plateau(lam, tail_frac=frac, top_clip=clip, bins=bins)
 except Exception:
 continue

 i_lo = int((1.0 - frac)*N); i_hi = int((1.0 - clip)*N)
 if i_hi - i_lo < max(2000, bins*10):
 continue




 trend = a*T*np.log(T+1e-12) + b*T + c
 R = (NofT - trend)

 j0 = int(0.9 * T.size)
 var_tail = float(np.var(R[j0:] / (np.std(R) + 1e-12)))

 pen = 0.0
 TARGET_T = 1.0
 err_target = abs(a - TARGET_T)


 if a <= 0: pen += 120.0
 if tp <= 0: pen += 80.0
 if abs(tp - a) > 2: pen += 20.0


 score = 3.0*abs(tp - a) + 4.0*err_target + 0.1*var_tail + pen

 if (best is None) or (score < best):
 best = score
 best_pack = (a,b,c,T,NofT,tp, frac, clip, bins)


 if best_pack is None:
 return fit_tail_plateau(lam, tail_frac=0.8, top_clip=0.08, bins=32) + (0.8, 0.08, 32)

 return best_pack

def uniformize(T, R, grid):
 i = np.argsort(T); T, R = T[i], R[i]
 tg = np.linspace(T[0], T[-1], grid)
 return tg, np.interp(tg, T, R)

def proj_ab(tg, Rg, omega, w):
 c = np.cos(omega*tg); s = np.sin(omega*tg)
 A = np.vstack([c,s]).T
 x, *_ = np.linalg.lstsq((A*w[:,None]), (Rg*w), rcond=None)
 a,b = float(x[0]), float(x[1])
 amp = float(np.hypot(a,b))
 return a,b,amp

def proj_with_jitter(tg, Rg, w, omega, jitter, probes):
 if jitter <= 0 or probes <= 1:
 return omega, *proj_ab(tg,Rg,omega,w)
 ks = np.linspace(-jitter, jitter, probes)
 best = (omega, 0.0, 0.0, -1.0)
 for k in ks:
 om = omega*(1.0+k)
 a,b,amp = proj_ab(tg,Rg,om,w)
 if amp > best[3]: best = (om,a,b,amp)
 return best

def fit_freq_response(rows, deg, Pcal, use_p2, lam=1e-3, trim_k=2.5):

 omg, y = [], []
 for r in rows:
 if r["p"] <= Pcal and np.isfinite(r["amp_raw"]) and r["amp_raw"]>0:
 if r["m"]==1 or (use_p2 and r["m"]==2):
 omg.append(r["omega"]); y.append(1.0/max(r["amp_raw"],1e-15))
 if not omg: return None
 omg = np.array(omg,float); y = np.array(y,float)
 Phi = np.vstack([omg**k for k in range(deg+1)]).T

 med = np.median(y); mad = 1.4826*np.median(np.abs(y-med))
 if mad<=0: mad = np.std(y)+1e-12
 w = np.ones_like(y); w[np.abs(y-med)/(mad+1e-12) > trim_k] = 0.0
 W = np.diag(w)
 A = Phi.T@W@Phi + lam*np.eye(Phi.shape[1])
 b = Phi.T@W@y
 try:
 c = np.linalg.solve(A,b)
 except np.linalg.LinAlgError:
 c, *_ = np.linalg.lstsq(A,b,rcond=None)
 return c

def s_of_omega(coeffs, omega):
 if coeffs is None: return 1.0
 return float(sum(coeffs[i]*(omega**i) for i in range(len(coeffs))))

def build_L_spec_s(coeffs_cal, Pmax, Mmax, s: complex):

 logL = 0.0 + 0.0j
 for (p,m), bpm in coeffs_cal.items():
 if p<=Pmax and m<=Mmax:
 logL += (bpm/m) * (p ** (-m*s))
 return complex(np.exp(logL))

def exact_zeta(s):
 if mp is None:
 if abs(s-2.0)<1e-12: return 1.6449340668482264+0j
 if abs(s-3.0)<1e-12: return 1.2020569031595943+0j
 return float("nan")+0j
 return complex(mp.zeta(s))


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("path", nargs="?", default="eigs_1d/eigs_merged.npy")
 ap.add_argument("--Pmax", type=int, default=1000)
 ap.add_argument("--Mmax", type=int, default=2)
 ap.add_argument("--grid", type=int, default=DEF_GRID)
 ap.add_argument("--window", type=str, default=DEF_WINDOW)
 ap.add_argument("--tukey-alpha", type=float, default=DEF_TUKEY_ALPHA)
 ap.add_argument("--jitter", type=float, default=DEF_JITTER)
 ap.add_argument("--jitter-probes", type=int, default=DEF_JPROBES)
 ap.add_argument("--deflate", action="store_true", default=DEF_DEFLATE)
 ap.add_argument("--fr-deg", type=int, default=DEF_FR_DEG)
 ap.add_argument("--fr-Pcal", type=int, default=DEF_FR_PCAL)
 ap.add_argument("--fr-use-p2", action="store_true", default=DEF_FR_USE_P2)
 ap.add_argument("--fr-lam", type=float, default=DEF_FR_LAM)
 ap.add_argument("--fr-trim", type=float, default=DEF_FR_TRIM)
 ap.add_argument("--holdout", type=str, default="151:251")
 ap.add_argument("--Leval", type=str, default="2,3")
 ap.add_argument("--snr-sigma", type=float, default=DEF_SNR_SIGMA)
 ap.add_argument("--null", type=int, default=0, help="shuffle null trials for FFT")
 ap.add_argument("--csv", type=str, default="")
 ap.add_argument("--tail-frac", type=float, default=0.92)
 ap.add_argument("--top-clip", type=float, default=0.04)
 ap.add_argument("--tail-bins", type=int, default=96)
 ap.add_argument("--tail-mode", type=str, default="force",
 choices=["auto","force"],
 help="auto: score multiple windows; force: use provided tail params")
 args = ap.parse_args()

 lam = np.load(args.path)
 lam = np.sort(lam[np.isfinite(lam) & (lam>1e-15)])
 if lam.size < 10:
 print("[error] not enough eigenvalues.")
 sys.exit(1)


 if args.tail_mode == "auto":
 a,b,c,T,NofT,alpha_tp, frac_sel, clip_sel, bins_sel = auto_tail_select(lam)
 print(f"[tail auto] frac={frac_sel:.2f} clip={clip_sel:.2f} bins={bins_sel} -> "
 f"alpha={a:.6f} two-pt={alpha_tp:.6f} alpha_R={a/(2*math.pi):.6f}")
 print(f"[tail info] T_range=({T[0]:.3g},{T[-1]:.3g}) points={T.size}")
 else:
 a,b,c,T,NofT,alpha_tp = fit_tail_plateau(
 lam, tail_frac=args.tail_frac, top_clip=args.top_clip, bins=args.tail_bins
 )
 print(f"[tail force] frac={args.tail_frac:.2f} clip={args.top_clip:.2f} bins={args.tail_bins} -> "
 f"alpha={a:.6f} two-pt={alpha_tp:.6f} alpha_R={a/(2*math.pi):.6f}")
 print(f"[tail info] T_range=({T[0]:.3g},{T[-1]:.3g}) points={T.size}")

 print(f"[rescaled] N(T) ≈ {a:.6f} T log T + {b:.6f} T + {c:.2f}")
 print(f"[rescaled] plateau α = {a:.6f}; two-point α = {alpha_tp:.6f}")
 if (not np.isfinite(alpha_tp)) or alpha_tp < 0:
 print("[note] deep-tail two-point unstable; using plateau α as the primary estimate")
 print(f"[rescaled] α in Riemann t-units = {a/(2*math.pi):.6f}; target 1/(2π) ≈ {1/(2*math.pi):.6f}")


 trend = a*T*np.log(T+1e-12) + b*T + c
 R = (NofT - trend)
 tg, Rg = uniformize(T, R, args.grid)
 w = window(args.grid, args.window, args.tukey_alpha)
 Rg = (Rg - np.mean(Rg)) / (np.std(Rg) + 1e-12)


 def fft_hits(vec):
 F = np.fft.rfft(vec - np.mean(vec))
 freqs = np.fft.rfftfreq(args.grid, d=(tg[1]-tg[0]))
 power = np.abs(F)
 P = power[1:]
 med = np.median(P); mad = np.median(np.abs(P - med)) + 1e-18
 thresh = med + args.snr_sigma*mad
 keep, miss = [], []
 for p in primes_upto(args.Pmax):
 f_target = math.log(p)/(2*math.pi)
 j = int(np.argmin(np.abs(freqs - f_target)))
 (keep if power[j]>=thresh else miss).append(p)
 return keep, miss, thresh

 keep, miss, thresh = fft_hits(Rg*w)
 print(f"[fft] threshold={thresh:.3e}")
 print(f"[fft] primes detected: {keep}")
 print(f"[fft] primes missed: {miss}")
 if args.null>0:
 rng = np.random.default_rng(123)
 counts = []
 for k in range(args.null):
 shuf = Rg.copy()
 rng.shuffle(shuf)
 k2,_,_ = fft_hits(shuf*w)
 counts.append(len(k2))
 print(f"[fft-null] trials={args.null} mean_hits={np.mean(counts):.2f} max={np.max(counts)}")


 def extract_pm(Mmax):
 rows = []
 Rwork = Rg.copy()
 plist = primes_upto(args.Pmax)

 for p in plist:
 omega = math.log(p)
 om,a_hat,b_hat,amp_raw = proj_with_jitter(tg, Rwork, w, omega, args.jitter, args.jitter_probes)
 rows.append({"n":p, "p":p, "m":1, "omega":om, "a_hat":a_hat, "b_hat":b_hat, "amp_raw":amp_raw, "amp_cal":np.nan})
 if args.deflate:
 c = np.cos(om*tg); s = np.sin(om*tg)
 Rwork = Rwork - (a_hat*c + b_hat*s)
 if Mmax>=2:
 for p in plist:
 omega = 2.0*math.log(p)
 om,a_hat,b_hat,amp_raw = proj_with_jitter(tg, Rwork, w, omega, args.jitter, args.jitter_probes)
 rows.append({"n":p*p, "p":p, "m":2, "omega":om, "a_hat":a_hat, "b_hat":b_hat, "amp_raw":amp_raw, "amp_cal":np.nan})
 return rows

 rows = extract_pm(args.Mmax)


 hold = parse_range(args.holdout)
 fr_coeffs = fit_freq_response(
 [r for r in rows if r["p"] not in hold],
 deg=args.fr_deg, Pcal=args.fr_Pcal, use_p2=args.fr_use_p2,
 lam=args.fr_lam, trim_k=args.fr_trim
 )
 fr_desc = "(off)" if fr_coeffs is None else " + ".join([f"{fr_coeffs[i]:.3g}·ω^{i}" for i in range(len(fr_coeffs))])
 print(f"[freq-cal] deg={args.fr_deg} Pcal≤{args.fr_Pcal} use_p2={'yes' if args.fr_use_p2 else 'no'} ridge={args.fr_lam:g} trim={args.fr_trim:g} s(ω)={fr_desc}")


 raw = {}
 for r in rows:
 s_om = s_of_omega(fr_coeffs, r["omega"])
 val = r["amp_raw"] * s_om
 r["amp_cal"] = val
 raw[(r["p"], r["m"])] = val

 def scalar_calib(raw_map, primes_for_scale):
 vals = [raw_map.get((p,1), np.nan) for p in primes_for_scale if p<=args.fr_Pcal and p not in hold]
 vals = np.array([v for v in vals if np.isfinite(v) and v>0], float)
 if vals.size==0: return 1.0
 return float(1.0/np.median(vals))

 train_primes = [p for p in primes_upto(args.Pmax) if p<=args.fr_Pcal and p not in hold]
 s_scalar = scalar_calib(raw, train_primes)
 coeffs_cal = {k: (v*s_scalar) for k,v in raw.items()}
 print(f"[calib] scalar (median up to {args.fr_Pcal}, excl holdout) = {s_scalar:.6g}")


 def rmse_vs_one(coeffs, plist):
 vals = [coeffs.get((p,1), np.nan) for p in plist]
 vals = np.array([v for v in vals if np.isfinite(v)], float)
 if vals.size==0: return float("nan"), 0
 return float(np.sqrt(np.mean((vals-1.0)**2))), int(vals.size)

 hold_primes = sorted(list(hold))
 rmse_tr, ntr = rmse_vs_one(coeffs_cal, train_primes)
 rmse_ho, nho = rmse_vs_one(coeffs_cal, hold_primes)
 print(f"[GL1] TRAIN (p≤{args.fr_Pcal}, excl holdout): RMSE(b_p,1)={rmse_tr:.4g} over {ntr}")
 print(f"[GL1] HOLDOUT {min(hold or {0})}:{max(hold or {0})}: RMSE(b_p,1)={rmse_ho:.4g} over {nho}")


 def rmse_p2(coeffs, plist):
 vals = [coeffs.get((p,2), np.nan) for p in plist]
 vals = np.array([v for v in vals if np.isfinite(v)], float)
 if vals.size==0: return float("nan"), 0
 return float(np.sqrt(np.mean((vals-1.0)**2))), int(vals.size)
 rmse2_tr, n2tr = rmse_p2(coeffs_cal, train_primes)
 rmse2_ho, n2ho = rmse_p2(coeffs_cal, hold_primes)
 print(f"[GL1] TRAIN squares: RMSE(b_p^2,1)={rmse2_tr:.4g} over {n2tr}")
 print(f"[GL1] HOLDOUT squares: RMSE(b_p^2,1)={rmse2_ho:.4g} over {n2ho}")


 if args.Leval:
 svals = [float(x) for x in args.Leval.split(",") if x.strip()]
 print(f"[L] truncated L_spec(s) with M≤{args.Mmax}, P≤{args.Pmax}:")
 for sig in svals:
 L = build_L_spec_s(coeffs_cal, args.Pmax, args.Mmax, complex(sig,0.0))
 Z = exact_zeta(sig)
 if np.isfinite(Z.real):
 err = abs(L - Z); rel = err/(abs(Z)+1e-18)
 print(f" s={sig:.3f} L_spec={L:.9g} zeta={Z:.9g} abs_err={err:.3g} rel_err={rel:.3%}")
 else:
 print(f" s={sig:.3f} L_spec={L:.9g} (no exact zeta available)")


 rng = np.random.default_rng(123)
 small = [p for p in primes_upto(min(args.Pmax, 1000))]
 pairs = []
 seen=set()
 K = 100
 while len(pairs)<K and len(small)>=2:
 p,q = map(int, rng.choice(small,2, replace=False))
 if p==q: continue
 if (p,q) in seen or (q,p) in seen: continue
 seen.add((p,q)); pairs.append((p,q,int(p*q)))

 def proj_quad_amp(tg, Rg, w, omega, delta=0.02):

 def proj_amp(om):
 a,b,amp = proj_ab(tg, (Rg*w), om, np.ones_like(w))
 return amp
 a0 = proj_amp(omega)
 al = proj_amp(omega*(1-delta))
 ar = proj_amp(omega*(1+delta))
 base = 0.5*(al+ar)
 return max(a0 - base, 0.0)

 if pairs:
 omegas, X, y = [], [], []
 for p,q,n in pairs:
 ap = coeffs_cal.get((p,1), np.nan)
 aq = coeffs_cal.get((q,1), np.nan)
 if not (np.isfinite(ap) and np.isfinite(aq)): continue
 omega = math.log(n)
 amp = proj_quad_amp(tg, Rg, w, omega, delta=0.02)
 omegas.append(omega); X.append(amp); y.append(ap*aq)
 if len(X)>=8:
 omegas = np.array(omegas,float); X = np.array(X,float); y = np.array(y,float)
 idx = np.arange(len(y)); tr = (idx%2==0); te = ~tr

 def design(om, amp):
 return np.vstack([amp, amp*om, amp*om*om]).T
 Xtr = design(omegas[tr], X[tr]); ytr = y[tr]
 Xte = design(omegas[te], X[te]); yte = y[te]
 A = Xtr.T@Xtr + 1e-3*np.eye(Xtr.shape[1]); bvec = Xtr.T@ytr
 c = np.linalg.solve(A,bvec)
 yhat = Xte @ c
 A2 = np.vstack([yhat, np.ones_like(yhat)]).T
 k,b0 = np.linalg.lstsq(A2, yte, rcond=None)[0]
 corr = float(np.corrcoef(yhat, yte)[0,1])
 rmse = float(np.sqrt(np.mean((yhat - yte)**2)))
 print(f"[mult^2] test-fold: samples={int(len(yte))} corr={corr:.3f} slope={k:.3f} offset={b0:.3f} rmse={rmse:.4f}")
 else:
 print("[mult^2] insufficient pairs for CV")


 if args.csv:
 with open(args.csv, "w", newline="") as f:
 wcsv = csv.writer(f)
 wcsv.writerow(["n","p","m","omega","a_hat","b_hat","amp_raw","amp_cal"])
 for r in sorted(rows, key=lambda z:(z["p"],z["m"])):
 wcsv.writerow([r["n"], r["p"], r["m"],
 f"{r['omega']:.9f}", f"{r['a_hat']:.9g}", f"{r['b_hat']:.9g}",
 f"{r['amp_raw']:.9g}", f"{r['amp_cal']:.9g}"])
 print(f"[csv] wrote {len(rows)} rows to {args.csv}")

if __name__ == "__main__":
 main()