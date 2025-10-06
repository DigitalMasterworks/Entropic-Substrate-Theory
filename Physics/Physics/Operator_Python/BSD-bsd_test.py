
import argparse, csv, math, sys
import numpy as np

def read_ap_csv(path):
 ap = {}
 with open(path, "r") as f:
 rd = csv.DictReader(f)
 for r in rd:
 p = int(r["p"])
 good = int(r.get("good", "1"))
 if good == 1:
 ap[p] = int(r["ap"])
 return ap

def primes_upto(n):
 if n < 2: return []
 s = np.ones(n+1, dtype=bool); s[:2] = False
 for p in range(2, int(n**0.5)+1):
 if s[p]: s[p*p::p] = False
 return np.nonzero(s)[0].tolist()


def semicircle_cdf(t):
 t = np.clip(t, -1.0, 1.0)
 return 0.5 + (t*np.sqrt(np.maximum(0.0, 1.0 - t*t)) + np.arcsin(t))/math.pi

def sato_tate_ks(ap_dict, Pmax):
 data = []
 for p, ap in ap_dict.items():
 if p <= Pmax:
 tp = (ap/np.sqrt(p))/2.0
 data.append(tp)
 if not data:
 return {"count":0, "KS": float("nan")}
 x = np.sort(np.array(data, float))
 n = x.size
 F_emp = (np.arange(1, n+1) / n)
 F_st = semicircle_cdf(x)
 KS = float(np.max(np.abs(F_emp - F_st)))
 return {"count": n, "KS": KS}

def L_from_ap_real_s(s, ap_dict, Pmax):
 prod = 1.0
 for p, ap in ap_dict.items():
 if p > Pmax: continue
 term = 1.0 - ap/(p**s) + (p**(1.0 - 2.0*s))
 prod *= 1.0/term
 return prod

def rank_near_one(ap_dict, Pmax, h=0.02, k=5):

 hs = np.array([h*(j+1) for j in range(k)], float)
 Ls = []
 for hj in hs:
 s = 1.0 + hj
 val = L_from_ap_real_s(s, ap_dict, Pmax)
 if not np.isfinite(val) or val <= 0:
 return {"h":h, "k":k, "r_hat": float("nan"), "R2": float("nan")}
 Ls.append(val)
 y = np.log(np.array(Ls))
 X = np.vstack([np.log(hs), np.ones_like(hs)]).T
 beta, *_ = np.linalg.lstsq(X, y, rcond=None)
 r_hat = float(beta[0])
 yhat = X @ beta
 ss_res = float(np.sum((y - yhat)**2))
 ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-18
 R2 = 1.0 - ss_res/ss_tot
 return {"h":h, "k":k, "r_hat": r_hat, "R2": R2}

def read_invariants(path):

 kv = {}
 with open(path, "r") as f:
 txt = f.read().strip().replace("\n", ",")
 toks = [t.strip() for t in txt.split(",") if t.strip()!=""]
 for i in range(0, len(toks)-1, 2):
 kv[toks[i]] = toks[i+1]
 N = int(kv.get("N","0"))
 w = int(kv.get("w","1"))
 return N, w

def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--ap-file", required=True)
 ap.add_argument("--Pmax", type=int, default=2000)
 ap.add_argument("--st-test", action="store_true")
 ap.add_argument("--rank-near-one", action="store_true")
 ap.add_argument("--rank-h", type=float, default=0.02)
 ap.add_argument("--rank-k", type=int, default=5)
 ap.add_argument("--ec-invariants", type=str, default="")
 args = ap.parse_args()

 ap_dict = read_ap_csv(args.ap_file)
 if not ap_dict:
 print("[error] empty a_p file"); sys.exit(1)

 Pcap = min(args.Pmax, max(ap_dict.keys()))
 print(f"[info] primes used: good p ≤ {Pcap} (from {args.ap_file})")

 if args.st_test:
 st = sato_tate_ks(ap_dict, Pcap)
 print(f"[ST] KS distance = {st['KS']:.4f} over {st['count']} primes")

 if args.rank_near_one:
 rk = rank_near_one(ap_dict, Pcap, h=args.rank_h, k=args.rank_k)
 print(f"[BSD-rank] r_hat = {rk['r_hat']:.3f}, R2 = {rk['R2']:.3f}, h = {rk['h']}, k = {rk['k']}")

 if args.ec_invariants:
 N, w = read_invariants(args.ec_invariants)
 if N > 0:
 r_int = int(round(rk['r_hat'])) if np.isfinite(rk['r_hat']) else None
 parity = (r_int % 2) if r_int is not None else None
 ok = (parity is not None) and ((w == -1 and parity == 1) or (w == +1 and parity == 0))
 print(f"[FE/parity] N={N}, w={w}, estimated rank≈{r_int} → parity={parity} "
 f"{'OK' if ok else ' parity mismatch' if parity is not None else 'n/a'}")

if __name__ == "__main__":
 main()