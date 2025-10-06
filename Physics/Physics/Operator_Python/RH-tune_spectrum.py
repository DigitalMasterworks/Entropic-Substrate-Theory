
import numpy as np, time
from pathlib import Path


import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as sla


from laplace_spectrum import (
 spacing_ratios, unfold, local_unfold, ks_to_ensemble
)
from build_S import halo_S

TARGET_R = 0.60266




def div_form_L_magnetic(S, h, alpha):
 import cupy as cp, cupyx.scipy.sparse as sp
 N = S.shape[0]
 S2 = S*S

 if alpha == 0.0:

 c_x = cp.zeros_like(S); c_y = cp.zeros_like(S)
 c_x[:-1,:] = 0.5*(S2[:-1,:] + S2[1:,:])
 c_y[:,:-1] = 0.5*(S2[:,:-1] + S2[:,1:])

 rows=[]; cols=[]; data=[]
 def add(r,c,v): rows.append(r); cols.append(c); data.append(float(v))
 def idx(i,j): return i*N + j

 diag = cp.zeros((N,N), dtype=cp.float64)


 for i in range(N-1):
 for j in range(N):
 w = c_x[i,j]/(h*h)
 add(idx(i,j), idx(i+1,j), -w)
 add(idx(i+1,j), idx(i,j), -w)
 diag[i,j] += w
 diag[i+1,j] += w


 for i in range(N):
 for j in range(N-1):
 w = c_y[i,j]/(h*h)
 add(idx(i,j), idx(i,j+1), -w)
 add(idx(i,j+1), idx(i,j), -w)
 diag[i,j] += w
 diag[i,j+1] += w

 rows = cp.asarray(rows, dtype=cp.int32)
 cols = cp.asarray(cols, dtype=cp.int32)
 data = cp.asarray(data, dtype=cp.float64)
 H_off = sp.csr_matrix((data, (rows, cols)), shape=(N*N, N*N))
 H_diag = sp.diags(diag.reshape(-1), 0, dtype=cp.float64, format="csr")
 return H_diag + H_off


 logS = cp.log(S + 1e-18)
 dlogS_dx = cp.zeros_like(S); dlogS_dy = cp.zeros_like(S)
 dlogS_dx[1:-1,:] = (logS[2:,:] - logS[:-2,:])/(2*h)
 dlogS_dy[:,1:-1] = (logS[:,2:] - logS[:,:-2])/(2*h)
 A_x = alpha * dlogS_dy
 A_y = -alpha * dlogS_dx

 c_x = cp.zeros_like(S); c_y = cp.zeros_like(S)
 c_x[:-1,:] = 0.5*(S2[:-1,:] + S2[1:,:])
 c_y[:,:-1] = 0.5*(S2[:,:-1] + S2[:,1:])

 rows=[]; cols=[]; data=[]
 def add(r,c,v): rows.append(r); cols.append(c); data.append(complex(v))
 def idx(i,j): return i*N + j

 diag = cp.zeros((N,N), dtype=cp.float64)

 for i in range(N-1):
 for j in range(N):
 phase = cp.exp(1j * h * A_x[i,j]); w = c_x[i,j]/(h*h)
 add(idx(i,j), idx(i+1,j), -w*phase)
 add(idx(i+1,j), idx(i,j), -w*cp.conj(phase))
 diag[i,j] += w; diag[i+1,j] += w

 for i in range(N):
 for j in range(N-1):
 phase = cp.exp(1j * h * A_y[i,j]); w = c_y[i,j]/(h*h)
 add(idx(i,j), idx(i,j+1), -w*phase)
 add(idx(i,j+1), idx(i,j), -w*cp.conj(phase))
 diag[i,j] += w; diag[i,j+1] += w

 rows = cp.asarray(rows, dtype=cp.int32)
 cols = cp.asarray(cols, dtype=cp.int32)
 data = cp.asarray(data, dtype=cp.complex128)
 H_off = sp.csr_matrix((data, (rows, cols)), shape=(N*N, N*N))
 H_diag = sp.diags(diag.reshape(-1), 0, dtype=cp.complex128, format="csr")
 return H_diag + H_off




def eval_spectrum(S_cpu, alpha=0.0, k=600, bulk_lo=0.10, bulk_hi=0.90, local_win=51):
 S = cp.asarray(S_cpu)
 N = S.shape[0]
 Lbox = 100.0
 h = Lbox/(N-1)

 H = div_form_L_magnetic(S, h, alpha)


 H_H = H.get()
 sym_err = np.linalg.norm(H_H.toarray() - H_H.T.conj(), ord='fro') / (1 + np.linalg.norm(H_H.toarray(), ord='fro'))


 rng = cp.random.RandomState(1234)
 if H.dtype == cp.complex128:
 v0 = (rng.rand(H.shape[0]) + 1j*rng.rand(H.shape[0])).astype(cp.complex128)
 else:
 v0 = rng.rand(H.shape[0]).astype(cp.float64)

 vals, _ = sla.eigsh(H, k=int(k), which="SA", v0=v0)
 vals = np.sort(cp.asnumpy(vals))


 lo, hi = int(bulk_lo*len(vals)), int(bulk_hi*len(vals))
 bulk = vals[lo:hi]


 stair = np.poly1d(np.polyfit(vals, np.arange(1, len(vals)+1), 3))
 s_global = np.diff(stair(vals))
 _, ks_gue_global = ks_to_ensemble(s_global, "GUE")


 s_loc = local_unfold(bulk, win=local_win)
 _, ks_gue_local = ks_to_ensemble(s_loc, "GUE")

 r_bulk = spacing_ratios(bulk).mean()
 r_all = spacing_ratios(vals).mean()

 return {
 "ks_gue_local": float(ks_gue_local),
 "ks_gue_global": float(ks_gue_global),
 "r_bulk": float(r_bulk),
 "r_all": float(r_all),
 "sym_err": float(sym_err),
 "n_spacings": int(len(s_loc)),
 }

def objective(m, w1=1.0, w2=0.5, w3=0.25):

 pen = 10.0 * min(1.0, m["sym_err"]*1e3)
 return (w1*m["ks_gue_local"]
 + w2*abs(m["r_bulk"] - TARGET_R)
 + w3*m["ks_gue_global"]
 + pen)




def coarse_search(N=128, k=300, local_win=51):
 print(f"\n[coarse] N={N} k={k} win={local_win}")
 candidates = []
 for alpha in np.geomspace(1e-4, 3e-1, 14):
 S = halo_S(N=N, L=100.0)
 m = eval_spectrum(S, alpha=alpha, k=k, bulk_lo=0.10, bulk_hi=0.90, local_win=local_win)
 J = objective(m)
 candidates.append((J, {"alpha":alpha, **m}))
 print(f"[grid] a={alpha:.2e} -> J={J:.4f} r~={m['r_bulk']:.5f} KS_loc={m['ks_gue_local']:.4f} KS_glob={m['ks_gue_global']:.4f} Herm={m['sym_err']:.2e}")
 candidates.sort(key=lambda x: x[0])
 return candidates[:4]

def nelder_mead_alpha(seed_alpha, N=128, iters=18):
 """
 1D Nelder–Mead on alpha (only), k=600. Keeps constants S fixed.
 """

 simplex = [float(seed_alpha), float(seed_alpha)*1.5]

 def eval_alpha(a):
 a = float(np.clip(a, 1e-6, 1.0))
 S = halo_S(N=N, L=100.0)
 m = eval_spectrum(S, alpha=a, k=600, bulk_lo=0.10, bulk_hi=0.90, local_win=51)
 return objective(m), m

 vals = [eval_alpha(a) for a in simplex]
 for _ in range(iters):

 order = sorted(zip(simplex, vals), key=lambda x: x[1][0])
 simplex, vals = [a for a,_ in order], [v for _,v in order]
 best = simplex[0]

 c = simplex[0]
 xr = 2*c - simplex[1]
 Jr, Mr = eval_alpha(xr)
 if Jr < vals[0][0]:

 xe = c + 1.5*(xr - c)
 Je, Me = eval_alpha(xe)
 if Je < Jr:
 simplex[1], vals[1] = xe, (Je, Me)
 else:
 simplex[1], vals[1] = xr, (Jr, Mr)
 elif Jr < vals[1][0]:
 simplex[1], vals[1] = xr, (Jr, Mr)
 else:

 xc = c + 0.5*(simplex[1] - c)
 Jc, Mc = eval_alpha(xc)
 if Jc < vals[1][0]:
 simplex[1], vals[1] = xc, (Jc, Mc)
 else:

 simplex[1] = c + 0.5*(simplex[1] - c)
 vals[1] = eval_alpha(simplex[1])


 order = sorted(zip(simplex, vals), key=lambda x: x[1][0])
 a_best, (J_best, M_best) = order[0]
 return {"alpha": float(a_best), "J": float(J_best), **M_best}


import json, time, sys
from build_S import halo_S, void_S, aniso_S


R_TARGET = 0.60266
TOL_R = 0.01
MAX_KS = 0.03
MAX_HERM = 1e-10

def meets_point_rules(metrics):
 return (
 abs(metrics["r_bulk"] - R_TARGET) <= TOL_R and
 metrics["ks_gue_local"] <= MAX_KS and
 metrics["ks_gue_local"] < metrics.get("ks_gue_global", 1.0) and
 metrics["sym_err"] <= MAX_HERM
 )

def run_metrics_field(field, N, alpha, k, win):
 if field == "halo":
 S = halo_S(N=N, L=100.0)
 elif field == "void":
 S = void_S(N=N, L=100.0)
 elif field == "aniso":
 S = aniso_S(N=N, L=100.0)
 else:
 raise ValueError("field must be halo|void|aniso")
 t0 = time.time()
 m = eval_spectrum(S, alpha=alpha, k=k, bulk_lo=0.10, bulk_hi=0.90, local_win=win)
 m["elapsed_s"] = float(time.time() - t0)
 m["field"] = field; m["N"] = int(N); m["k"] = int(k); m["win"] = int(win)
 m["pass"] = bool(meets_point_rules(m))
 return m

def verify_all(alpha, quick=False):
 fields = ["halo", "void", "aniso"]
 if quick:
 Ns = [128]
 wins = [51]
 ks = [600]
 else:
 Ns = [128, 160, 192]
 wins = [31, 51, 71]
 ks = [400, 600]

 results = []
 for f in fields:
 for N in Ns:
 for w in wins:
 for k in ks:
 r = run_metrics_field(f, N, alpha, k, w)
 print(f"[verify] {f:5s} N={N:3d} k={k:3d} win={w:2d} "
 f"r~={r['r_bulk']:.5f} KS_loc={r['ks_gue_local']:.4f} "
 f"KS_glob={r['ks_gue_global']:.4f} Herm={r['sym_err']:.2e} "
 f"t={r['elapsed_s']:.1f}s {'PASS' if r['pass'] else 'FAIL'}")
 results.append(r)
 passes = sum(int(r["pass"]) for r in results)
 total = len(results)
 return results, passes, total

def save_report(alpha, results, passes, total, tag=""):
 Path("artifacts").mkdir(exist_ok=True)
 stamp = time.strftime("%Y%m%d_%H%M%S")
 report = {
 "locked_alpha": float(alpha),
 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
 "criteria": {
 "r_target": R_TARGET, "tol_r": TOL_R,
 "max_ks_local": MAX_KS, "max_herm": MAX_HERM,
 "require_ks_gue_lt_global": True
 },
 "summary": {"passes": passes, "total": total},
 "results": results
 }
 json_path = Path("artifacts")/f"LOCKED_REPORT{('_'+tag) if tag else ''}_{stamp}.json"
 with open(json_path, "w") as f:
 json.dump(report, f, indent=2)

 verdict = "YES" if passes == total else "NO"
 txt_path = Path("artifacts")/f"RESULT{('_'+tag) if tag else ''}.txt"
 with open(txt_path, "w") as f:
 f.write(f"RIEMANN_RULES_MET={verdict} (passes={passes}/{total}) alpha={alpha}\n")
 print(f"\nSaved {json_path}")
 print(f"Saved {txt_path}")
 return verdict, str(json_path), str(txt_path)


if __name__ == "__main__":

 for a in [1e-3, 1e-2, 5e-2]:
 S = halo_S(N=128, L=100.0)
 m = eval_spectrum(S, alpha=a, k=300, bulk_lo=0.10, bulk_hi=0.90, local_win=51)
 print(f"[spot] a={a:.2e} -> r~={m['r_bulk']:.5f} KS_loc={m['ks_gue_local']:.4f} KS_glob={m['ks_gue_global']:.4f}")


 seeds = coarse_search()
 print("\nTop coarse seeds:")
 for J, rec in seeds:
 print(rec)


 best = None
 for _, rec in seeds:
 cand = nelder_mead_alpha(rec["alpha"], N=128, iters=18)
 if (best is None) or (cand["J"] < best["J"]):
 best = cand

 print("\n=== LOCKED (copy these) ===")
 print({k:best[k] for k in ["alpha","r_bulk","ks_gue_local","ks_gue_global","n_spacings","J"]})


 quick_mode = False
 results, passes, total = verify_all(best["alpha"], quick=quick_mode)

 verdict, jpath, tpath = save_report(best["alpha"], results, passes, total, tag="FULL" if not quick_mode else "QUICK")

 print(f"\nRIEMANN_RULES_MET={verdict} (passes={passes}/{total}) alpha={best['alpha']}\n")


 sys.exit(0 if verdict == "YES" else 1)