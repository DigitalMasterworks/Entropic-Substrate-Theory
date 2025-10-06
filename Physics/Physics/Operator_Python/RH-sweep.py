#!/usr/bin/env python3











import argparse, itertools, os, re, subprocess, sys, time, csv, math
from pathlib import Path

HERE = Path(__file__).resolve().parent

def parse_list(s, typ=float):
 """
 Accepts:
 - comma list: "1,2,3"
 - range spec: "start:end:step" (inclusive-ish; step may be float)
 Returns a list[typ]
 """
 if s is None:
 return []
 s = s.strip()
 if ":" in s:
 a, b, c = s.split(":")
 a = typ(a); b = typ(b); c = typ(c)
 out = []
 x = a

 count = 0
 while (x <= b + (1e-12 if typ is float else 0)):
 out.append(typ(x))
 x = x + c
 count += 1
 if count > 100000:
 raise ValueError("range spec produced too many values")
 return out
 vals = [v for v in s.split(",") if v.strip()!= ""]
 return [typ(v) for v in vals]

def as_int_list(s): return parse_list(s, int)
def as_float_list(s):return parse_list(s, float)

def slug(v):
 return str(v).replace(".", "p").replace("-", "m")

def run_and_stream(cmd, run_log_path):
 run_log_path.parent.mkdir(parents=True, exist_ok=True)
 with open(run_log_path, "w", encoding="utf-8") as logf:
 proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
 lines = []
 for line in proc.stdout:
 sys.stdout.write(line)
 sys.stdout.flush()
 logf.write(line)
 logf.flush()
 lines.append(line.rstrip("\n"))
 ret = proc.wait()
 return ret, lines

SUMMARY_REGEXES = {
 "explicit": re.compile(r"explicit_formula_check\s*:\s*(PASS|WARN|FAIL)", re.I),
 "zeta": re.compile(r"dyn_zeta_vs_euler\s*:\s*(PASS|WARN|FAIL)", re.I),
 "heat": re.compile(r"heat_trace_coeffs\s*:\s*(PASS|WARN|FAIL).*?coeff_B\s*~\s*([-+eE0-9\.]+)", re.I),
 "weyl": re.compile(r"weyl_const_check\s*:\s*(PASS|WARN|FAIL).*?tail max \|D\|\s*([-+eE0-9\.]+)", re.I),
 "scatter": re.compile(r"scattering_grid\s*:\s*(PASS|WARN|FAIL).*?sup error\s*([-+eE0-9\.]+)", re.I),
 "all": re.compile(r"\[Phase II\]\s*ALL\s*(PASS|DONE.*)", re.I),
}

def parse_summary(lines):
 blob = "\n".join(lines[-400:])
 out = {"explicit":"", "zeta":"", "heat":"", "weyl":"", "scatter":"", "B":None, "Dtail":None, "sup":None, "overall":""}
 for key, rx in SUMMARY_REGEXES.items():
 m = rx.search(blob)
 if not m: continue
 if key == "heat":
 out["heat"] = m.group(1).upper()
 try: out["B"] = float(m.group(2))
 except: pass
 elif key == "weyl":
 out["weyl"] = m.group(1).upper()
 try: out["Dtail"] = float(m.group(2))
 except: pass
 elif key == "scatter":
 out["scatter"] = m.group(1).upper()
 try: out["sup"] = float(m.group(2))
 except: pass
 elif key == "all":
 out["overall"] = m.group(1).upper()
 else:
 out[key] = m.group(1).upper()
 return out

def main():
 ap = argparse.ArgumentParser(description="Parameter sweep driver for verify.py (Phase II).")
 ap.add_argument("--verify-script", default=str(HERE / "verify.py"))

 ap.add_argument("--N", default="128")
 ap.add_argument("--L", default="100.0")
 ap.add_argument("--k", default="2000")
 ap.add_argument("--eps", default="1e-3")
 ap.add_argument("--seed", default="1234")

 ap.add_argument("--Pmax", default="101")
 ap.add_argument("--PmaxEuler", default="200")
 ap.add_argument("--fft-sigma", default="6.0")
 ap.add_argument("--allow-depth", default="2")
 ap.add_argument("--sigma-euler", default="2.0")
 ap.add_argument("--anchor-euler", default="20.0")

 ap.add_argument("--t0-heat", default="0.0")
 ap.add_argument("--tail-bins", default="5")


 ap.add_argument("--logdir", default="sweep_logs")
 ap.add_argument("--cache-dir", default="eigs_cache")
 ap.add_argument("--print-cycle", type=int, default=1, help="print scoreboard every N runs")
 ap.add_argument("--max-combos", type=int, default=100000)
 ap.add_argument("--dry-run", action="store_true")
 args = ap.parse_args()


 N_list = as_int_list(args.N)
 L_list = as_float_list(args.L)
 k_list = as_int_list(args.k)
 eps_list = as_float_list(args.eps)
 seed_list = as_int_list(args.seed)

 Pmax_list = as_int_list(args.Pmax)
 PmaxEuler_list = as_int_list(args.PmaxEuler)
 fftsigma_list = as_float_list(args.fft-sigma) if False else as_float_list(args.__dict__["fft-sigma"])
 allowdepth_list = as_int_list(args.__dict__["allow-depth"])
 sigmaEuler_list = as_float_list(args.__dict__["sigma-euler"])
 anchorEuler_list= as_float_list(args.__dict__["anchor-euler"])

 t0heat_list = as_float_list(args.__dict__["t0-heat"])
 tailbins_list = as_int_list(args.__dict__["tail-bins"])

 combos = list(itertools.product(
 N_list, L_list, k_list, eps_list, seed_list,
 Pmax_list, PmaxEuler_list, fftsigma_list, allowdepth_list,
 sigmaEuler_list, anchorEuler_list, t0heat_list, tailbins_list
 ))

 if len(combos) == 0:
 print("No combos parsed. Check arguments."); sys.exit(1)
 if len(combos) > args.max_combos:
 print(f"Refusing to run {len(combos)} combos (> --max-combos). Narrow sweeps."); sys.exit(1)

 print(f" sweep starting: {len(combos)} combos")
 print(f"verify.py = {args.verify_script}")


 logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)
 cachedir = Path(args.cache_dir); cachedir.mkdir(parents=True, exist_ok=True)
 csv_path = logdir / "sweep_results.csv"
 with open(csv_path, "w", newline="", encoding="utf-8") as f:
 w = csv.writer(f)
 w.writerow(["run_idx","N","L","k","eps","seed","Pmax","PmaxEuler","fft_sigma","allow_depth","sigma_euler","anchor_euler","t0_heat","tail_bins",
 "explicit","zeta","heat","weyl","scatter","B","Dtail","sup","overall","seconds","logfile"])


 def eigs_key(N,L,k,eps,seed):
 return f"eigs_N{N}_L{slug(L)}_k{k}_e{slug(eps)}_s{seed}.npy"

 scoreboard = {"PASS":0,"WARN":0,"OTHER":0}
 t_global = time.time()

 for idx, (N,L,k,eps,seed,Pmax,PmaxEuler,fft_sigma,allow_depth,sigma_euler,anchor_euler,t0_heat,tail_bins) in enumerate(combos, start=1):
 key = eigs_key(N,L,k,eps,seed)
 eigs_path = cachedir / key
 logfile = logdir / f"run_{idx}_N{N}_L{slug(L)}_k{k}_P{Pmax}_fs{slug(fft_sigma)}_s{seed}.log"

 cmd = [sys.executable, args.verify_script,
 "--phase","2",
 "--N", str(N),
 "--L", str(L),
 "--k", str(k),
 "--eps", str(eps),
 "--seed", str(seed),
 "--Pmax", str(Pmax),
 "--PmaxEuler", str(PmaxEuler),
 "--fft-sigma", str(fft_sigma),
 "--allow-depth", str(allow_depth),
 "--sigma-euler", str(sigma_euler),
 "--anchor-euler", str(anchor_euler),
 "--t0-heat", str(t0_heat),
 "--tail-bins", str(tail_bins),
 "--logfile", str(logfile),
 ]
 if eigs_path.exists():
 cmd += ["--load-eigs", str(eigs_path)]
 else:
 cmd += ["--save-eigs", str(eigs_path)]

 print(f"\n─── Run {idx}/{len(combos)} • N={N} L={L} k={k} eps={eps} seed={seed} • Pmax={Pmax} fs={fft_sigma} σ={sigma_euler} anchor={anchor_euler} t0={t0_heat} bins={tail_bins}")
 print("cmd:", " ".join(cmd))

 if args.dry_run:
 continue

 t0 = time.time()
 ret, lines = run_and_stream(cmd, logfile)
 dt = time.time() - t0

 summary = parse_summary(lines)
 overall = summary.get("overall","")
 if "PASS" in overall:
 scoreboard["PASS"] += 1
 elif "WARN" in (summary.get("explicit",""), summary.get("zeta",""), summary.get("heat",""), summary.get("weyl",""), summary.get("scatter","")):
 scoreboard["WARN"] += 1
 else:
 scoreboard["OTHER"] += 1


 with open(csv_path, "a", newline="", encoding="utf-8") as f:
 w = csv.writer(f)
 w.writerow([idx,N,L,k,eps,seed,Pmax,PmaxEuler,fft_sigma,allow_depth,sigma_euler,anchor_euler,t0_heat,tail_bins,
 summary.get("explicit",""), summary.get("zeta",""), summary.get("heat",""), summary.get("weyl",""),
 summary.get("scatter",""), summary.get("B",""), summary.get("Dtail",""), summary.get("sup",""),
 overall, f"{dt:.1f}", str(logfile)])


 if (idx % max(1,args.print_cycle)) == 0:
 print(f"\n scoreboard after {idx} / {len(combos)}:")
 print(f" PASS: {scoreboard['PASS']} WARN: {scoreboard['WARN']} OTHER: {scoreboard['OTHER']}")
 print(f" elapsed: {time.time()-t_global:,.1f}s (last run {dt:.1f}s)")
 print(f" csv: {csv_path}")

 print(f"\n sweep complete. PASS={scoreboard['PASS']} WARN={scoreboard['WARN']} OTHER={scoreboard['OTHER']}")
 print(f"results: {csv_path} | logs: {logdir}")

if __name__ == "__main__":
 main()