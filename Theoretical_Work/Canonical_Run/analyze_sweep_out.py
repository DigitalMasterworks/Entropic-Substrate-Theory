import numpy as np
import json, os, glob, re, sys
import pandas as pd

MIN_RATIO_THRESHOLD = 5.0
OUTPUT_DIR = "out"
OUTPUT_CSV = "high_ratio_sweep_results.csv"

all_ratios = []

print(f"--- Starting Global Ratio Sweep: Reporting all λk/λj > {MIN_RATIO_THRESHOLD} ---\n")

# Only analyze final run files
file_paths = glob.glob(os.path.join(OUTPUT_DIR, "*-run.json"))
if not file_paths:
    print(f"[ERROR] No *-run.json files found in '{OUTPUT_DIR}'.")
    sys.exit(0)

print(f"[INFO] Processing {len(file_paths)} JSON files...")

for filepath in file_paths:
    filename = os.path.basename(filepath)

    # 1) Load JSON once
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Skipping {filename}: could not read JSON ({e})")
        continue

    # 2) eps_e: prefer JSON meta, else parse from filename (_e0.88 or _e_0.88)
    eps_e = None
    eps_e = (data.get("meta", {}) or {}).get("eps_e", None)
    if eps_e is None:
        m = re.search(r"_e_?([0-9]+(?:\.[0-9]+)?)", filename)
        if m:
            try:
                eps_e = float(m.group(1))
            except ValueError:
                eps_e = None
    if eps_e is None:
        print(f"[WARN] Skipping {filename}: Could not determine eps_e.")
        continue

    # 3) eigenvalues array
    if "evals" not in data or not isinstance(data["evals"], list):
        print(f"[WARN] Skipping {filename}: no 'evals' array.")
        continue

    lambdas = [d.get("lambda") for d in data["evals"] if isinstance(d, dict) and "lambda" in d]
    if not lambdas:
        print(f"[WARN] Skipping {filename}: empty 'evals'.")
        continue

    # sort spectrum ascending
    eigenvalues = np.sort(np.array(lambdas, dtype=float))
    if eigenvalues[0] <= 1e-12:
        continue

    N = len(eigenvalues)
    # 4) collect all ratios > threshold
    for j in range(N):
        denom = eigenvalues[j]
        for k in range(j + 1, N):
            ratio = eigenvalues[k] / denom
            if ratio > MIN_RATIO_THRESHOLD:
                all_ratios.append({
                    "eps_e": eps_e,
                    "k_e": j + 1,      # 1-based
                    "k_mu": k + 1,     # 1-based
                    "ratio": ratio,
                    "lambda_1": eigenvalues[0],
                    "file": filename
                })

# 5) report
if not all_ratios:
    print("\n" + "="*80)
    print(f"GLOBAL RATIO SWEEP COMPLETE: Found 0 ratios > {MIN_RATIO_THRESHOLD}")
    print("="*80)
    sys.exit(0)

df = pd.DataFrame(all_ratios)
df_sorted = df.sort_values(by="ratio", ascending=False)
df_sorted.to_csv(OUTPUT_CSV, index=False, float_format="%.6e")

print("\n" + "="*80)
print(f"GLOBAL RATIO SWEEP COMPLETE: Found {len(all_ratios):,} ratios > {MIN_RATIO_THRESHOLD}")
print(f"Results saved to: {OUTPUT_CSV}")
print("="*80)

print("\n--- Top 10 Highest Ratios Found ---")
print(df_sorted.head(10).to_string(index=False))

print("\n--- Closest Matches to Muon/Electron (206.768283) ---")
df["diff"] = (df["ratio"] - 206.768283).abs()
df_muon = df.sort_values(by="diff").head(10)
print(df_muon[["eps_e","k_e","k_mu","ratio","diff","file"]].to_string(index=False))