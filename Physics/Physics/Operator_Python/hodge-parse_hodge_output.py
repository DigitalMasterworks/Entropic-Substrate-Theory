#!/usr/bin/env python3

import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fn = Path(sys.argv[1]) if len(sys.argv)>1 else Path("hodge_exact_output.txt")
txt = fn.read_text()



kv_lines = re.findall(r'^(.*N\s*=\s*\d.*)$', txt, flags=re.MULTILINE)
rows = []

def parse_kv(line):
 d = {}
 for k,v in re.findall(r'(\b[A-Za-z_]+)\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)', line):
 d[k] = float(v) if re.search(r'[0-9]', v) else v
 return d

if kv_lines:
 for L in kv_lines:
 d = parse_kv(L)
 if 'N' in d and ('error' in d or 'err' in d or 'E' in d):
 rows.append(d)


if not rows:

 for line in txt.splitlines():
 if re.search(r'\bN\b', line) or re.search(r'\bmode\b', line) or re.search(r'error|err|E', line, re.I):
 nums = re.findall(r'([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)', line)
 if len(nums) >= 2:

 try:
 N = int(float(nums[0]))
 err = float(nums[-1])
 rows.append({"N": N, "error": err})
 except:
 pass

if not rows:
 print("Could not auto-parse results. Paste a sample of the output (last 50 lines).")
 print("Or run: tail -n 200 hodge_exact_output.txt")
 sys.exit(1)

df = pd.DataFrame(rows)

for c in df.columns:
 if c.lower() in ('err','e'): df.rename(columns={c:'error'}, inplace=True)
 if c.lower()=='mode': df.rename(columns={c:'mode'}, inplace=True)

df = df.sort_values(['mode' if 'mode' in df.columns else 'N','N']).reset_index(drop=True)
df.to_csv("hodge_results.csv", index=False)
print("Wrote hodge_results.csv")
print(df)


modes = df['mode'].unique() if 'mode' in df.columns else [None]
for m in modes:
 sub = df if m is None else df[df['mode']==m]
 if len(sub) >= 2:
 x = np.log(sub['N'].astype(float))
 y = np.log(sub['error'].astype(float))
 slope, intercept = np.polyfit(x, y, 1)
 print(f"mode={m} empirical rate ≈ { -slope:.3f } (slope={slope:.3f})")


plt.figure()
if 'mode' in df.columns:
 for m in df['mode'].unique():
 s = df[df['mode']==m]
 plt.loglog(s['N'], s['error'], marker='o', label=f"mode {m}")
else:
 plt.loglog(df['N'], df['error'], marker='o')
plt.xlabel('N')
plt.ylabel('error')
plt.grid(True, which='both', ls=':')
if 'mode' in df.columns:
 plt.legend()
plt.title('Hodge FEM: error vs N (log-log)')
plt.tight_layout()
plt.savefig("hodge_error_vs_N.png")
print("Saved hodge_error_vs_N.png")
plt.show()