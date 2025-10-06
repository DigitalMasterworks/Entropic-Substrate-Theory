#!/usr/bin/env python3


import numpy as np, glob, os, time

merged = "../eigs_1d/eigs_merged.npy"
wins = sorted(glob.glob("../eigs_1d/eigs_mu_*.npy"), key=os.path.getmtime, reverse=True)

def ts(p):
 try:
 return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
 except:
 return "?"

print("== Mister Stats ==")

if os.path.exists(merged):
 try:
 E = np.load(merged)
 E = E[np.isfinite(E)]
 print(f"[merged] unique={E.size:,} min={E.min():.6g} max={E.max():.6g}")
 except Exception as e:
 print(f"[merged] failed to load: {e}")
else:
 print("[merged] MISSING")

print(f"[windows] count={len(wins)}")
for p in wins[:5]:
 try:
 e = np.load(p)
 n = e[np.isfinite(e)].size
 print(f" {p} {ts(p)} n={n}")
 except Exception as ex:
 print(f" {p} {ts(p)} (read fail: {ex})")