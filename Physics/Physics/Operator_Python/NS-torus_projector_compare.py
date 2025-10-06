#!/usr/bin/env python3
"""
torus_projector_compare.py

Memory-safe version of the torus projector compare.
Uses low-rank formulas only (never forms dense dim x dim matrices).

Drop in and run locally:
 python3 torus_projector_safe.py

Change R_MAX to increase the truncation, but keep an eye on 'dim' reported.
"""
import numpy as np, json, math
from itertools import product
from pathlib import Path

OUT = Path(".")
R_MAX = 30
EPS = 1e-12

def estimate_dense_bytes(R):
 nmodes = (2*R + 1)**2
 dim = nmodes * 4
 entries = dim * dim
 bytes_needed = entries * 8
 return nmodes, dim, entries, bytes_needed


nmodes, dim, entries, bytes_needed = estimate_dense_bytes(R_MAX)
print(f"[info] R_MAX={R_MAX}, nmodes={nmodes}, dim={dim}")
GB = bytes_needed / (1024**3)
print(f"[info] forming a dense dim×dim matrix would require ~{GB:.2f} GB; this script WILL NOT do that.")


modes = []
for nx in range(-R_MAX, R_MAX+1):
 for ny in range(-R_MAX, R_MAX+1):
 k2 = nx*nx + ny*ny
 modes.append(((nx,ny), float(k2)))
modes.sort(key=lambda t: (t[1], t[0]))
nmodes = len(modes)
dim = nmodes * 4


k0_idx = None
for i, ((nx,ny),k2) in enumerate(modes):
 if int(k2) == 0:
 k0_idx = i
 break
if k0_idx is None:
 raise RuntimeError("No zero mode found in truncation (unexpected)")


r = 4
Qz = np.zeros((dim, r), dtype=float)
base_off = 4 * k0_idx
Qz[base_off + 0, 0] = 1.0
Qz[base_off + 1, 1] = 1.0
Qz[base_off + 2, 2] = 1.0
Qz[base_off + 3, 3] = 1.0


Qz, _ = np.linalg.qr(Qz)


Qb = np.zeros((dim, r), dtype=float)
Qb[base_off + 0, 0] = 1.0
Qb[base_off + 1, 1] = 1.0
Qb[base_off + 2, 2] = 1.0
Qb[base_off + 3, 3] = 1.0
Qb, _ = np.linalg.qr(Qb)


M = Qz.T @ Qb

normM2 = float(np.sum(np.abs(M)**2))


fro_sq = 2.0 * r - 2.0 * normM2
fro_norm = math.sqrt(max(0.0, fro_sq))


sv = np.linalg.svd(M, compute_uv=False)
sv_clamped = np.clip(sv, -1.0, 1.0)
angles = [math.acos(float(s)) for s in sv_clamped]
angles_deg = [float(a * 180.0 / math.pi) for a in angles]

report = {
 "R_MAX": R_MAX,
 "nmodes": int(nmodes),
 "dim": int(dim),
 "rank_r": int(r),
 "frobenius_norm_P_minus_C": float(fro_norm),
 "frobenius_norm_squared": float(fro_sq),
 "singular_values": [float(x) for x in sv_clamped],
 "principal_angles_radians": [float(x) for x in angles],
 "principal_angles_degrees": angles_deg
}

out_path = OUT / "torus_projector_safe_report.json"
with open(out_path, "w") as f:
 json.dump(report, f, indent=2)
print("Wrote", out_path.resolve().as_posix())
print(json.dumps(report, indent=2))