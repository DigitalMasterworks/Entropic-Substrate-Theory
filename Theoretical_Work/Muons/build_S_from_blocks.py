#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, csv, math
import numpy as np

C = 299_792_458.0
G = 6.67430e-11
M_E = 5.9722e24
R_E = 6_371_000.0

def earth_phi(alt_m: float) -> float:
    r = R_E + alt_m
    return -G * M_E / r

def ring_points(R: float, n: int):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = R*np.cos(th); y = R*np.sin(th); z = np.zeros_like(x)
    return th, x, y, z

def read_blocks(path: str):
    blocks = []
    if not (path and os.path.exists(path)):
        return blocks
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].strip().startswith("#"): continue
            x,y,z,dx,dy,dz,dens = map(float, row)
            m = dens * dx * dy * dz
            blocks.append((x,y,z,m))
    return blocks

def phi_blocks(xr, yr, zr, blocks):
    phi = np.zeros_like(xr, dtype=float)
    for (xb,yb,zb,mb) in blocks:
        dx = xr - xb; dy = yr - yb; dz = zr - zb
        r = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9  # softening to avoid div/0
        phi += -G*mb/r
    return phi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring-radius", type=float, default=7.112, help="ring radius (m)")
    ap.add_argument("--n", type=int, default=3600, help="samples around ring")
    ap.add_argument("--alt-m", type=float, default=228.0, help="site altitude (m)")
    ap.add_argument("--blocks", type=str, default="env_blocks.csv", help="CSV of local mass blocks (optional)")
    ap.add_argument("--out", type=str, default="outputs/S_real.npy")
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)

    th, xr, yr, zr = ring_points(args.ring_radius, args.n)
    phi_E = earth_phi(args.alt_m)             # constant around ring
    blocks = read_blocks(args.blocks)         # optional
    phi_B = phi_blocks(xr, yr, zr, blocks) if blocks else np.zeros_like(xr)

    phi_tot = phi_E + phi_B                   # Newtonian potential [J/kg]
    S = 1.0 + (phi_tot / (C**2))              # S ≈ 1 + Φ/c^2
    np.save(args.out, S)

    print(f"Wrote {args.out}")
    print(f"  Earth Φ/c^2 contribution: {phi_E/(C**2):+.3e}")
    if blocks:
        print(f"  Blocks: {len(blocks)} (total potential span from blocks: {(S.max()-S.min()):.3e})")
    print(f"  S stats: mean={S.mean():.12f}  min={S.min():.12f}  max={S.max():.12f}")

if __name__ == "__main__":
    main()