#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthesis tools for the substrate g−2 program.

Generates:
  1) S_field_ring.npy   — 1D array: S(θ) sampled around a ring (θ∈[0,2π))
  2) Rs_data.csv        — synthetic R(s) for HVP testing

Ricci-flow update and time-scaling follow the substrate law:
  S_{t+Δt} = S_t + Δt_eff * ΔS,  with  Δt_eff ∝ S    # Substrate Ricci Flow & tick law.  :contentReference[oaicite:3]{index=3}
The optical/weak-field map (S≈1+Φ/c²) and dτ = S dt motivate spin/clock couplings.     :contentReference[oaicite:4]{index=4}

E989 variables are the target for downstream analysis & ωa/ωp ratio extraction.           # :contentReference[oaicite:5]{index=5}
SM baseline/room used for judging δaμ magnitudes per WP25.                               # :contentReference[oaicite:6]{index=6}
"""

import argparse
import json
import math
import os
from typing import Tuple, Optional

import numpy as np

OUTDIR = os.path.abspath("./outputs")
os.makedirs(OUTDIR, exist_ok=True)

C = 299_792_458.0             # m/s
G = 6.67430e-11               # m^3 kg^-1 s^-2
M_EARTH = 5.9722e24           # kg
R_EARTH = 6_371_000.0         # m

# ---------------------------
# Helpers
# ---------------------------

def earth_surface_S() -> float:
    """
    Compute S ≈ 1 + Φ/c² at Earth surface (Φ = -GM/R).
    Returns a value slightly below 1 (since Φ<0).
    """
    phi = -G * M_EARTH / R_EARTH
    return 1.0 + phi / (C**2)

def save_npy(path: str, arr: np.ndarray) -> str:
    np.save(path, arr)
    return os.path.abspath(path)

def bilinear_sample(grid: np.ndarray, x: float, y: float) -> float:
    """
    Sample grid at float coords (x,y) with bilinear interpolation.
    grid shape: (Ny, Nx). x along columns, y along rows. 0 ≤ x < Nx, 0 ≤ y < Ny.
    """
    Ny, Nx = grid.shape
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = min(x0 + 1, Nx - 1); y1 = min(y0 + 1, Ny - 1)

    dx = x - x0; dy = y - y0
    v00 = grid[y0, x0]; v10 = grid[y0, x1]; v01 = grid[y1, x0]; v11 = grid[y1, x1]
    v0 = v00 * (1 - dx) + v10 * dx
    v1 = v01 * (1 - dx) + v11 * dx
    return float(v0 * (1 - dy) + v1 * dy)

# ---------------------------
# 1) S(θ) — ring samplers
# ---------------------------

def make_S_ring_uniform(n: int = 3600, S0: Optional[float] = None) -> np.ndarray:
    if S0 is None:
        S0 = earth_surface_S()  # default: Earth gravity at surface
    S = np.full(n, S0, dtype=np.float64)
    return S

def make_S_ring_multipole(n: int = 3600, base_S: Optional[float] = None,
                          harmonics: Tuple[Tuple[int, float, float], ...] = ((1, -1e-11, 0.0), (2, 5e-12, 1.0)),
                          seed: int = 0) -> np.ndarray:
    """
    Build S(θ) = base_S + sum_k a_k cos(kθ + φ_k) + small jitter.
      harmonics: tuple of (k, amplitude, phase)
    """
    if base_S is None:
        base_S = earth_surface_S()
    theta = np.linspace(0, 2*np.pi, n, endpoint=False, dtype=np.float64)
    S = np.full_like(theta, base_S)
    for (k, amp, ph) in harmonics:
        S += amp * np.cos(k * theta + ph)
    rng = np.random.default_rng(seed)
    S += rng.normal(0.0, 1e-12, size=theta.shape)  # tiny corrugation
    S = np.clip(S, 1e-12, 1.0)  # keep in (0,1]
    return S

def make_S_ring_from_ricci(flow_N: int = 256,
                           steps: int = 500,
                           dt: float = 0.05,
                           seeds: int = 4,
                           seed_S: float = 0.5,
                           S_floor: float = 1e-6,
                           ring_center: Tuple[float, float] = (128.0, 128.0),
                           ring_radius_px: float = 80.0,
                           ring_samples: int = 3600,
                           laplacian: str = "5pt",
                           rng_seed: int = 7) -> np.ndarray:
    """
    Mini 2D Ricci-flow synthesizer on an NxN grid:
      S <- S + (dt * S) * ΔS    with clamp to [S_floor, 1].
    Seeds: set some pixels to seed_S (collapse injection).
    Then sample along a circle to produce S(θ).
    """
    rng = np.random.default_rng(rng_seed)
    S = np.ones((flow_N, flow_N), dtype=np.float64)
    # sprinkle seed voxels
    for _ in range(seeds):
        y = rng.integers(0, flow_N); x = rng.integers(0, flow_N)
        S[y, x] = seed_S

    # Neumann boundary via copying edges
    def lap(S_arr: np.ndarray) -> np.ndarray:
        if laplacian == "5pt":
            up    = np.vstack([S_arr[0:1, :], S_arr[:-1, :]])
            down  = np.vstack([S_arr[1:, :],  S_arr[-1:, :]])
            left  = np.hstack([S_arr[:, 0:1], S_arr[:, :-1]])
            right = np.hstack([S_arr[:, 1:],  S_arr[:, -1:]])
            return (up + down + left + right - 4.0 * S_arr)
        elif laplacian == "9pt":
            # 9-point stencil
            padded = np.pad(S_arr, 1, mode="edge")
            kern = np.array([[1, 4, 1],
                             [4, -20, 4],
                             [1, 4, 1]], dtype=np.float64)
            out = np.zeros_like(S_arr)
            for i in range(S_arr.shape[0]):
                for j in range(S_arr.shape[1]):
                    block = padded[i:i+3, j:j+3]
                    out[i, j] = np.sum(block * kern)
            return out / 6.0
        else:
            raise ValueError("laplacian must be '5pt' or '9pt'")

    for _ in range(steps):
        L = lap(S)
        dt_eff = dt * S  # time-scaling: Δt_eff ∝ S   :contentReference[oaicite:7]{index=7}
        S = S + dt_eff * L
        S = np.clip(S, S_floor, 1.0)

    # ring sampling
    cx, cy = ring_center
    theta = np.linspace(0, 2*np.pi, ring_samples, endpoint=False)
    rr = ring_radius_px
    vals = []
    for th in theta:
        x = cx + rr * np.cos(th)
        y = cy + rr * np.sin(th)
        x = np.clip(x, 0.0, flow_N - 1.0)
        y = np.clip(y, 0.0, flow_N - 1.0)
        vals.append(bilinear_sample(S, x, y))
    S_ring = np.array(vals, dtype=np.float64)
    return S_ring

# ---------------------------
# 2) Rs(s) — synthetic hadronic ratio for HVP tests
# ---------------------------

def breit_wigner(s: np.ndarray, m: float, gamma: float, amp: float) -> np.ndarray:
    """
    Simple relativistic Breit–Wigner peak in s (GeV^2):  amp * (m^2 Γ^2) / ((s - m^2)^2 + m^2 Γ^2)
    """
    m2 = m * m
    return amp * (m2 * gamma * gamma) / ((s - m2)**2 + (m2 * gamma * gamma))

def make_Rs(default: bool = True,
            s_min: float = 0.1, s_max: float = 40.0, n: int = 5000,
            continuum_level: float = 2.5, continuum_turn_on: float = 4.0,
            seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a toy R(s) with a few resonances + pQCD-like continuum.
      s in GeV^2. sqrt(s) up to ~6.3 GeV by default.
    This is a placeholder to exercise the HVP pipeline; replace with real data when ready.
    """
    rng = np.random.default_rng(seed)
    s = np.linspace(s_min, s_max, n, dtype=np.float64)

    R = np.zeros_like(s)

    # a few illustrative resonances (masses in GeV, widths in GeV, amplitudes unitless)
    peaks = [
        (0.775, 0.149, 8.0),   # ρ(770)
        (0.782, 0.0085, 2.0),  # ω(782)
        (1.019, 0.0042, 2.5),  # φ(1020)
        (3.097, 0.000093, 15), # J/ψ
        (3.686, 0.0003, 6.0),  # ψ(2S)
    ]
    for (m, g, a) in peaks:
        R += breit_wigner(s, m, g, a)

    # Smooth continuum: step up after ~2 GeV (s~4), with slow rise
    cont = continuum_level * (1.0 / (1.0 + np.exp(-(np.sqrt(s) - math.sqrt(continuum_turn_on)) / 0.25)))
    # mild high-energy slope
    cont *= (1.0 + 0.05 * np.log1p(s))
    R += cont

    # small random ripple to avoid a perfectly smooth function
    R *= 1.0 + rng.normal(0.0, 0.005, size=R.shape)

    R = np.clip(R, 0.0, None)
    return s, R

def save_Rs_csv(path: str, s: np.ndarray, R: np.ndarray) -> str:
    with open(path, "w") as f:
        f.write("s,R\n")
        for si, Ri in zip(s, R):
            f.write(f"{si:.8f},{Ri:.8f}\n")
    return os.path.abspath(path)

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(prog="substrate_synth")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Uniform S
    pU = sub.add_parser("s-uniform", help="Emit S_field_ring.npy for uniform S (Earth default if not specified).")
    pU.add_argument("--n", type=int, default=3600)
    pU.add_argument("--S0", type=float, default=None)
    pU.add_argument("--out", type=str, default=os.path.join(OUTDIR, "S_field_ring.npy"))

    # Multipole S
    pM = sub.add_parser("s-multipole", help="Emit S_field_ring.npy with cosine harmonics.")
    pM.add_argument("--n", type=int, default=3600)
    pM.add_argument("--base-S", dest="base_S", type=float, default=None)
    pM.add_argument("--harmonics", type=str, default="1:-1e-11:0.0,2:5e-12:1.0",
                    help="Comma list of k:amp:phase (phase in radians).")
    pM.add_argument("--seed", type=int, default=0)
    pM.add_argument("--out", type=str, default=os.path.join(OUTDIR, "S_field_ring.npy"))

    # Ricci S
    pR = sub.add_parser("s-ricci", help="Run a mini 2D Ricci-flow and sample a ring.")
    pR.add_argument("--N", type=int, default=256)
    pR.add_argument("--steps", type=int, default=500)
    pR.add_argument("--dt", type=float, default=0.05)
    pR.add_argument("--seeds", type=int, default=4)
    pR.add_argument("--seed-S", dest="seed_S", type=float, default=0.5)
    pR.add_argument("--S-floor", dest="S_floor", type=float, default=1e-6)
    pR.add_argument("--ring-cx", type=float, default=128.0)
    pR.add_argument("--ring-cy", type=float, default=128.0)
    pR.add_argument("--ring-r",  type=float, default=80.0)
    pR.add_argument("--ring-n",  type=int, default=3600)
    pR.add_argument("--laplacian", type=str, choices=["5pt","9pt"], default="5pt")
    pR.add_argument("--seed", type=int, default=7)
    pR.add_argument("--out", type=str, default=os.path.join(OUTDIR, "S_field_ring.npy"))

    # R(s)
    pH = sub.add_parser("make-Rs", help="Emit a synthetic R(s) CSV for HVP tests.")
    pH.add_argument("--s-min", type=float, default=0.1)
    pH.add_argument("--s-max", type=float, default=40.0)
    pH.add_argument("--n", type=int, default=5000)
    pH.add_argument("--cont", type=float, default=2.5)
    pH.add_argument("--turn-on", type=float, default=4.0)
    pH.add_argument("--seed", type=int, default=0)
    pH.add_argument("--out", type=str, default=os.path.join(OUTDIR, "Rs_data.csv"))

    args = ap.parse_args()

    if args.cmd == "s-uniform":
        S = make_S_ring_uniform(n=args.n, S0=args.S0)
        path = save_npy(args.out, S)
        print(json.dumps({"S_mean": float(S.mean()), "file": path}, indent=2))

    elif args.cmd == "s-multipole":
        harmonics = []
        if args.harmonics.strip():
            for spec in args.harmonics.split(","):
                k, amp, ph = spec.split(":")
                harmonics.append((int(k), float(amp), float(ph)))
        else:
            harmonics = tuple()
        S = make_S_ring_multipole(n=args.n, base_S=args.base_S, harmonics=tuple(harmonics), seed=args.seed)
        path = save_npy(args.out, S)
        print(json.dumps({"S_mean": float(S.mean()), "file": path, "harmonics": harmonics}, indent=2))

    elif args.cmd == "s-ricci":
        S = make_S_ring_from_ricci(flow_N=args.N, steps=args.steps, dt=args.dt, seeds=args.seeds,
                                   seed_S=args.seed_S, S_floor=args.S_floor,
                                   ring_center=(args.ring_cx, args.ring_cy),
                                   ring_radius_px=args.ring_r, ring_samples=args.ring_n,
                                   laplacian=args.laplacian, rng_seed=args.seed)
        path = save_npy(args.out, S)
        print(json.dumps({"S_mean": float(S.mean()), "S_min": float(S.min()),
                          "S_max": float(S.max()), "file": path}, indent=2))

    elif args.cmd == "make-Rs":
        s, R = make_Rs(s_min=args.s_min, s_max=args.s_max, n=args.n,
                       continuum_level=args.cont, continuum_turn_on=args.turn_on,
                       seed=args.seed)
        path = save_Rs_csv(args.out, s, R)
        print(json.dumps({"s_min": float(s.min()), "s_max": float(s.max()),
                          "rows": int(len(s)), "file": path}, indent=2))

if __name__ == "__main__":
    main()