#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
operator_probe.py — Cusp-operator sandbox for the g−2 substrate mapping (no args).

Builds the divergence-form cusp operator
    H = -∇·(k ∇),   with  k(x,y) = r^2 * (1 + ε_B b(θ) + ε_E e(θ))
on an annulus (ring cross-section), computes a spectral invariant
    J = (1/|Ω|) tr[ exp(-τ H) ]
via Hutchinson trace + expm_multiply, calibrates a running coupling λ(E)
from the baseline to match <ln S>_req = 2.144e-6, and predicts ΔlnR for:
  - baseline (uniform),
  - B-dent (m=2 azimuthal modulation),
  - E-plates (π-periodic alternation),
  - no-beam (E=0 → λ(E)=0 by construction).

Outputs:
  - prints a short table,
  - writes outputs/operator_probe_summary.md and .json
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import math, datetime, json, os

# -----------------------------
# User-tunable (kept internal)
# -----------------------------

# Geometry (annulus in a square grid)
R_out = 1.00          # outer radius (grid units)
R_in  = 0.72          # inner radius  (grid units) -> ring thickness ~ 0.28
N     = 180           # grid points per side (square [-R_out, R_out]^2)
tau_filter = 5.0      # heat-kernel time for exp(-τH) (dimensionless)
probes = 16           # Hutchinson probes for trace estimate
rng_seed = 7

# Required substrate shift (from pipeline)
lnS_required = 2.144e-6

# Scenario amplitudes (small dimensionless modulations of the principal symbol)
eps_B = 0.08  # amplitude for B-dent (m=2 azimuthal)
eps_E = 0.06  # amplitude for E-plate pattern (π-periodic)

# ---- Running-coupling settings (new) ----
RUN_MODEL    = "sigmoid"   # choose: "sigmoid" or "power"
gamma_base   = 29.3        # muon γ in g−2 ring
B_base_T     = 1.45        # ring B-field [Tesla]
p_run        = 2.0         # slope of the running
E_star_frac  = 0.50        # threshold as fraction of baseline energy scale

def energy_scale(gamma, B_T):
    """Lorentz-sensible local energy scale proxy ~ γ^2 B^2 (dimensionless up to a constant)."""
    return (gamma**2) * (B_T**2)

def lambda_sigmoid(E, lam_max, E_star, p):
    """λ(E) = lam_max / (1 + (E_star/E)^p), saturates at lam_max for large E."""
    E = max(E, 1e-30)
    return lam_max / (1.0 + (E_star / E)**p)

def lambda_power(E, lam0, E_star, p):
    """λ(E) = 0 for E<E*, and λ0 * (E/E*)^p for E>=E* (no saturation)."""
    if E < E_star:
        return 0.0
    return lam0 * (E / E_star)**p

# -----------------------------
# Helpers: grid, mask, radii
# -----------------------------

def build_grid(N, Rout):
    # square grid [-Rout, Rout]^2 with step h
    xs = np.linspace(-Rout, Rout, N)
    ys = np.linspace(-Rout, Rout, N)
    h  = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    R = np.sqrt(X*X + Y*Y)
    TH = np.arctan2(Y, X)  # [-π, π]
    return xs, ys, X, Y, R, TH, h

def annulus_mask(R, Rin, Rout):
    return (R >= Rin) & (R <= Rout)

# -----------------------------
# Divergence-form FD assembly
# H u = -∇·(k ∇u), here k = r^2 * (1 + eps_B*b(θ) + eps_E*e(θ))
# Flux at face uses arithmetic average of k on the two cells.
# Dirichlet zero outside annulus (mask).
# -----------------------------

def assemble_operator(R, TH, mask, h, epsB=0.0, epsE=0.0):
    """
    Build sparse SPD operator H on the masked nodes (Dirichlet outside).
    """
    Ny, Nx = R.shape
    # principal symbol k(x,y)
    b_theta = np.cos(2*TH)              # B "dent" (m=2)
    e_theta = np.sign(np.cos(TH))       # crude plate-like alternating regions
    k = (R**2) * (1.0 + epsB*b_theta + epsE*e_theta)
    k = np.clip(k, 1e-6, None)          # avoid degenerate faces

    # Map (i,j) -> linear index over mask
    idx = -np.ones(R.shape, dtype=int)
    coords = np.argwhere(mask)
    for n, (j,i) in enumerate(coords):  # argwhere returns (row=j, col=i)
        idx[j,i] = n
    M = coords.shape[0]

    data, rows, cols = [], [], []
    inv_h2 = 1.0/(h*h)

    def add_entry(p, q, val):
        rows.append(p); cols.append(q); data.append(val)

    for n, (j,i) in enumerate(coords):
        diag = 0.0
        for dj, di in [(0,1), (0,-1), (1,0), (-1,0)]:
            jj, ii = j+dj, i+di
            if jj<0 or jj>=Ny or ii<0 or ii>=Nx:
                continue
            if not mask[jj,ii]:
                continue
            k_face = 0.5*(k[j,i] + k[jj,ii])
            w = k_face * inv_h2
            q = idx[jj,ii]
            add_entry(n, q, -w)
            diag += w
        add_entry(n, n, diag)
    H = sp.csr_matrix((data, (rows, cols)), shape=(M, M))
    return H, idx

# -----------------------------
# Spectral invariant via Hutchinson
# J = (1/|Ω|) tr[ exp(-τ H) ]
# -----------------------------

def estimate_J(H, area, tau, probes=16, rng=None):
    """
    Hutchinson trace estimator with Rademacher probes:
        tr[f(H)] ≈ (1/m) Σ v^T f(H) v, v_i=±1
    Here f(H) = exp(-τ H) and we apply with expm_multiply.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    m = H.shape[0]
    acc = 0.0
    for _ in range(probes):
        v = rng.choice([-1.0, 1.0], size=m).astype(float)
        y = expm_multiply((-tau)*H, v)  # exp(-τH) v
        acc += float(v @ y)
    tr_est = acc / probes
    return tr_est / area

# -----------------------------
# Scenario builder
# -----------------------------

def run_scenario(epsB, epsE, label, X, Y, R, TH, mask, h, tau, probes, rng):
    H, idx = assemble_operator(R, TH, mask, h, epsB=epsB, epsE=epsE)
    area = float(np.sum(mask)) * (h*h)
    J = estimate_J(H, area, tau, probes=probes, rng=rng)
    return dict(label=label, J=J, M=H.shape[0], area=area)

# -----------------------------
# Main
# -----------------------------

def main():
    now = datetime.datetime.utcnow().isoformat() + "Z"
    xs, ys, X, Y, R, TH, h = build_grid(N, R_out)
    mask = annulus_mask(R, R_in, R_out)
    rng = np.random.default_rng(rng_seed)

    # Build scenarios FIRST so base["J"] exists
    base  = run_scenario(0.0, 0.0, "baseline",  X, Y, R, TH, mask, h, tau_filter, probes, rng)
    bdent = run_scenario(eps_B, 0.0, "B-dent (m=2)", X, Y, R, TH, mask, h, tau_filter, probes, rng)
    eplat = run_scenario(0.0, eps_E, "E-plates",     X, Y, R, TH, mask, h, tau_filter, probes, rng)

    # ---- Running-coupling calibration (new) ----
    E_base = energy_scale(gamma_base, B_base_T)
    E_star = E_star_frac * E_base

    if RUN_MODEL.lower() == "sigmoid":
        # lnS_req = λ(E_base) * J_base, with λ(E) = lam_max / (1 + (E_star/E)^p)
        lam_max = (lnS_required / base["J"]) * (1.0 + (E_star / E_base)**p_run)
        def lam_of_E(E):
            return lambda_sigmoid(E, lam_max, E_star, p_run)
        lam_params = dict(kind="sigmoid", lam_max=lam_max, E_star=E_star, p=p_run)
    elif RUN_MODEL.lower() == "power":
        # lnS_req = λ(E_base) * J_base, with λ(E) = lam0 * (E/E_star)^p  (E>=E_star)
        lam0 = (lnS_required / base["J"]) / max((E_base / E_star)**p_run, 1e-30)
        def lam_of_E(E):
            return lambda_power(E, lam0, E_star, p_run)
        lam_params = dict(kind="power", lam0=lam0, E_star=E_star, p=p_run)
    else:
        raise ValueError("RUN_MODEL must be 'sigmoid' or 'power'")

    # Helper: predict lnS & ΔlnR for a scenario at energy scale E
    def predict(scen, E):
        lamE = lam_of_E(E)
        lnS  = lamE * scen["J"]
        return dict(label=scen["label"], J=scen["J"], lnS=lnS, dlnR=lnS, lamE=lamE, E=E)

    # baseline-scale predictions (same E for baseline, dent, plates)
    p_base   = predict(base,  E_base)
    p_bd     = predict(bdent, E_base)
    p_ep     = predict(eplat, E_base)
    # strict no-beam (MRI-like) check: E=0 → λ=0
    p_nobeam = predict(base, 0.0); p_nobeam["label"] = "no-beam (E=0)"

    rows = [p_base, p_bd, p_ep, p_nobeam]

    # Pretty print
    print("\n=== Cusp-operator sandbox ===")
    print(f"Generated: {now}")
    print(f"Grid: N={N}  annulus Rin={R_in:.3f}, Rout={R_out:.3f}, area≈{base['area']:.6f}")
    print(f"Heat-kernel τ={tau_filter:.3f}, probes={probes}")
    print(f"Running model: {lam_params['kind']},  p={lam_params['p']:.2f},  E_base={E_base:.3e},  E_star={E_star:.3e}")
    if lam_params['kind']=="sigmoid":
        print(f"  lam_max = {lam_params['lam_max']:.6e}")
    else:
        print(f"  lam0    = {lam_params['lam0']:.6e}")
    print(f"Calibrated at baseline: J_base={base['J']:.6e}  →  ⟨lnS⟩_req={lnS_required:.6e}")
    print("\nScenario results:")
    for p in rows:
        print(f"  - {p['label']:<14}  J={p['J']:.6e}   λ(E)={p['lamE']:.6e}   lnS_pred={p['lnS']:.6e}   ΔlnR={p['dlnR']:.6e}")

    # Write a compact summary
    md = []
    md.append("# Cusp-Operator Sandbox Results")
    md.append(f"_Generated: {now}_\n")
    md.append(f"- Grid: N={N}, annulus Rin={R_in:.3f}, Rout={R_out:.3f}, area≈{base['area']:.6f}")
    md.append(f"- Heat-kernel τ={tau_filter:.3f}, probes={probes}")
    if lam_params['kind']=="sigmoid":
        md.append(f"- Running λ(E): sigmoid, p={p_run}, E_base={E_base:.3e}, E_star={E_star:.3e}, lam_max={lam_params['lam_max']:.6e}")
    else:
        md.append(f"- Running λ(E): power, p={p_run}, E_base={E_base:.3e}, E_star={E_star:.3e}, lam0={lam_params['lam0']:.6e}")
    md.append(f"- Baseline calibration: J_base={base['J']:.6e} → ⟨lnS⟩_req={lnS_required:.6e}\n")
    md.append("| Scenario | J | λ(E) | ⟨lnS⟩(pred) | ΔlnR |")
    md.append("|---|---:|---:|---:|---:|")
    for p in rows:
        md.append(f"| {p['label']} | {p['J']:.6e} | {p['lamE']:.6e} | {p['lnS']:.6e} | {p['dlnR']:.6e} |")
    os.makedirs("outputs", exist_ok=True)
    open("outputs/operator_probe_summary.md","w").write("\n".join(md)+"\n")

    blob = dict(
        generated=now,
        grid=dict(N=N, Rin=R_in, Rout=R_out, area=base["area"], h=float(2*R_out/(N-1))),
        filter_tau=tau_filter, probes=probes,
        lnS_required=lnS_required,
        running=lam_params,
        scenarios=rows
    )
    open("outputs/operator_probe_summary.json","w").write(json.dumps(blob, indent=2))
    print("\nWrote: outputs/operator_probe_summary.md, outputs/operator_probe_summary.json\n")

if __name__ == "__main__":
    main()
