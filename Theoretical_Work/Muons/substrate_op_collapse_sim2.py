#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
substrate_op_collapse_sim.py — Ultra-fast, stable operator+ECFM sandbox (no args).

This version avoids heavy/numerically delicate heat-kernel evaluation and instead uses a
robust local proxy for the cusp-operator spectral invariant along the ring:
    k(θ) = r^2*(1 + eps_B*cos 2θ + eps_E*sign cos θ)
    J(θ) ∝ k(θ)^(-β), normalized to ⟨J⟩ = 1
Then:
  • running λ(E) (sigmoid) calibrated to match ⟨ln S⟩_req at baseline
  • small finite-speed ECFM correction band on the ring
  • scenarios: baseline / half-beam / no-beam
Artifacts:
  - outputs/S_ring.npy
  - outputs/substrate_op_collapse_summary.md
  - outputs/substrate_op_collapse_summary.json
"""

import os, json, math, datetime, numpy as np

# -----------------------
# Config (fast & stable)
# -----------------------
R_OUT = 1.00
R_IN  = 0.72
N_THETA   = 360                   # ring sampling (azimuth)
BETA_J    = 1.0                   # exponent in J(θ) ∝ k(θ)^(-β), tune 0.5–1.5 if desired

# principal-symbol modulation (structure knobs)
EPS_B = 0.05                      # m=2 dent
EPS_E = 0.04                      # π-periodic E-plate pattern

# required average lnS from g-2 pipeline
LN_S_REQUIRED = 2.144e-6

# running λ(E) (sigmoid)
GAMMA_BASE = 29.3
B_BASE_T   = 1.45
P_RUN      = 2.0
E_STAR_FRAC= 0.50

# ECFM ring collapse (tiny dynamic correction)
ECFM_T     = 80
ECFM_D     = 0.20
ECFM_ALPHA = 0.85
ECFM_SINK_POS = 0.0
ECFM_SINK_WIDTH_FRAC = 0.04
ECFM_GAMMA = 0.08
ECFM_DT    = 1.0
ECFM_INIT_S= 1.5e-7

OUTDIR = "outputs"

# -----------------------
# Helpers
# -----------------------
def energy_scale(gamma, B_T):
    # muon-rest-frame EM energy proxy ~ γ^2 B^2 (dimensionless up to constant)
    return (gamma**2) * (B_T**2)

def lambda_sigmoid(E, lam_max, E_star, p):
    E = max(E, 1e-30)
    return lam_max / (1.0 + (E_star/E)**p)

def ecfm_ring_delta_lnS(Ntheta, steps, D, alpha, sink_pos, sink_wfrac, Gamma, dt, init_amp, rng=np.random):
    """
    1-D ring S(t,θ) with finite-speed-ish local updates:
      S_{t+dt} = S + dt * D*(1 - α C)*Δθ^2 S - dt * Γ(θ) S,  C=1-S
    returns small time-averaged δlnS(θ) over last third of steps (centered to zero mean)
    """
    S = init_amp * (1.0 + 0.1*rng.standard_normal(Ntheta))
    theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False)
    w = sink_wfrac * 2*np.pi
    def angdiff(a,b): d=a-b; d=(d+np.pi)%(2*np.pi)-np.pi; return d
    dist = np.abs(angdiff(theta, sink_pos))
    sink = (dist <= 0.5*w).astype(float)
    def lap1(u): return np.roll(u,-1) - 2*u + np.roll(u,1)
    acc = np.zeros_like(S); count = 0
    for t in range(steps):
        C = 1.0 - S
        f = (1.0 - alpha * C)
        S = S + dt*( D * f * lap1(S) - Gamma * sink * S )
        S = np.clip(S, 0.0, 1.0)
        if t >= int(steps*(2.0/3.0)):
            acc += S; count += 1
    Savg = acc/max(count,1)
    delta = 1.0e-7 * (Savg - np.mean(Savg))  # tiny, zero-mean
    return delta

# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    now = datetime.datetime.utcnow().isoformat() + "Z"

    # Ring geometry
    r_mid = 0.5*(R_IN + R_OUT)
    theta = np.linspace(-np.pi, np.pi, N_THETA, endpoint=False)

    # Principal symbol k(θ) and invariant proxy J(θ) ∝ k(θ)^(-β)
    k_theta = r_mid**2 * (1.0 + EPS_B*np.cos(2*theta) + EPS_E*np.sign(np.cos(theta)))
    k_theta = np.maximum(k_theta, 1e-8)       # safety
    J_theta = k_theta**(-BETA_J)
    # normalize to ⟨J⟩ = 1 to keep λ calibration simple and robust
    J_theta /= float(np.mean(J_theta))
    J_mean = float(np.mean(J_theta))          # = 1.0 by construction

    # Running λ(E): calibrate lam_max so baseline ⟨lnS⟩ matches required
    E_base = energy_scale(GAMMA_BASE, B_BASE_T)
    E_star = E_STAR_FRAC * E_base
    # lnS_req = [lam_max/(1+(E_star/E_base)^p)] * ⟨J⟩  → lam_max = lnS_req * (1+(E_star/E_base)^p)
    lam_max = LN_S_REQUIRED * (1.0 + (E_star/E_base)**P_RUN)
    def lam_of_E(E): return lambda_sigmoid(E, lam_max, E_star, P_RUN)

    # ECFM dynamic tiny correction
    delta_lnS = ecfm_ring_delta_lnS(
        Ntheta=N_THETA, steps=ECFM_T, D=ECFM_D, alpha=ECFM_ALPHA,
        sink_pos=ECFM_SINK_POS, sink_wfrac=ECFM_SINK_WIDTH_FRAC,
        Gamma=ECFM_GAMMA, dt=ECFM_DT, init_amp=ECFM_INIT_S
    )

    # Baseline ring profile to feed precession harness
    lnS_base  = lam_of_E(E_base) * J_theta + delta_lnS
    np.save(os.path.join(OUTDIR,"S_ring.npy"), np.exp(lnS_base))  # S = e^{ln S}

    # Scenarios (means only; no arrays in JSON)
    def scenario(E, label):
        lamE   = lam_of_E(E)
        lnS_op = lamE * J_theta
        lnS_tot= lnS_op + delta_lnS
        return dict(
            label=label,
            E=float(E),
            lamE=float(lamE),
            J_mean=float(J_mean),
            lnS_op_mean=float(np.mean(lnS_op)),
            lnS_dyn_mean=float(np.mean(delta_lnS)),
            lnS_tot_mean=float(np.mean(lnS_tot)),
            lnS_required=float(LN_S_REQUIRED),
            gap=float(LN_S_REQUIRED - float(np.mean(lnS_tot)))
        )

    scen_base = scenario(E_base, "baseline (full beam)")
    scen_half = scenario(0.5*E_base, "half-beam (0.5·E_base)")
    scen_off  = scenario(0.0, "no-beam (E=0)")

    # Console summary
    print("\n=== Ultra-fast Substrate Operator + ECFM Collapse ===")
    print(f"Generated: {now}")
    print(f"Ring r_mid={r_mid:.3f}, Nθ={N_THETA}, β={BETA_J}")
    print(f"Running λ(E): sigmoid, p={P_RUN:.2f}, E_base={E_base:.3e}, E_star={E_star:.3e}, lam_max={lam_max:.6e}")
    for s in (scen_base, scen_half, scen_off):
        print(f"  - {s['label']}")
        print(f"      λ(E)={s['lamE']:.6e}  lnS_op={s['lnS_op_mean']:.6e}  lnS_dyn={s['lnS_dyn_mean']:.6e}")
        print(f"      lnS_tot={s['lnS_tot_mean']:.6e}  required={s['lnS_required']:.6e}  gap={s['gap']:.6e}")

    # MD + JSON
    md = []
    md.append("# Ultra-fast Substrate Operator + ECFM Collapse — Summary")
    md.append(f"_Generated: {now}_\n")
    md.append(f"- Ring: r_mid={r_mid:.3f}, Nθ={N_THETA}, β={BETA_J}")
    md.append(f"- Running λ(E): sigmoid, p={P_RUN}, E_base={E_base:.3e}, E_star={E_star:.3e}, lam_max={lam_max:.6e}\n")
    md.append("| Scenario | λ(E) | ⟨lnS⟩_op | ⟨lnS⟩_dyn | ⟨lnS⟩_tot | ⟨lnS⟩_req | gap |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in (scen_base, scen_half, scen_off):
        md.append(f"| {s['label']} | {s['lamE']:.6e} | {s['lnS_op_mean']:.6e} | {s['lnS_dyn_mean']:.6e} "
                  f"| {s['lnS_tot_mean']:.6e} | {s['lnS_required']:.6e} | {s['gap']:.6e} |")
    os.makedirs(OUTDIR, exist_ok=True)
    open(os.path.join(OUTDIR,"substrate_op_collapse_summary.md"),"w").write("\n".join(md)+"\n")

    blob = dict(
        generated=now,
        ring=dict(r_mid=r_mid, Ntheta=N_THETA, beta=BETA_J),
        running=dict(model="sigmoid", p=P_RUN,
                     E_base=E_base, E_star_frac=E_STAR_FRAC, lam_max=lam_max),
        J_mean=float(J_mean),
        scenarios=dict(baseline=scen_base, half=scen_half, no_beam=scen_off),
        files=dict(S_ring="outputs/S_ring.npy",
                   md="outputs/substrate_op_collapse_summary.md")
    )
    open(os.path.join(OUTDIR,"substrate_op_collapse_summary.json"),"w").write(json.dumps(blob, indent=2))
    print("\nWrote: outputs/S_ring.npy, outputs/substrate_op_collapse_summary.md, outputs/substrate_op_collapse_summary.json\n")

if __name__ == "__main__":
    main()