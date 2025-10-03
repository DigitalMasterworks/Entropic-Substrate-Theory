#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
substrate_op_collapse_sim.py — FAST operator + ECFM collapse sandbox (no args).

Speed & stability:
  • Chebyshev heat-kernel with time-splitting (|alpha|<=6) to avoid Bessel overflow
  • Block Hutchinson probes (no expm_multiply)
  • No NumPy arrays in JSON; baseline ring profile saved as outputs/S_ring.npy

Artifacts:
  - outputs/substrate_op_collapse_summary.md
  - outputs/substrate_op_collapse_summary.json
  - outputs/S_ring.npy
"""

import os, json, math, datetime, time
import numpy as np
import scipy.sparse as sp
from scipy.special import iv as besseli  # modified Bessel I_k

# -----------------------
# Global knobs (speed-quality)
# -----------------------
R_OUT = 1.00
R_IN  = 0.72

# Tunables (safe/fast defaults)
N_GRID    = 128             # square [-R_OUT, R_OUT]^2
N_THETA   = 360             # ring samples along r_mid
TAU_FILTER= 3.0             # heat-kernel time exp(-τH)  (smaller = safer)
PROBES    = 4               # Hutchinson probes (block size)
CHEB_K    = 24              # Chebyshev degree

RNG_SEED  = 7

# Principal symbol modulation (scenarios)
EPS_B = 0.05
EPS_E = 0.04

# Required average lnS from g-2 pipeline
LN_S_REQUIRED = 2.144e-6

# Running coupling λ(E) model (sigmoid)
GAMMA_BASE = 29.3
B_BASE_T   = 1.45
P_RUN      = 2.0
E_STAR_FRAC= 0.50

# ECFM ring-collapse params (1-D)
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
# Utilities
# -----------------------
def build_grid(N, Rout):
    xs = np.linspace(-Rout, Rout, N)
    ys = np.linspace(-Rout, Rout, N)
    h  = xs[1] - xs[0]
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    R = np.sqrt(X*X + Y*Y)
    TH = np.arctan2(Y, X)
    return xs, ys, X, Y, R, TH, h

def annulus_mask(R, Rin, Rout):
    return (R >= Rin) & (R <= Rout)

def assemble_operator(R, TH, mask, h, epsB=0.0, epsE=0.0):
    """
    Divergence-form face-centered flux assembly:
      H u = -∇·(k ∇u), with k = r^2 * (1 + epsB*cos 2θ + epsE*sign cos θ).
    Dirichlet outside annulus (no rows outside).
    """
    Ny, Nx = R.shape
    b_theta = np.cos(2*TH)
    e_theta = np.sign(np.cos(TH))
    k = (R**2) * (1.0 + epsB*b_theta + epsE*e_theta)
    k = np.clip(k, 1e-8, None)

    idx = -np.ones(R.shape, dtype=int)
    coords = np.argwhere(mask)
    for n,(j,i) in enumerate(coords):
        idx[j,i] = n
    M = coords.shape[0]

    data, rows, cols = [], [], []
    inv_h2 = 1.0/(h*h)

    def add(p,q,val):
        rows.append(p); cols.append(q); data.append(val)

    for n,(j,i) in enumerate(coords):
        diag = 0.0
        for dj,di in [(0,1),(0,-1),(1,0),(-1,0)]:
            jj,ii = j+dj, i+di
            if jj<0 or jj>=Ny or ii<0 or ii>=Nx: continue
            if not mask[jj,ii]:                 continue
            k_face = 0.5*(k[j,i] + k[jj,ii])
            w = k_face * inv_h2
            q = idx[jj,ii]
            add(n,q,-w)
            diag += w
        add(n,n,diag)

    H = sp.csr_matrix((data,(rows,cols)), shape=(M,M))
    return H, idx

# --- Power iteration for λ_max (cheap) ---
def power_lmax(H, iters=30, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    M = H.shape[0]
    v = rng.standard_normal(M)
    v /= np.linalg.norm(v)+1e-30
    lam = 0.0
    for _ in range(iters):
        w = H @ v
        lam = float(np.linalg.norm(w))
        v = w / (lam+1e-30)
    return lam

# --- Chebyshev one step (stable) ---
def cheb_step(H, V, tau_step, rng=None, kmax=24):
    """
    One Chebyshev step: Y = exp(-tau_step * H) @ V,
    using scaling H -> B in [-1,1], exp(alpha*B) expansion with |alpha| moderate.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M = H.shape[0]
    lmax = power_lmax(H, iters=20, rng=rng)
    lmax = max(lmax, 1e-12)
    s = lmax/2.0
    m = lmax/2.0
    alpha = -tau_step * s
    aabs = abs(alpha)

    Bop = (H * (2.0/lmax)) - sp.eye(M, format='csr')  # spectrum in [-1,1]

    # Modified Bessel I_k(|alpha|)
    c0 = besseli(0, aabs)
    coeff = [besseli(k, aabs) for k in range(1, kmax+1)]
    beta = math.exp(-tau_step * m)

    # Chebyshev recurrence
    T0 = V.copy()
    T1 = Bop @ V
    Y  = c0 * T0 + (2.0 * coeff[0] * T1 if kmax >= 1 else 0.0)
    Tk_minus2 = T0
    Tk_minus1 = T1
    for k in range(2, kmax+1):
        Tk = 2.0 * (Bop @ Tk_minus1) - Tk_minus2
        Y += 2.0 * coeff[k-1] * Tk
        Tk_minus2, Tk_minus1 = Tk_minus1, Tk

    Y *= beta
    if not np.isfinite(Y).all():
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    return Y

# --- Chebyshev exp(-τH) with time-splitting ---
def cheb_expm_apply(H, V, tau, kmax=24, rng=None):
    """
    Stable exp(-tau H) @ V via Chebyshev with time splitting.
    Enforce |alpha| = tau_step*(λmax/2) <= 6 to keep I_k(alpha) moderate.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lmax = power_lmax(H, iters=12, rng=rng)
    lmax = max(lmax, 1e-12)
    max_alpha = 6.0
    n_splits = max(1, int(math.ceil((tau * (lmax/2.0)) / max_alpha)))
    tau_step = tau / n_splits

    Y = V.copy()
    for _ in range(n_splits):
        Y = cheb_step(H, Y, tau_step, rng=rng, kmax=kmax)
        if not np.isfinite(Y).all():
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    return Y

# --- Block Hutchinson diagonal of exp(-τH) ---
def hutchinson_diag_exp_block(H, tau, probes=4, rng=None):
    """
    diag(exp(-τH)) ≈ mean over probes of v ∘ (exp(-τH) v), v_i ∈ {±1}.
    Block version with Chebyshev.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    M = H.shape[0]
    V = rng.choice([-1.0, 1.0], size=(M, probes)).astype(float)
    Y = cheb_expm_apply(H, V, tau, kmax=CHEB_K, rng=rng)
    est = np.mean(V * Y, axis=1)
    return est

def sample_ring_on_midradius(J_diag, idx, X, Y, R, TH, mask, Ntheta):
    r_mid = 0.5*(R_IN + R_OUT)
    thetas = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False)
    xs = r_mid*np.cos(thetas); ys = r_mid*np.sin(thetas)
    coords = np.argwhere(mask)
    ptsX = X[mask].ravel(); ptsY = Y[mask].ravel(); lin  = idx[mask].ravel()
    J_theta = np.zeros(Ntheta, dtype=float)
    for k,(xc,yc) in enumerate(zip(xs,ys)):
        dx = ptsX - xc; dy = ptsY - yc
        jmin = np.argmin(dx*dx + dy*dy)
        J_theta[k] = J_diag[ lin[jmin] ]
    return np.maximum(J_theta, 1e-16)

# Running coupling λ(E)
def energy_scale(gamma, B_T):  return (gamma**2) * (B_T**2)
def lambda_sigmoid(E, lam_max, E_star, p):
    E = max(E, 1e-30)
    return lam_max / (1.0 + (E_star/E)**p)

# 1-D ECFM collapse (fast)
def ecfm_ring_delta_lnS(Ntheta, steps, D, alpha, sink_pos, sink_wfrac, Gamma, dt, init_amp, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    S = init_amp * (1.0 + 0.1*rng.standard_normal(Ntheta))
    theta = np.linspace(-np.pi, np.pi, Ntheta, endpoint=False)
    w = sink_wfrac * 2*np.pi
    def angdiff(a,b):
        d = a-b; d = (d + np.pi)%(2*np.pi) - np.pi; return d
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
    delta_lnS = 1.0e-7 * (Savg - np.mean(Savg))
    return delta_lnS

# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    now = datetime.datetime.utcnow().isoformat() + "Z"
    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()

    # Grid + annulus operator
    xs, ys, X, Y, R, TH, h = build_grid(N_GRID, R_OUT)
    mask = annulus_mask(R, R_IN, R_OUT)
    H_base, idx = assemble_operator(R, TH, mask, h, epsB=0.0, epsE=0.0)
    tA = time.time()

    # FAST diagonal of exp(-τH) via Chebyshev + block probes
    Jdiag_base = hutchinson_diag_exp_block(H_base, TAU_FILTER, probes=PROBES, rng=rng)
    J_theta = sample_ring_on_midradius(Jdiag_base, idx, X, Y, R, TH, mask, N_THETA)
    # Sanitize
    if not np.isfinite(J_theta).all():
        finite = np.isfinite(J_theta)
        repl = np.mean(J_theta[finite]) if finite.any() else 1.0
        J_theta[~finite] = repl
    J_mean = float(np.mean(J_theta))
    tB = time.time()

    # Running λ(E) calibration
    E_base = energy_scale(GAMMA_BASE, B_BASE_T)
    E_star = E_STAR_FRAC * E_base
    lam_max = (LN_S_REQUIRED / max(J_mean,1e-30)) * (1.0 + (E_star/E_base)**P_RUN)
    def lam_of_E(E): return lambda_sigmoid(E, lam_max, E_star, P_RUN)

    # ECFM dynamic correction
    delta_lnS = ecfm_ring_delta_lnS(
        Ntheta=N_THETA, steps=ECFM_T, D=ECFM_D, alpha=ECFM_ALPHA,
        sink_pos=ECFM_SINK_POS, sink_wfrac=ECFM_SINK_WIDTH_FRAC,
        Gamma=ECFM_GAMMA, dt=ECFM_DT, init_amp=ECFM_INIT_S, rng=rng
    )
    tC = time.time()

    # Baseline ring profile (save for precession harness)
    scen_base_profile = lam_of_E(E_base) * J_theta + delta_lnS
    np.save(os.path.join(OUTDIR,"S_ring.npy"), scen_base_profile)

    # Scenario helper (means only; no arrays in JSON)
    def scenario(E, label):
        lamE = lam_of_E(E)
        lnS_op  = lamE * J_theta
        lnS_tot = lnS_op + delta_lnS
        return dict(
            label=label,
            E=E,
            lamE=float(lamE),
            J_mean=J_mean,
            lnS_op_mean=float(np.mean(lnS_op)),
            lnS_dyn_mean=float(np.mean(delta_lnS)),
            lnS_tot_mean=float(np.mean(lnS_tot)),
            lnS_required=LN_S_REQUIRED,
            gap=float(LN_S_REQUIRED - float(np.mean(lnS_tot)))
        )

    scen_base = scenario(E_base, "baseline (full beam)")
    scen_half = scenario(0.5*E_base, "half-beam (0.5·E_base)")
    scen_off  = scenario(0.0, "no-beam (E=0)")

    # Console summary
    print("\n=== FAST Substrate Operator + ECFM Collapse ===")
    print(f"Generated: {now}")
    print(f"Grid N={N_GRID}, annulus Rin={R_IN:.3f}, Rout={R_OUT:.3f}, ring Nθ={N_THETA}")
    print(f"Heat-kernel τ={TAU_FILTER:.3f}, probes={PROBES}, Cheb degree={CHEB_K}")
    print(f"Timing: assemble {tA-t0:.2f}s | heat-diag {tB-tA:.2f}s | ECFM {tC-tB:.2f}s | total {time.time()-t0:.2f}s")
    print(f"Calibration: J_mean={J_mean:.6e}  → lam_max={lam_max:.6e}")
    for s in (scen_base, scen_half, scen_off):
        print(f"  - {s['label']}")
        print(f"      λ(E)={s['lamE']:.6e}  lnS_op={s['lnS_op_mean']:.6e}  lnS_dyn={s['lnS_dyn_mean']:.6e}")
        print(f"      lnS_tot={s['lnS_tot_mean']:.6e}  required={s['lnS_required']:.6e}  gap={s['gap']:.6e}")

    # MD + JSON
    md = []
    md.append("# FAST Substrate Operator + ECFM Collapse — Summary")
    md.append(f"_Generated: {now}_\n")
    md.append(f"- Grid: N={N_GRID}, annulus Rin={R_IN:.3f}, Rout={R_OUT:.3f}, ring Nθ={N_THETA}")
    md.append(f"- Heat-kernel: τ={TAU_FILTER:.3f}, probes={PROBES}, Chebyshev degree={CHEB_K}")
    md.append(f"- Timing: assemble {tA-t0:.2f}s | heat-diag {tB-tA:.2f}s | ECFM {tC-tB:.2f}s | total {time.time()-t0:.2f}s")
    md.append(f"- Calibration: J_mean={J_mean:.6e} → lam_max={lam_max:.6e}\n")
    md.append("| Scenario | λ(E) | ⟨lnS⟩_op | ⟨lnS⟩_dyn | ⟨lnS⟩_tot | ⟨lnS⟩_req | gap |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in (scen_base, scen_half, scen_off):
        md.append(f"| {s['label']} | {s['lamE']:.6e} | {s['lnS_op_mean']:.6e} | {s['lnS_dyn_mean']:.6e} "
                  f"| {s['lnS_tot_mean']:.6e} | {s['lnS_required']:.6e} | {s['gap']:.6e} |")
    os.makedirs(OUTDIR, exist_ok=True)
    open(os.path.join(OUTDIR,"substrate_op_collapse_summary.md"),"w").write("\n".join(md)+"\n")

    blob = dict(
        generated=now,
        geometry=dict(N=N_GRID, Rin=R_IN, Rout=R_OUT, Ntheta=N_THETA),
        filter=dict(tau=TAU_FILTER, probes=PROBES, cheb_deg=CHEB_K),
        timing=dict(assemble=round(tA-t0,2), heatdiag=round(tB-tA,2),
                    ecfm=round(tC-tB,2), total=round(time.time()-t0,2)),
        running=dict(model="sigmoid", p=P_RUN,
                     E_base=energy_scale(GAMMA_BASE,B_BASE_T),
                     E_star_frac=E_STAR_FRAC, lam_max=lam_max),
        J_mean=J_mean,
        scenarios=dict(baseline=scen_base, half=scen_half, no_beam=scen_off),
        files=dict(S_ring="outputs/S_ring.npy",
                   md="outputs/substrate_op_collapse_summary.md")
    )
    open(os.path.join(OUTDIR,"substrate_op_collapse_summary.json"),"w").write(json.dumps(blob, indent=2))
    print("\nWrote: outputs/substrate_op_collapse_summary.md, outputs/substrate_op_collapse_summary.json, outputs/S_ring.npy\n")

if __name__ == "__main__":
    main()