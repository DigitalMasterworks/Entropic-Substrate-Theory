#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot substrate g−2 digest (no args).
- Computes required <ln S> for a target δaμ (default 250 × 1e-11).
- Computes Earth <ln S> at site altitude (default 228 m) and implied δaμ.
- Computes QED vacuum estimate for given ring B, E.
- Scans ./outputs for precession_run_*_summary.json and reports measured ΔlnR, δaμ.
- Writes outputs/everything_summary.(md|json) and prints a tidy console report.
"""

import os, glob, json, math, datetime
import numpy as np

# ------------------------
# Constants / defaults
# ------------------------
OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# Target anomaly (dimensionless). 250 × 1e-11 -> 2.5e-9
DELTA_A_TARGET_X1E11 = 250.0
DELTA_A_TARGET = DELTA_A_TARGET_X1E11 * 1e-11

# Standard-Model a_mu (dimensionless)
A_MU_SM = 1.16592033e-3

# Spin-specific tick by default (kappa_mu - kappa_p)
KAPPA_DIFF = 1.0

# Ring fields (typical E989 values / order of magnitude for E)
B_RING_T = 1.45      # Tesla
E_RING_VPM = 1.0e6   # V/m (representative quadrupole scale)

# Site altitude (m) for Earth potential
ALT_M = 228.0

# Physics constants (SI)
c  = 299_792_458.0
c2 = c**2
alpha = 1/137.035999084
E_cr  = 1.32e18          # V/m (Schwinger critical field)
B_cr  = 4.414e9          # T
G   = 6.67430e-11        # m^3 kg^-1 s^-2
M_E = 5.9722e24          # kg
R_E = 6_371_000.0        # m

# ------------------------
# Helpers
# ------------------------
def required_lnS(delta_a_mu: float, a_mu_sm: float, kappa_diff: float) -> float:
    if kappa_diff == 0:
        raise ValueError("kappa_diff=0: universal tick cancels by construction.")
    return delta_a_mu / (a_mu_sm * kappa_diff)

def earth_lnS(alt_m: float) -> float:
    r = R_E + alt_m
    phi = -G * M_E / r
    return phi / c2

def qed_vacuum_lnS(B_T: float, E_Vpm: float):
    """
    Euler–Heisenberg low-field estimate:
      n - 1 ~ (alpha / 45π)[ (B/B_cr)^2 + 7 (E/E_cr)^2 ],
    and near-unity S: n ≈ 1/S => ln S ≈ -(n - 1).
    Returns (lnS_total, (termB, termE, coeff)) for inspection.
    """
    coeff = alpha / (45.0 * math.pi)
    termB = (B_T / B_cr)**2
    termE = 7.0 * (E_Vpm / E_cr)**2
    n_minus_1 = coeff * (termB + termE)
    lnS = -n_minus_1
    return lnS, (termB, termE, coeff)

def newest_summaries():
    paths = sorted(glob.glob(os.path.join(OUTDIR, "precession_run_*_summary.json")))
    runs = []
    for p in paths:
        try:
            J = json.load(open(p))
            runs.append((p, J))
        except Exception:
            pass
    return runs

def mean_lnS_from_Sfile(path: str):
    if not os.path.exists(path):
        return None
    S = np.load(path)
    if np.ndim(S) == 0:
        return float(math.log(float(S)))
    return float(math.log(np.mean(S)))

# ------------------------
# Calculations
# ------------------------
now = datetime.datetime.utcnow().isoformat() + "Z"

lnS_req = required_lnS(DELTA_A_TARGET, A_MU_SM, KAPPA_DIFF)
phi_req = lnS_req * c2  # J/kg equivalent, for intuition

lnS_earth = earth_lnS(ALT_M)
delta_a_from_earth = A_MU_SM * KAPPA_DIFF * lnS_earth
delta_a_from_earth_x1e11 = delta_a_from_earth * 1e11

lnS_qed, (tB, tE, coeff) = qed_vacuum_lnS(B_RING_T, E_RING_VPM)

# Optional: if outputs/S_real.npy exists, compute lnS directly from it
Sreal_path = os.path.join(OUTDIR, "S_real.npy")
lnS_from_Sreal = mean_lnS_from_Sfile(Sreal_path)

# Gather measured results (if present)
measured = []
for p, J in newest_summaries():
    cfg = J.get("config", {})
    S   = J.get("S_stats", {})
    entry = {
        "file": os.path.basename(p),
        "mode": cfg.get("mode"),
        "kappa_mu": cfg.get("kappa_mu"),
        "kappa_p": cfg.get("kappa_p"),
        "S_mean": S.get("S_mean"),
        "S_min": S.get("S_min"),
        "S_max": S.get("S_max"),
        "Delta_lnR": J.get("delta_ln_ratio"),
        "delta_a_mu": J.get("delta_a_linear"),
        "delta_a_mu_x1e11": J.get("delta_a_linear_x1e11"),
        "within_room": J.get("within_current_room_1sigma"),
        "omega_a_std": J.get("omega_a_std_rad_s"),
        "omega_a_meas": J.get("omega_a_meas_fit_rad_s"),
        "ratio_std": J.get("ratio_std"),
        "ratio_meas": J.get("ratio_meas"),
    }
    measured.append(entry)

# ------------------------
# Pretty print
# ------------------------
def sci(x): 
    try: 
        return f"{x:.3e}"
    except Exception: 
        return str(x)

print("\n=== Substrate <ln S> one-shot ===")
print(f"Generated: {now}")
print("\n-- Targets / Assumptions --")
print(f"  δa_mu target                = {DELTA_A_TARGET_X1E11:.1f} × 1e-11  ({sci(DELTA_A_TARGET)})")
print(f"  a_mu^SM                     = {A_MU_SM:.9e}")
print(f"  kappa_diff (=κμ-κp)         = {KAPPA_DIFF:.1f}")
print(f"  Ring fields (B,T; E,V/m)    = {B_RING_T:.3f}, {E_RING_VPM:.3e}")
print(f"  Site altitude (m)           = {ALT_M:.1f}")

print("\n-- Required to match anomaly --")
print(f"  <ln S>_required             = {sci(lnS_req)}  (dimensionless)")
print(f"  Φ_required (=<ln S> c^2)    = {sci(phi_req)}  J/kg")

print("\n-- Earth contribution --")
print(f"  <ln S>_Earth (from Φ/c^2)   = {sci(lnS_earth)}")
print(f"  implied δa_mu (×1e-11)      = {delta_a_from_earth_x1e11:.3f}")

if lnS_from_Sreal is not None:
    print(f"  <ln S> from outputs/S_real  = {sci(lnS_from_Sreal)}  (should match Earth value above)")

print("\n-- QED vacuum (Euler–Heisenberg, low-field) --")
print(f"  termB=(B/B_cr)^2            = {sci(tB)}")
print(f"  termE=7(E/E_cr)^2           = {sci(tE)}")
print(f"  coeff=α/(45π)               = {sci(coeff)}")
print(f"  => <ln S>_QED ≈ -(n-1)      = {sci(lnS_qed)}")

# Measured section
print("\n-- Measured (from outputs/precession_run_*_summary.json) --")
if not measured:
    print("  (no summaries found)")
else:
    # show the last 6 entries (most recent)
    show = measured[-6:]
    for m in show:
        tag = f"{m['file']} | mode={m['mode']} κμ={m['kappa_mu']} κp={m['kappa_p']}"
        print(f"  {tag}")
        print(f"    S_mean={sci(m['S_mean'])}, ΔlnR={sci(m['Delta_lnR'])}, δaμ(×1e-11)={m['delta_a_mu_x1e11']:.3f}, within_room={m['within_room']}")

# ------------------------
# Write MD + JSON artifacts
# ------------------------
md = []
md.append("# Substrate g−2: One-shot Digest")
md.append(f"_Generated: {now}_\n")
md.append("## Required vs. Realistic Contributions\n")
md.append(f"- **Target** δa_μ: {DELTA_A_TARGET_X1E11:.1f} × 1e-11  ({sci(DELTA_A_TARGET)})")
md.append(f"- **Required** ⟨ln S⟩: {sci(lnS_req)}  (Φ_required = {sci(phi_req)} J/kg)")
md.append(f"- **Earth** ⟨ln S⟩: {sci(lnS_earth)}  → implied δa_μ (×1e-11) = {delta_a_from_earth_x1e11:.3f}")
if lnS_from_Sreal is not None:
    md.append(f"- **From `S_real.npy`** ⟨ln S⟩: {sci(lnS_from_Sreal)} (should match Earth value)")
md.append("\n## QED Vacuum Estimate (Low-field EH)\n")
md.append(f"- (B/B_cr)^2 = {sci(tB)},  7(E/E_cr)^2 = {sci(tE)},  α/(45π) = {sci(coeff)}")
md.append(f"- **⟨ln S⟩_QED ≈ −(n−1)** = {sci(lnS_qed)}")
md.append("\n## Measured (latest runs)\n")
if measured:
    md.append("| file | mode | κμ | κp | S_mean | ΔlnR | δaμ (×1e-11) | within room? |")
    md.append("|---|---|---:|---:|---:|---:|---:|:---:|")
    for m in measured[-6:]:
        md.append(f"| {m['file']} | {m['mode']} | {m['kappa_mu']} | {m['kappa_p']} | "
                  f"{m['S_mean']:.12f} | {sci(m['Delta_lnR'])} | {m['delta_a_mu_x1e11']:.3f} | "
                  f"{'✅' if m['within_room'] else '❌'}\" |")
else:
    md.append("(no summaries found)")

open(os.path.join(OUTDIR, "everything_summary.md"), "w").write("\n".join(md) + "\n")

blob = {
    "generated": now,
    "inputs": {
        "delta_a_target_x1e11": DELTA_A_TARGET_X1E11,
        "a_mu_sm": A_MU_SM,
        "kappa_diff": KAPPA_DIFF,
        "B_T": B_RING_T,
        "E_Vpm": E_RING_VPM,
        "alt_m": ALT_M
    },
    "required": {
        "lnS_required": lnS_req,
        "phi_required_J_per_kg": phi_req
    },
    "earth": {
        "lnS_earth": lnS_earth,
        "delta_a_from_earth": delta_a_from_earth,
        "delta_a_from_earth_x1e11": delta_a_from_earth_x1e11,
        "lnS_from_S_real": lnS_from_Sreal
    },
    "qed": {
        "lnS_qed": lnS_qed,
        "termB": tB,
        "termE": tE,
        "coeff": coeff
    },
    "measured_latest": measured[-6:] if measured else []
}
open(os.path.join(OUTDIR, "everything_summary.json"), "w").write(json.dumps(blob, indent=2))

print("\nWrote:")
print("  outputs/everything_summary.md")
print("  outputs/everything_summary.json")
print("\nDone.")