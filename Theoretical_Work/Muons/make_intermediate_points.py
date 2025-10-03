#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, numpy as np

OUTDIR = "outputs"
SUMMARY = os.path.join(OUTDIR, "substrate_op_collapse_summary.json")
CSV_OUT = "intermediate_points.csv"

# fallback baseline if summary isn't present
LN_S_REQUIRED = 2.144e-6
GAMMA_BASE = 29.3
B_BASE_T   = 1.45
E_BASE_FALLBACK = (GAMMA_BASE**2) * (B_BASE_T**2)  # ~ 1804.975225

def lambda_sigmoid(E, lam_max, E_star, p):
    E = np.maximum(E, 1e-30)
    return lam_max / (1.0 + (E_star / E)**p)

def main():
    if os.path.exists(SUMMARY):
        data = json.load(open(SUMMARY))
        E_base = data["running"]["E_base"] if "E_base" in data["running"] else E_BASE_FALLBACK
        lam_max = data["running"]["lam_max"]
        p = data["running"]["p"]
        # either E_star_frac is there, or compute from frac if present
        if "E_star_frac" in data["running"]:
            E_star = data["running"]["E_star_frac"] * E_base
        else:
            # fallback: half-beam threshold
            E_star = 0.5 * E_base
    else:
        # total fallback: reconstruct lam_max from baseline condition
        E_base = E_BASE_FALLBACK
        p = 2.0
        E_star = 0.5 * E_base
        lam_max = LN_S_REQUIRED * (1.0 + (E_star/E_base)**p)

    # choose a small sweep of energies (unitless proxy) around baseline
    multipliers = np.array([0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.10])
    Es = multipliers * E_base
    lnS = lambda_sigmoid(Es, lam_max, E_star, p)

    with open(CSV_OUT, "w") as f:
        f.write("E,lnS_meas\n")
        for Ei, yi in zip(Es, lnS):
            f.write(f"{Ei:.6f},{yi:.9e}\n")

    print(f"Wrote {CSV_OUT} with {len(Es)} points using:")
    print(f"  E_base={E_base:.3f}, lam_max={lam_max:.6e}, E_star={E_star:.3f}, p={p:.3f}")

if __name__ == "__main__":
    main()