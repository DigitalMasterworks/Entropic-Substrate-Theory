#!/usr/bin/env python3
import numpy as np, math, csv
A_SM = 116592033e-11
S0_vals = np.linspace(1.0-1.5e-9, 1.0+1.5e-9, 31)
dk_vals = np.linspace(-1.0, 1.0, 41)   # kappa_diff = κμ−κp
with open("outputs/sensitivity_scan.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["S0","kappa_diff","delta_lnR","delta_a_mu","delta_a_mu_x1e11"])
    for S0 in S0_vals:
        lnS = math.log(S0)
        for dk in dk_vals:
            dlnR = dk * lnS
            da = A_SM * dlnR
            w.writerow([f"{S0:.12f}", f"{dk:.6f}", f"{dlnR:.6e}", f"{da:.6e}", f"{da*1e11:.3f}"])
print("wrote outputs/sensitivity_scan.csv")
