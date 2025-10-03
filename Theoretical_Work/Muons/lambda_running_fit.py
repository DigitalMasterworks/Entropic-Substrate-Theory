#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lambda_running_fit.py — Fit the running law λ(E) from intermediate-E measurements (no args).

Input (optional): ./intermediate_points.csv with two columns:
    E, lnS_meas
If not present, a synthetic demo dataset is generated around the baseline.

Model (sigmoid):
    λ(E) = λ_max / (1 + (E_star/E)^p),
with λ_max fixed by the baseline constraint:
    lnS_required = λ(E_base) * <J>   (we use <J>=1 by normalization)
→  λ_max = lnS_required * (1 + (E_star/E_base)^p).

We fit (p, E_star) by least squares on lnS(E) = λ(E).
Outputs: prints summary + writes outputs/lambda_running_fit.json
"""

import os, json, numpy as np

# ----- constants / baseline calibration -----
LN_S_REQUIRED = 2.144e-6
GAMMA_BASE = 29.3
B_BASE_T   = 1.45
E_BASE     = (GAMMA_BASE**2) * (B_BASE_T**2)   # ~ 1804.975225
OUTDIR     = "outputs"

# ----- model -----
def lambda_sigmoid(E, lam_max, E_star, p):
    E = np.maximum(E, 1e-30)
    return lam_max / (1.0 + (E_star / E)**p)

def lnS_model(E, E_star, p):
    # λ_max fixed by baseline (J_mean=1)
    lam_max = LN_S_REQUIRED * (1.0 + (E_star / E_BASE)**p)
    return lambda_sigmoid(E, lam_max, E_star, p)

# ----- loss -----
def rss(params, E, y):
    E_star, p = params
    if E_star <= 0 or p <= 0:
        return 1e99
    yhat = lnS_model(E, E_star, p)
    return float(np.sum((yhat - y)**2))

# ----- simple grid + local polish (no SciPy dependency) -----
def fit_running(E, y):
    # coarse grid
    E_star_grid = np.geomspace(0.1*E_BASE, 2.0*E_BASE, 60)
    p_grid      = np.linspace(0.5, 4.0, 36)
    best = (1e99, None)
    for Es in E_star_grid:
        for p in p_grid:
            r = rss((Es,p), E, y)
            if r < best[0]:
                best = (r, (Es,p))
    # one-step polish around best
    Es0, p0 = best[1]
    Es_cand = Es0 * np.array([0.8, 0.9, 1.0, 1.1, 1.25])
    p_cand  = p0  + np.array([-0.3, -0.15, 0.0, 0.15, 0.3])
    for Es in Es_cand:
        for p in p_cand:
            r = rss((Es,p), E, y)
            if r < best[0]:
                best = (r, (Es,p))
    Es, p = best[1]
    lam_max = LN_S_REQUIRED * (1.0 + (Es / E_BASE)**p)
    return dict(E_star=Es, p=p, lam_max=lam_max, rss=best[0])

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    path = "intermediate_points.csv"
    if os.path.exists(path):
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        if data.ndim == 1: data = data[None,:]
        E  = data[:,0].astype(float)
        y  = data[:,1].astype(float)
        src = "file"
    else:
        # synthetic demo around baseline
        rng = np.random.default_rng(7)
        E  = np.array([0.25, 0.5, 0.75, 1.0])*E_BASE
        true_Estar, true_p = 0.5*E_BASE, 2.0
        lam_max = LN_S_REQUIRED * (1.0 + (true_Estar / E_BASE)**true_p)
        y_clean = lambda_sigmoid(E, lam_max, true_Estar, true_p)
        y = y_clean * (1.0 + 0.03*rng.standard_normal(E.size))
        src = "synthetic"
    fit = fit_running(E, y)

    # predictions
    E_grid = np.linspace(0.0, 1.2*E_BASE, 64)
    y_fit  = lnS_model(E_grid, fit["E_star"], fit["p"])
    # print
    print("\n=== λ(E) running fit (sigmoid) ===")
    print(f"source: {src}, baseline E_base={E_BASE:.3f}, lnS_req={LN_S_REQUIRED:.6e}")
    print(f"fit: E_star={fit['E_star']:.3f}, p={fit['p']:.3f}, lam_max={fit['lam_max']:.6e}, rss={fit['rss']:.3e}")
    print("\nE\tlnS_meas\tlnS_fit")
    for Ei, yi in zip(E, y):
        yhat = lnS_model(np.array([Ei]), fit["E_star"], fit["p"])[0]
        print(f"{Ei:.3f}\t{yi:.6e}\t{yhat:.6e}")

    # write json
    blob = dict(
        baseline=dict(E_base=E_BASE, lnS_required=LN_S_REQUIRED),
        fit=fit,
        data_source=src,
        data=dict(E=E.tolist(), lnS_meas=y.tolist()),
        grid_preview=dict(E=E_grid.tolist(), lnS_fit=y_fit.tolist())
    )
    open(os.path.join(OUTDIR, "lambda_running_fit.json"), "w").write(json.dumps(blob, indent=2))
    print("\nwrote: outputs/lambda_running_fit.json\n")

if __name__ == "__main__":
    main()