#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processes existing outputs/ from main.py:
- reads all precession_run_*_summary.json
- reads matching *_timeseries.csv and *_ideal_timeseries.csv (if present)
- makes clean, labeled PNG plots
- writes a consolidated metrics CSV and a human-readable markdown summary
- also picks up hvp_result_*.json if present

No LaTeX. No extra deps beyond numpy/matplotlib.
"""

import os, glob, json, datetime
import numpy as np
import matplotlib.pyplot as plt

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

def read_csv_auto(path):
    """Read CSV and auto-pick time + y column ('counts' or 'counts_ideal' or 2nd col)."""
    arr = np.genfromtxt(path, delimiter=",", names=True)
    names = arr.dtype.names
    if names is None or len(names) < 2:
        raise ValueError(f"Unexpected CSV format: {path}")
    tcol = "t_s" if "t_s" in names else names[0]
    if "counts" in names:
        ycol = "counts"
    elif "counts_ideal" in names:
        ycol = "counts_ideal"
    else:
        # fallback: second column
        ycol = names[1]
    return arr[tcol], arr[ycol], ycol

def plot_timeseries(prefix, title, csv_meas, csv_ideal, omega_a_meas, delta_a_x1e11, within_room):
    """Make a clear, labeled PNG plot for the run."""
    plt.figure(figsize=(9, 4.8))
    legend = []
    if csv_meas and os.path.exists(csv_meas):
        t, y, yn = read_csv_auto(csv_meas)
        plt.plot(t, y, linewidth=1.0)
        legend.append("measured counts")
    if csv_ideal and os.path.exists(csv_ideal):
        t0, y0, yn0 = read_csv_auto(csv_ideal)
        plt.plot(t0, y0, linewidth=1.0, linestyle="--")
        legend.append("ideal counts")

    plt.xlabel("time (s)")
    plt.ylabel("counts")
    room_label = "within room" if within_room else "exceeds room"
    sub = f"ω_a (fit) = {omega_a_meas:.6e} rad/s   |   δa_μ (×1e11) = {delta_a_x1e11:.3f}   |   {room_label}"
    plt.title(f"{title}\n{sub}")
    if legend:
        plt.legend(legend, loc="best")
    plt.tight_layout()
    out_png = os.path.join(OUT, f"{prefix}_plot.png")
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png

def main():
    now = datetime.datetime.utcnow().isoformat() + "Z"

    # collect precession summaries
    summ_paths = sorted(glob.glob(os.path.join(OUT, "precession_run_*_summary.json")))
    if not summ_paths:
        print("No precession_run_*_summary.json found in outputs/. Nothing to do.")
        return

    # prepare metrics sinks
    metrics_csv = os.path.join(OUT, "metrics_summary.csv")
    metrics_md  = os.path.join(OUT, "metrics_summary.md")

    csv_rows = []
    md_lines = [
        "# Substrate g−2: Processed Metrics",
        f"_Generated: {now}_",
        "",
        "| Run | Mode | κμ | κp | S_mean | S_min | S_max | ΔlnR | δaμ (×1e11) | within room? | ω_a std | ω_a meas | R_std | R_meas | Plot |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---|",
    ]

    for sp in summ_paths:
        J = json.load(open(sp))
        cfg = J["config"]
        S  = J["S_stats"]
        prefix = os.path.basename(sp).replace("_summary.json","")

        # locate timeseries files
        csv_meas  = os.path.join(OUT, f"{prefix}_timeseries.csv")
        csv_ideal = os.path.join(OUT, f"{prefix}_ideal_timeseries.csv")

        mode = cfg["mode"]
        kmu  = cfg["kappa_mu"]
        kp   = cfg["kappa_p"]
        S_mean = S["S_mean"]; S_min = S["S_min"]; S_max = S["S_max"]

        dlnR  = J["delta_ln_ratio"]
        da    = J["delta_a_linear"]
        da11  = J["delta_a_linear_x1e11"]
        within = J["within_current_room_1sigma"]

        wstd  = J["omega_a_std_rad_s"]
        wfit  = J["omega_a_meas_fit_rad_s"]
        Rstd  = J["ratio_std"]
        Rmeas = J["ratio_meas"]

        # plot
        title = f"{prefix}   |   mode={mode}   κμ={kmu}   κp={kp}"
        plot_path = plot_timeseries(prefix, title, csv_meas, csv_ideal, wfit, da11, within)

        # write CSV row (human readable order)
        csv_rows.append([
            prefix, mode, kmu, kp,
            f"{S_mean:.12f}", f"{S_min:.12f}", f"{S_max:.12f}",
            f"{dlnR:.6e}", f"{da11:.3f}", str(within),
            f"{wstd:.6e}", f"{wfit:.6e}",
            f"{Rstd:.12e}", f"{Rmeas:.12e}",
            os.path.basename(plot_path)
        ])

        # md line
        md_lines.append(
            f"| `{prefix}` | {mode} | {kmu:.3f} | {kp:.3f} | "
            f"{S_mean:.12f} | {S_min:.12f} | {S_max:.12f} | "
            f"{dlnR:.3e} | {da11:.3f} | {'✅' if within else '❌'} | "
            f"{wstd:.6e} | {wfit:.6e} | {Rstd:.3e} | {Rmeas:.3e} | {os.path.basename(plot_path)} |"
        )

    # also pick up HVP results if present
    hvp_paths = sorted(glob.glob(os.path.join(OUT, "hvp_result_*.json")))
    if hvp_paths:
        md_lines += ["", "## HVP Integrals (toy)", ""]
        md_lines += ["| File | a_mu_HVP_like |", "|---|---:|"]
        for hp in hvp_paths:
            H = json.load(open(hp))
            md_lines.append(f"| {os.path.basename(hp)} | {H.get('a_mu_HVP_like','n/a'):.6e} |")

    # write summary CSV
    with open(metrics_csv, "w") as f:
        f.write(",".join([
            "run_id","mode","kappa_mu","kappa_p",
            "S_mean","S_min","S_max",
            "delta_lnR","delta_a_mu_x1e11","within_room",
            "omega_a_std","omega_a_meas",
            "ratio_std","ratio_meas","plot_png"
        ]) + "\n")
        for row in csv_rows:
            f.write(",".join(map(str,row)) + "\n")

    # write MD
    with open(metrics_md,"w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("Wrote:", metrics_csv)
    print("Wrote:", metrics_md)
    print("Done. PNGs and summaries are in ./outputs/")

if __name__ == "__main__":
    main()