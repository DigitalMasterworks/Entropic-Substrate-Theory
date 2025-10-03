#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, numpy as np
import matplotlib.pyplot as plt

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

def read_csv_any(path):
    """Read our CSV and auto-detect y column ('counts' or 'counts_ideal' or 2nd col)."""
    with open(path, "r", newline="") as f:
        header = f.readline().strip()
    cols = [c.strip() for c in header.split(",")]
    arr = np.genfromtxt(path, delimiter=",", names=True)
    # pick time + y
    tcol = "t_s" if "t_s" in arr.dtype.names else arr.dtype.names[0]
    if "counts" in arr.dtype.names:
        ycol = "counts"
    elif "counts_ideal" in arr.dtype.names:
        ycol = "counts_ideal"
    else:
        # fallback: take the second column
        ycol = [n for n in arr.dtype.names if n != tcol][0]
    return arr[tcol], arr[ycol], ycol

def plot_csv(path):
    t, y, yname = read_csv_any(path)
    base = os.path.basename(path)
    out = os.path.join(OUT, base.replace(".csv", ".png"))
    plt.figure()
    plt.plot(t, y, linewidth=1)
    plt.xlabel("time (s)")
    plt.ylabel(yname.replace("_", " "))
    plt.title(base)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print("wrote", out)

def main():
    # noisy/measured
    for p in sorted(glob.glob(os.path.join(OUT, "precession_run_*_timeseries.csv"))):
        plot_csv(p)
    # ideal
    for p in sorted(glob.glob(os.path.join(OUT, "precession_run_*_ideal_timeseries.csv"))):
        plot_csv(p)

if __name__ == "__main__":
    main()