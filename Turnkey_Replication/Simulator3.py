import numpy as np
import math
import sys
from datetime import datetime
import json
import matplotlib.pyplot as plt # Kept for potential figure saving, though none is currently implemented

# Global dictionary to hold all tuned parameters
TUNED_PARAMS = {}

# --- Logging Configuration ---
def setup_logging(filename):
    original_stdout = sys.stdout
    log_file = open(filename, 'w')
    sys.stdout = log_file
    print(f"--- Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Script: {filename.replace('.log', '.py')}")
    return original_stdout, log_file

def restore_logging(original_stdout, log_file):
    sys.stdout = original_stdout
    log_file.close()

# ----------------------------------------
# Parameter Loading (Tuned constants must be saved by Autotuner.py)
# ----------------------------------------

def load_tuned_parameters(filename="tuned_params.json"):
    """Loads tuned parameters from a JSON file saved by Autotuner.py."""
    global TUNED_PARAMS
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            TUNED_PARAMS = data
            print(f"--- Parameters loaded from {filename} ---")
            # Set fixed constants explicitly here, as they were in the hardcoded block
            TUNED_PARAMS['G_MATTER'] = 1.0
            TUNED_PARAMS['VEL_CAP'] = 1.0
            TUNED_PARAMS['ANISO_CAP'] = 1.0
            TUNED_PARAMS['ANISO_BINS'] = 24
            print(f"Loaded {len(TUNED_PARAMS)} parameters/constants.")
    except FileNotFoundError:
        print(f"ERROR: Could not find '{filename}'. Ensure Autotuner.py has run and saved its output.")
        # Fallback to hardcoded values to allow testing if file is missing (for development)
        print("Using fallback values for development/testing.")
        TUNED_PARAMS.update({
            "HALO_S0": 0.8968170670639414, "HALO_CORE_R": 8.327195999840875,
            "HALO_INNER": 12.791490262237398, "HALO_OUTER": 66.59679707295717,
            "HALO_S_HALO": 0.9956807477353172, "HALO_S_CORE": 0.22309312365146727,
            "G_LENS": 6.6367187566410255, "G_MATTER": 1.0, "VEL_CAP": 1.0,
            "VOID_S_BG": 0.75, "VOID_S_VOID": 0.985, "VOID_R": 44.57953001819358,
            "VOID_WALL_SIG": 4.245098536820368, "VOID_STEPS": 1065,
            "VOID_DRAG": 0.02291120719726379, "VOID_LAUNCH_B": 0.5369270679387592,
            "ANISO_HILL_DX": 131.25979787923387, "ANISO_STEPS": 402,
            "ANISO_G": 1.0292364736750452, "ANISO_CAP": 1.0, "ANISO_RMIN": 35.0617337025549,
            "ANISO_BINS": 24
        })

# ----------------------------------------
# Common helpers (unchanged)
# ----------------------------------------

def clip_idx(ix, nx):
    return int(np.clip(ix, 1, nx - 2))

def grad_centered(F, ix, iy):
    gx = (F[ix + 1, iy] - F[ix - 1, iy]) * 0.5
    gy = (F[ix, iy + 1] - F[ix, iy - 1]) * 0.5
    return gx, gy

def fit_A_over_b(abs_b, abs_alpha):
    x = 1.0 / np.asarray(abs_b, dtype=float)
    y = np.asarray(abs_alpha, dtype=float)
    num = np.sum(x * y)
    den = np.sum(x * x) + 1e-15
    return num / den

# ----------------------------------------
# 1) Halo test
# ----------------------------------------

def run_halo_block():
    print("\n--- Running Halo Rotation Curve and Lensing Test ---")
    GRID = 200
    cx, cy = GRID // 2, GRID // 2

    # Access parameters from the global dictionary
    S0 = TUNED_PARAMS['HALO_S0']
    core_r = TUNED_PARAMS['HALO_CORE_R']
    r_in = TUNED_PARAMS['HALO_INNER']
    r_out = TUNED_PARAMS['HALO_OUTER']
    S_halo = TUNED_PARAMS['HALO_S_HALO']
    S_core = TUNED_PARAMS['HALO_S_CORE']
    G_MATTER = TUNED_PARAMS['G_MATTER']
    VEL_CAP = TUNED_PARAMS['VEL_CAP']
    G_LENS = TUNED_PARAMS['G_LENS']

    # Build S field using tuned core/halo annulus and levels
    S = np.ones((GRID, GRID), dtype=float) * S0
    for x in range(GRID):
        for y in range(GRID):
            r = math.hypot(x - cx, y - cy)
            if r <= core_r:
                S[x, y] = S_core
            elif r_in < r <= r_out:
                frac = (r - r_in) / max(1e-9, (r_out - r_in))
                S[x, y] = S0 + (S_halo - S0) * (math.cos(frac * math.pi) ** 2)
    S = np.clip(S, 0.0, 0.99)
    C = 1.0 - S

    # Matter tracers (feels G_MATTER)
    radii = np.linspace(5, 90, 20)
    orbiters = []
    for r in radii:
        pos = np.array([cx + r, cy], dtype=float)
        vel = np.array([0.0, 0.30], dtype=float)
        orbiters.append({"pos": pos, "vel": vel})

    for _ in range(500):
        for o in orbiters:
            ix = clip_idx(int(round(o["pos"][0])), GRID)
            iy = clip_idx(int(round(o["pos"][1])), GRID)
            gx, gy = grad_centered(C, ix, iy)
            o["vel"][0] += G_MATTER * gx
            o["vel"][1] += G_MATTER * gy
            sp = float(np.linalg.norm(o["vel"]))
            if sp > VEL_CAP:
                o["vel"] *= VEL_CAP / sp
            o["pos"] += o["vel"]

    velocities = np.array([float(np.linalg.norm(o["vel"])) for o in orbiters])
    dv = np.abs(np.diff(velocities))
    thr = 0.05 * float(np.max(velocities))
    flat_idx = None
    for i in range(len(dv)):
        if np.all(dv[i:] < thr):
            flat_idx = i
            break
    r_flat = float(radii[flat_idx]) if flat_idx is not None else float(radii[-1])
    v_flat = float(np.mean(velocities[flat_idx:])) if flat_idx is not None else float(np.mean(velocities[-3:]))
    M_dyn = (v_flat ** 2) * r_flat / G_MATTER

    # Photon-like rays for lensing (feels G_LENS)
    impact = np.linspace(-40, 40, 11)
    rays = []
    for b in impact:
        rays.append({"pos": np.array([0.0, cy + b], dtype=float),
                     "vel": np.array([1.0, 0.0], dtype=float)})

    for _ in range(300):
        for RY in rays:
            ix = clip_idx(int(round(RY["pos"][0])), GRID)
            iy = clip_idx(int(round(RY["pos"][1])), GRID)
            gx, gy = grad_centered(C, ix, iy)
            RY["vel"][0] += G_LENS * gx
            RY["vel"][1] += G_LENS * gy
            sp = float(np.linalg.norm(RY["vel"]))
            if sp > VEL_CAP:
                RY["vel"] *= VEL_CAP / sp
            RY["pos"] += RY["vel"]

    # Deflection angle
    bends = []
    for RY in rays:
        dy = RY["pos"][1] - (cy)
        dx = RY["pos"][0] - 0.0
        bends.append(math.atan2(dy, dx))

    bends = np.array(bends, dtype=float)
    abs_b = np.abs(impact)
    abs_a = np.abs(bends)
    mask = abs_b > 5.0
    A_fit = fit_A_over_b(abs_b[mask], abs_a[mask])
    M_lens = A_fit / 4.0
    ratio = M_lens / M_dyn if M_dyn != 0.0 else float("nan")

    # Log metrics
    metrics = {
        "r_flat": r_flat, "v_flat": v_flat, "M_dyn": M_dyn,
        "A_fit": A_fit, "M_lens": M_lens, "ratio": ratio
    }
    print(f"[halo] r_flat={r_flat:.3f} v_flat={v_flat:.6g} M_dyn={M_dyn:.6g} A_fit(rad)={A_fit:.6g} M_lens={M_lens:.6g} ratio={ratio:.6g}")

    # Return metrics and necessary field data for final save
    field_data = {"C_field": C, "S_field": S, "GRID": GRID, "cy": cy, "speed_cap": VEL_CAP, "G": G_LENS}
    return metrics, field_data

# ----------------------------------------
# 2) High-S void
# ----------------------------------------

def run_void_block():
    print("\n--- Running Void Repulsion Test ---")
    N = 240
    cx, cy = N // 2, N // 2

    # Access parameters
    S_BG = TUNED_PARAMS['VOID_S_BG']
    S_VOID = TUNED_PARAMS['VOID_S_VOID']
    R = TUNED_PARAMS['VOID_R']
    WALL_SIG = TUNED_PARAMS['VOID_WALL_SIG']
    STEPS = TUNED_PARAMS['VOID_STEPS']
    DRAG = TUNED_PARAMS['VOID_DRAG']
    LAUNCH_B = TUNED_PARAMS['VOID_LAUNCH_B']
    CAP = TUNED_PARAMS['VEL_CAP'] # Shared constant
    G = 1.0 # Fixed constant

    # Tuned S field for void + soft wall
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    rr = np.hypot(xx - cx, yy - cy)

    wall = np.exp(-((rr - R) / max(1e-9, WALL_SIG)) ** 2)
    S = S_BG + (S_VOID - S_BG) * (rr <= R) + 0.18 * wall * (rr > R)
    S = np.clip(S, 0.0, 0.99)
    C = 1.0 - S

    rng = np.random.default_rng(1)
    m = 800
    theta = 2.0 * np.pi * rng.random(m)
    r_start = rng.uniform(70.0, 100.0, size=m)
    px = cx + r_start * np.cos(theta)
    py = cy + r_start * np.sin(theta)

    # Tuned inward launch magnitude
    vx = -LAUNCH_B * np.cos(theta)
    vy = -LAUNCH_B * np.sin(theta)

    for _ in range(STEPS):
        ix = np.clip(px.astype(int), 1, N - 2)
        iy = np.clip(py.astype(int), 1, N - 2)
        gx = (C[ix + 1, iy] - C[ix - 1, iy]) * 0.5
        gy = (C[ix, iy + 1] - C[ix, iy - 1]) * 0.5
        vx += G * gx
        vy += G * gy
        sp = np.sqrt(vx * vx + vy * vy) + 1e-15
        scale = np.minimum(1.0, CAP / sp)
        vx = vx * scale * (1.0 - DRAG)
        vy = vy * scale * (1.0 - DRAG)
        px += vx
        py += vy
        px = np.clip(px, 1, N - 2)
        py = np.clip(py, 1, N - 2)

    r_end = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    # Boundary occupancy within ±20% of tuned R
    win = 0.20 * R
    on_wall = np.abs(r_end - R) <= win
    boundary_fraction = float(np.mean(on_wall))

    # Inward bias
    inward_delta = np.maximum(0.0, r_start - r_end)
    inward_bias = float(np.mean(inward_delta) / R)

    # Log metrics
    metrics = {"boundary_fraction": boundary_fraction, "inward_bias": inward_bias}
    print(f"[void] boundary_fraction={boundary_fraction:.6g} inward_bias={inward_bias:.6g}")

    return metrics

# ----------------------------------------
# 3) Anisotropy
# ----------------------------------------

def run_aniso_block():
    print("\n--- Running Anisotropy Test ---")
    rng = np.random.default_rng(2)
    N = 600
    cx, cy = N // 2, N // 2
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

    # Access parameters
    HILL_DX = TUNED_PARAMS['ANISO_HILL_DX']
    STEPS = TUNED_PARAMS['ANISO_STEPS']
    G = TUNED_PARAMS['ANISO_G']
    CAP = TUNED_PARAMS['ANISO_CAP'] # Fixed constant
    RMIN = TUNED_PARAMS['ANISO_RMIN']
    NUM_BINS = TUNED_PARAMS['ANISO_BINS'] # Fixed constant

    # Place "hill" center
    hill_cx = cx - HILL_DX
    r = np.sqrt((xx - hill_cx) ** 2 + (yy - cy) ** 2)

    # Gentle gradient in S
    S = 0.45 + 0.55 * (r / np.max(r))
    S = np.clip(S, 0.0, 0.99)
    C = 1.0 - S

    num_p = 2000
    r0 = rng.uniform(RMIN, RMIN + 30.0, size=num_p)
    th0 = rng.uniform(0.0, 2.0 * np.pi, size=num_p)
    px = cx + r0 * np.cos(th0)
    py = cy + r0 * np.sin(th0)
    vx = np.zeros(num_p)
    vy = np.zeros(num_p)

    r_init = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    ang_init = (np.arctan2(py - cy, px - cx)) % (2.0 * np.pi)

    for _ in range(STEPS):
        ix = np.clip(px.astype(int), 1, N - 2)
        iy = np.clip(py.astype(int), 1, N - 2)
        gx = (C[ix + 1, iy] - C[ix - 1, iy]) * 0.5
        gy = (C[ix, iy + 1] - C[ix, iy - 1]) * 0.5
        vx += G * gx
        vy += G * gy
        sp = np.sqrt(vx * vx + vy * vy)
        over = sp > CAP
        vx[over] = vx[over] / sp[over] * CAP
        vy[over] = vy[over] / sp[over] * CAP
        px += vx
        py += vy

    r_fin = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    scale = r_fin / np.maximum(1e-12, r_init)

    # Angular binning and directional mean
    edges = np.linspace(0.0, 2.0 * np.pi, NUM_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.searchsorted(edges, ang_init, side="right") - 1, 0, NUM_BINS - 1)
    mean_dir = np.zeros(NUM_BINS)
    for b in range(NUM_BINS):
        sel = idx == b
        if np.any(sel):
            mean_dir[b] = float(np.mean(scale[sel]))
        else:
            mean_dir[b] = np.nan

    # Remove monopole, dipole, quadrupole via linear least squares
    theta = centers
    M = np.column_stack([
        np.ones_like(theta),
        np.cos(theta), np.sin(theta),
        np.cos(2.0 * theta), np.sin(2.0 * theta),
    ])
    mask = np.isfinite(mean_dir)
    coef, *_ = np.linalg.lstsq(M[mask], mean_dir[mask], rcond=None)
    fit = M @ coef
    resid = mean_dir - fit

    # Residual power spectrum
    y = resid.copy()
    msk = np.isfinite(y)
    if np.any(msk):
        y[~msk] = np.nanmean(y[msk])
    else:
        y[:] = 0.0
    Y = np.fft.rfft(y - np.mean(y))
    P = np.abs(Y) ** 2

    A0, A1c, A1s, A2c, A2s = [float(v) for v in coef]
    P_k3 = float(P[3]) if len(P) > 3 else float("nan")

    # Log metrics
    metrics = {
        "A0": A0, "A1c": A1c, "A1s": A1s, "A2c": A2c, "A2s": A2s, "P_k3": P_k3
    }

    resid_head = [float(v) for v in resid[:6]]
    power_head = [float(v) for v in P[:8]]

    print(f"[aniso] A0={A0:.6g} A1c={A1c:.6g} A1s={A1s:.6g} A2c={A2c:.6g} A2s={A2s:.6g}")
    print(f"[aniso] resid_head {resid_head}")
    print(f"[aniso] power_head {power_head}")
    if len(P) > 3:
        print(f"[aniso] P_resid_k3={P_k3:.6g}")

    return metrics

# ----------------------------------------
# Main Execution
# ----------------------------------------

if __name__ == "__main__":
    load_tuned_parameters()

    log_filename = "simulator3.log"
    original_stdout, log_file = setup_logging(log_filename)

    all_metrics = {}
    field_data = {}

    try:
        # 1. Run Halo block, capturing metrics and field state
        halo_metrics, field_data = run_halo_block()
        all_metrics.update(halo_metrics)

        # 2. Run Void block, capturing metrics
        void_metrics = run_void_block()
        all_metrics.update(void_metrics)

        # 3. Run Anisotropy block, capturing metrics
        aniso_metrics = run_aniso_block()
        all_metrics.update(aniso_metrics)

        print("\n--- Saving Simulation Output ---")

        # Prepare data for NPZ save
        npz_save_data = {}
        for k, v in field_data.items():
            npz_save_data[k] = v

        # Save metrics with a clear prefix
        for k, v in all_metrics.items():
            npz_save_data[f'metric_{k}'] = v

        print(f"Metrics captured: {all_metrics}")

        # Final save of both field state and metrics
        np.savez_compressed('simulator_output.npz', **npz_save_data)
        print("Final field state and metrics saved to simulator_output.npz.")

    except Exception as e:
        print(f"\nFATAL ERROR: The simulation failed with an exception: {e}")
    finally:
        # Restore stdout
        restore_logging(original_stdout, log_file)
        # Print confirmation to the actual terminal
        print(f"Log output saved to {log_filename}")
        print("Simulation data saved to simulator_output.npz")