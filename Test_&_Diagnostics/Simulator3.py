# Simulator.py
# Single file. No external data. Numpy and matplotlib only.
# Locks in three diagnostics with the requested fixes and autotuned constants:
# 1) Halo: uses radians for lensing fit and reports M_lens/M_dyn correctly, with separate G_matter/G_lens
# 2) Void: longer integration, tuned wall sigma, true r_start based inward bias using tuned launch_bias
# 3) Anisotropy: tuned hill offset, steps, G, cap; dipole & quadrupole removal + residual power scalar

import numpy as np
import math

# ----------------------------------------
# Tuned constants (baked in from Autotuner.py)
# ----------------------------------------

# --- halo (structure & physics) ---
HALO_S0        = 0.8968170670639414
HALO_CORE_R    = 8.327195999840875
HALO_INNER     = 12.791490262237398
HALO_OUTER     = 66.59679707295717
HALO_S_HALO    = 0.9956807477353172
HALO_S_CORE    = 0.22309312365146727
G_MATTER       = 1.0
G_LENS         = 7.45976835333536  # separate scaling for lightlike rays
VEL_CAP        = 1.0

# (reported by autotuner for reference)
HALO_RFLAT_REF = 72.176
HALO_VFLAT_REF = 0.300
HALO_M_DYN_REF = 6.49588
HALO_AFIT_REF  = 27.1698  # radians * b (dimensionless A in rad)
HALO_MLENS_REF = 6.79245
HALO_RATIO_REF = 1.04565

# --- void (structure & dynamics) ---
VOID_S_BG      = 0.75
VOID_S_VOID    = 0.985
VOID_R         = 44.57953001819358
VOID_WALL_SIG  = 4.245098536820368
VOID_STEPS     = 1065
VOID_DRAG      = 0.02291120719726379
VOID_LAUNCH_B  = 0.5369270679387592   # inward launch speed magnitude
# (reported by autotuner for reference)
VOID_BF_REF    = 0.0101529
VOID_IB_REF    = 0.529019

# --- anisotropy (geometry & dynamics) ---
ANISO_HILL_DX  = 131.25979787923387   # hill center shift (pixels) to the left of cx
ANISO_STEPS    = 402
ANISO_G        = 1.0292364736750452
ANISO_CAP      = 1.0
ANISO_RMIN     = 35.0617337025549
ANISO_SHELLS   = 7
ANISO_BINS     = 24
# (reported by autotuner for reference)
ANISO_A0_REF   = 2.3786
ANISO_A1C_REF  = -0.975047
ANISO_A1S_REF  = 0.0079539
ANISO_A2C_REF  = 0.0173959
ANISO_A2S_REF  = -0.0105654
ANISO_PK3_REF  = 0.0129563
ANISO_TAIL_REF = 0.125905

# ----------------------------------------
# Common helpers
# ----------------------------------------

def clip_idx(ix, nx):
    return int(np.clip(ix, 1, nx - 2))

def grad_centered(F, ix, iy):
    gx = (F[ix + 1, iy] - F[ix - 1, iy]) * 0.5
    gy = (F[ix, iy + 1] - F[ix, iy - 1]) * 0.5
    return gx, gy

def fit_A_over_b(abs_b, abs_alpha):
    # Fit abs_alpha = A / abs_b by linear least squares without scipy
    # Let x = 1/abs_b. Model abs_alpha = A * x
    x = 1.0 / np.asarray(abs_b, dtype=float)
    y = np.asarray(abs_alpha, dtype=float)
    num = np.sum(x * y)
    den = np.sum(x * x) + 1e-15
    return num / den

# ----------------------------------------
# 1) Halo test with correct angle units & tuned field
# ----------------------------------------

def run_halo_block():
    GRID = 200
    cx, cy = GRID // 2, GRID // 2

    # Build S field using tuned core/halo annulus and levels
    S = np.ones((GRID, GRID), dtype=float) * HALO_S0
    for x in range(GRID):
        for y in range(GRID):
            r = math.hypot(x - cx, y - cy)
            if r <= HALO_CORE_R:
                S[x, y] = HALO_S_CORE
            elif HALO_INNER < r <= HALO_OUTER:
                # smooth bump from S0 -> S_halo with cosine^2 easing across [inner, outer]
                frac = (r - HALO_INNER) / max(1e-9, (HALO_OUTER - HALO_INNER))
                S[x, y] = HALO_S0 + (HALO_S_HALO - HALO_S0) * (math.cos(frac * math.pi) ** 2)
            else:
                # leave at baseline S0 outside halo annulus, except inner gap between core and inner edge stays baseline
                pass
    S = np.clip(S, 0.0, 0.99)
    C = 1.0 - S

    # Orbital matter tracers to estimate r_flat and v_flat (matter feels G_MATTER)
    rng = np.random.default_rng(0)
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

    # Photon-like rays for lensing (light feels G_LENS)
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

    # Deflection angle in radians relative to initial x direction
    bends = []
    for RY in rays:
        dy = RY["pos"][1] - (cy)
        dx = RY["pos"][0] - 0.0
        ang = math.atan2(dy, dx)  # radians
        bends.append(ang)

    bends = np.array(bends, dtype=float)
    abs_b = np.abs(impact)
    abs_a = np.abs(bends)

    # Avoid singular fit at small |b|
    mask = abs_b > 5.0
    A_fit = fit_A_over_b(abs_b[mask], abs_a[mask])  # A dimensionless in radians
    M_lens = A_fit / 4.0  # with G=c=1 convention in this toy
    ratio = M_lens / M_dyn if M_dyn != 0.0 else float("nan")

    print(f"[halo] r_flat={r_flat:.3f} v_flat={v_flat:.6g} M_dyn={M_dyn:.6g} A_fit(rad)={A_fit:.6g} M_lens={M_lens:.6g} ratio={ratio:.6g}")

# ----------------------------------------
# 2) High-S void with tuned wall sigma & inward launch bias
# ----------------------------------------

def run_void_block():
    N = 240
    cx, cy = N // 2, N // 2

    # Tuned S field for void + soft wall
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    rr = np.hypot(xx - cx, yy - cy)

    wall = np.exp(-((rr - VOID_R) / max(1e-9, VOID_WALL_SIG)) ** 2)  # tuned sigma
    S = VOID_S_BG + (VOID_S_VOID - VOID_S_BG) * (rr <= VOID_R) + 0.18 * wall * (rr > VOID_R)
    S = np.clip(S, 0.0, 0.99)
    C = 1.0 - S

    rng = np.random.default_rng(1)
    m = 800
    theta = 2.0 * np.pi * rng.random(m)
    r_start = rng.uniform(70.0, 100.0, size=m)
    px = cx + r_start * np.cos(theta)
    py = cy + r_start * np.sin(theta)

    # Tuned inward launch magnitude (bias toward center)
    vx = -VOID_LAUNCH_B * np.cos(theta)
    vy = -VOID_LAUNCH_B * np.sin(theta)

    steps = VOID_STEPS
    drag = VOID_DRAG
    cap = VEL_CAP
    G = 1.0

    for _ in range(steps):
        ix = np.clip(px.astype(int), 1, N - 2)
        iy = np.clip(py.astype(int), 1, N - 2)
        gx = (C[ix + 1, iy] - C[ix - 1, iy]) * 0.5
        gy = (C[ix, iy + 1] - C[ix, iy - 1]) * 0.5
        vx += G * gx
        vy += G * gy
        sp = np.sqrt(vx * vx + vy * vy) + 1e-15
        scale = np.minimum(1.0, cap / sp)
        vx = vx * scale * (1.0 - drag)
        vy = vy * scale * (1.0 - drag)
        px += vx
        py += vy
        px = np.clip(px, 1, N - 2)
        py = np.clip(py, 1, N - 2)

    r_end = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

    # Boundary occupancy within ±20% of tuned R (matches your prior diagnostic window)
    win = 0.20 * VOID_R
    on_wall = np.abs(r_end - VOID_R) <= win
    boundary_fraction = float(np.mean(on_wall))

    # Inward bias measured from true r_start
    inward_delta = np.maximum(0.0, r_start - r_end)
    inward_bias = float(np.mean(inward_delta) / VOID_R)

    print(f"[void] boundary_fraction={boundary_fraction:.6g} inward_bias={inward_bias:.6g}")

# ----------------------------------------
# 3) Anisotropy cleaning and residual power (tuned)
# ----------------------------------------

def run_aniso_block():
    rng = np.random.default_rng(2)
    N = 600
    cx, cy = N // 2, N // 2
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

    # Place "hill" center to the left by tuned dx
    hill_cx = cx - ANISO_HILL_DX
    r = np.sqrt((xx - hill_cx) ** 2 + (yy - cy) ** 2)

    # Gentle gradient in S increasing with distance from hill center
    S = 0.45 + 0.55 * (r / np.max(r))
    S = np.clip(S, 0.0, 0.99)
    C = 1.0 - S

    num_p = 2000
    # Use tuned minimum radius; keep a modest spread outward (30 px) to avoid razor-thin shelling
    r0 = rng.uniform(ANISO_RMIN, ANISO_RMIN + 30.0, size=num_p)
    th0 = rng.uniform(0.0, 2.0 * np.pi, size=num_p)
    px = cx + r0 * np.cos(th0)
    py = cy + r0 * np.sin(th0)
    vx = np.zeros(num_p)
    vy = np.zeros(num_p)

    r_init = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    ang_init = (np.arctan2(py - cy, px - cx)) % (2.0 * np.pi)

    steps = ANISO_STEPS
    G = ANISO_G
    cap = ANISO_CAP

    for _ in range(steps):
        ix = np.clip(px.astype(int), 1, N - 2)
        iy = np.clip(py.astype(int), 1, N - 2)
        gx = (C[ix + 1, iy] - C[ix - 1, iy]) * 0.5
        gy = (C[ix, iy + 1] - C[ix, iy - 1]) * 0.5
        vx += G * gx
        vy += G * gy
        sp = np.sqrt(vx * vx + vy * vy)
        over = sp > cap
        vx[over] = vx[over] / sp[over] * cap
        vy[over] = vy[over] / sp[over] * cap
        px += vx
        py += vy

    r_fin = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    scale = r_fin / np.maximum(1e-12, r_init)

    # Angular binning and directional mean (tuned bins)
    num_bins = ANISO_BINS
    edges = np.linspace(0.0, 2.0 * np.pi, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.searchsorted(edges, ang_init, side="right") - 1, 0, num_bins - 1)
    mean_dir = np.zeros(num_bins)
    for b in range(num_bins):
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
    resid_head = [float(v) for v in resid[:6]]
    power_head = [float(v) for v in P[:8]]

    print(f"[aniso] A0={A0:.6g} A1c={A1c:.6g} A1s={A1s:.6g} A2c={A2c:.6g} A2s={A2s:.6g}")
    print(f"[aniso] resid_head {resid_head}")
    print(f"[aniso] power_head {power_head}")
    if len(P) > 3:
        print(f"[aniso] P_resid_k3={float(P[3]):.6g}")

# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":
    run_halo_block()
    run_void_block()
    run_aniso_block()
