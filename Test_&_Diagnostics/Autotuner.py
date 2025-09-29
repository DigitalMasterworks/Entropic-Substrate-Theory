# AutoTune.py
# Self-contained parameter tuner for halo, void, and anisotropy toy programs
# No external data, no external libs beyond numpy and matplotlib (matplotlib used only if you flip PLOT=True)

import numpy as np
import math
import time

RNG = np.random.default_rng(1234)
PLOT = False  # set True if you want quick plots during tuning

# ------------------------------
# shared helpers
# ------------------------------

def clip01(x): 
    return max(0.0, min(1.0, x))

def grad_centered(F, ix, iy):
    nx, ny = F.shape
    ix = int(np.clip(ix, 1, nx-2))
    iy = int(np.clip(iy, 1, ny-2))
    gx = (F[ix+1, iy] - F[ix-1, iy]) * 0.5
    gy = (F[ix, iy+1] - F[ix, iy-1]) * 0.5
    return gx, gy

def find_flatten_radius(radii, velocity_series, frac=0.05):
    v = np.array(velocity_series, float)
    dv = np.abs(np.diff(v))
    thresh = frac * np.max(v) if np.max(v) > 0 else 0.0
    idx = None
    for i in range(len(dv)):
        if np.all(dv[i:] < thresh):
            idx = i
            break
    if idx is None:
        return radii[-1], float(np.mean(v[-max(1, len(v)//4):]))
    return radii[idx], float(np.mean(v[idx:]))

# ------------------------------
# block A: halo mass ratio tuner
# target: M_lens / M_dyn near 1
# knobs: S0 baseline, halo_inner, halo_outer, G_matter, G_lens
# ------------------------------

def simulate_halo_block(
    GRID=180,
    S0=0.85,
    core_r=10.0,
    halo_inner=15.0,
    halo_outer=60.0,
    S_halo=0.995,
    S_core=0.20,
    G_matter=1.0,
    G_lens=6.0,
    steps_orb=500,
    steps_ray=320,
    C_cap=1.0
):
    cx = cy = GRID // 2
    S = np.ones((GRID, GRID), float) * S0
    for x in range(GRID):
        for y in range(GRID):
            r = math.hypot(x - cx, y - cy)
            if r <= core_r:
                S[x, y] = S_core
            elif halo_inner < r <= halo_outer:
                t = (r - halo_inner) / max(1e-9, (halo_outer - halo_inner))
                # smooth shoulder toward S_halo
                S[x, y] = S0 + (S_halo - S0) * math.cos(math.pi * t) ** 2
    S = np.clip(S, 0.0, 0.999)
    C = 1.0 - S

    # orbiters
    radii = np.linspace(6.0, GRID*0.45, 18)
    orbiters = []
    for r in radii:
        pos = np.array([cx + r, cy], float)
        vel = np.array([0.0, 0.30], float)
        orbiters.append({"pos": pos, "vel": vel, "v_hist": []})

    for _ in range(steps_orb):
        for o in orbiters:
            x, y = o["pos"]
            gx, gy = grad_centered(C, x, y)
            o["vel"][0] += G_matter * gx
            o["vel"][1] += G_matter * gy
            sp = np.linalg.norm(o["vel"])
            if sp > C_cap:
                o["vel"] *= C_cap / sp
            o["pos"] += o["vel"]
            o["v_hist"].append(np.linalg.norm(o["vel"]))

    v_mean = np.array([np.mean(o["v_hist"]) for o in orbiters])
    r_flat, v_flat = find_flatten_radius(radii, v_mean, frac=0.05)
    M_dyn = (v_flat ** 2) * r_flat / max(1e-12, G_matter)

    # photons
    impact = np.linspace(-0.22*GRID, 0.22*GRID, 13)
    rays = []
    for b in impact:
        rays.append({"pos": np.array([0.0, cy + b], float), "vel": np.array([1.0, 0.0], float)})

    for _ in range(steps_ray):
        for RY in rays:
            x, y = RY["pos"]
            gx, gy = grad_centered(C, x, y)
            RY["vel"][0] += G_lens * gx
            RY["vel"][1] += G_lens * gy
            sp = np.linalg.norm(RY["vel"])
            if sp > 1.0:
                RY["vel"] /= sp
            RY["pos"] += RY["vel"]

    # deflection angles in radians relative to x axis
    bends = []
    for RY in rays:
        vx, vy = RY["vel"]
        bends.append(math.atan2(vy, vx))
    bends = np.array(bends, float)

    # fit |alpha| ~ A / |b| avoiding center
    mask = np.abs(impact) > 0.08*GRID
    xfit = np.abs(impact[mask]) + 1e-9
    yfit = np.abs(bends[mask])
    if np.all(yfit == 0):
        A = 0.0
    else:
        # linear in 1/b
        X = np.vstack([1.0 / xfit, np.ones_like(xfit)]).T
        sol, *_ = np.linalg.lstsq(X, yfit, rcond=None)
        A = max(0.0, float(sol[0]))  # radians * pixels

    M_lens = A / 4.0  # with G=c=1 conventions in this toy
    ratio = M_lens / max(1e-12, M_dyn)
    return {
        "r_flat": float(r_flat),
        "v_flat": float(v_flat),
        "M_dyn": float(M_dyn),
        "A_fit_rad": float(A),
        "M_lens": float(M_lens),
        "ratio": float(ratio),
        "params": {
            "S0": S0,
            "core_r": core_r,
            "halo_inner": halo_inner,
            "halo_outer": halo_outer,
            "S_halo": S_halo,
            "S_core": S_core,
            "G_matter": G_matter,
            "G_lens": G_lens
        }
    }

# ------------------------------
# block B: void boundary tuner
# target: boundary_fraction high with inward_bias positive
# knobs: drag, steps, R, wall_sigma, launch_bias
# ------------------------------

def simulate_void_block(
    N=220,
    S_bg=0.75,
    S_void=0.985,
    R=44.0,
    wall_sigma=6.0,
    steps=1200,
    drag=0.04,
    launch_bias=0.40
):
    cx = cy = N // 2
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    rr = np.hypot(xx - cx, yy - cy)

    wall = np.exp(-((rr - R) / max(1e-9, wall_sigma)) ** 2)
    S = S_bg + (S_void - S_bg) * (rr <= R) + 0.18 * wall * (rr > R)
    S = np.clip(S, 0.0, 0.999)
    C = 1.0 - S

    m = 700
    theta = RNG.uniform(0, 2*np.pi, size=m)
    r0 = RNG.uniform(R + 18.0, min(N-20, R + 60.0), size=m)
    px = cx + r0 * np.cos(theta)
    py = cy + r0 * np.sin(theta)
    vx = -launch_bias * np.cos(theta)
    vy = -launch_bias * np.sin(theta)

    boundary_acc = 0.0
    inward_acc = 0.0

    for _ in range(steps):
        ix = np.clip(px.astype(int), 1, N-2)
        iy = np.clip(py.astype(int), 1, N-2)
        gx = (C[ix+1, iy] - C[ix-1, iy]) * 0.5
        gy = (C[ix, iy+1] - C[ix, iy-1]) * 0.5
        vx += gx
        vy += gy
        sp = np.sqrt(vx*vx + vy*vy) + 1e-12
        cap = 1.0
        over = sp > cap
        if np.any(over):
            vx[over] *= cap / sp[over]
            vy[over] *= cap / sp[over]
        # drag
        vx *= (1.0 - drag)
        vy *= (1.0 - drag)
        # move
        px += vx
        py += vy
        px = np.clip(px, 1, N-2)
        py = np.clip(py, 1, N-2)

        r_now = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        on_wall = np.abs(r_now - R) <= (0.20 * R)
        boundary_acc += float(np.mean(on_wall))

        # inward bias measured by mean radial velocity sign
        vr = ((px - cx) * vx + (py - cy) * vy) / (r_now + 1e-9)
        inward_acc += float(np.mean(vr < 0.0))

    boundary_fraction = boundary_acc / steps
    inward_bias = inward_acc / steps
    return {
        "boundary_fraction": float(boundary_fraction),
        "inward_bias": float(inward_bias),
        "params": {
            "S_bg": S_bg,
            "S_void": S_void,
            "R": R,
            "wall_sigma": wall_sigma,
            "steps": steps,
            "drag": drag,
            "launch_bias": launch_bias
        }
    }

# ------------------------------
# block C: anisotropy cleaner tuner
# target: small residual at k=3 and low high-tail mean
# knobs: hill_offset, steps, G, Rmin, shell_count
# ------------------------------

def simulate_aniso_block(
    N=520,
    hill_dx=140.0,
    steps=420,
    G=1.0,
    cap=1.0,
    Rmin=28.0,
    shell_bins=8,
    bins=24
):
    cx = cy = N // 2
    xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt((xx - (cx - hill_dx))**2 + (yy - cy)**2)
    S = 0.45 + 0.55 * (r / max(1.0, r.max()))
    S = np.clip(S, 0.0, 0.999)
    C = 1.0 - S

    m = 1800
    th0 = RNG.uniform(0, 2*np.pi, size=m)
    r0 = RNG.uniform(16.0, 46.0, size=m)
    px = cx + r0 * np.cos(th0)
    py = cy + r0 * np.sin(th0)
    vx = np.zeros(m)
    vy = np.zeros(m)

    r_init = np.sqrt((px - cx)**2 + (py - cy)**2)
    ang_init = (np.arctan2(py - cy, px - cx)) % (2*np.pi)

    for _ in range(steps):
        ix = np.clip(px.astype(int), 1, N-2)
        iy = np.clip(py.astype(int), 1, N-2)
        gx = (C[ix+1, iy] - C[ix-1, iy]) * 0.5
        gy = (C[ix, iy+1] - C[ix, iy-1]) * 0.5
        vx += G * gx
        vy += G * gy
        sp = np.sqrt(vx*vx + vy*vy)
        over = sp > cap
        if np.any(over):
            vx[over] *= cap / sp[over]
            vy[over] *= cap / sp[over]
        px += vx
        py += vy

    r_fin = np.sqrt((px - cx)**2 + (py - cy)**2)
    scale = r_fin / np.maximum(1e-9, r_init)

    # angular bins
    edges = np.linspace(0, 2*np.pi, bins+1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.searchsorted(edges, ang_init, side="right") - 1, 0, bins-1)

    # local cut
    far = r_init >= Rmin

    # shell indices over r_fin
    rf = r_fin[far]
    if rf.size < 10:
        return {"P_k3": float('inf'), "tail_mean": float('inf'), "params": {}}
    se = np.linspace(np.percentile(rf, 5), np.percentile(rf, 95), shell_bins+1)
    sh_idx = np.clip(np.searchsorted(se, r_fin, side="right") - 1, 0, shell_bins-1)

    clean = np.zeros(bins)
    count = np.zeros(bins)
    for s in range(shell_bins):
        in_s = far & (sh_idx == s)
        if not np.any(in_s):
            continue
        bsub = np.clip(np.searchsorted(edges, ang_init[in_s], side="right") - 1, 0, bins-1)
        for b in range(bins):
            ss = bsub == b
            if np.any(ss):
                clean[b] += float(np.mean(scale[in_s][ss]))
                count[b] += 1.0
    clean = np.where(count > 0, clean / np.maximum(1.0, count), np.nan)

    # fit monopole dipole quadrupole
    theta = centers
    M = np.column_stack([
        np.ones_like(theta),
        np.cos(theta), np.sin(theta),
        np.cos(2*theta), np.sin(2*theta)
    ])
    mask = np.isfinite(clean)
    coef, *_ = np.linalg.lstsq(M[mask], clean[mask], rcond=None)
    fit = M @ coef
    resid = clean - fit

    # angular power of residual
    yy = resid.copy()
    mm = np.isfinite(yy)
    yy[~mm] = np.nanmean(yy[mm]) if np.any(mm) else 0.0
    Y = np.fft.rfft(yy - np.mean(yy))
    P = np.abs(Y)**2
    k3 = int(3) if len(P) > 3 else len(P)-1
    tail = np.mean(P[6:]) if len(P) > 7 else np.mean(P[3:])
    return {
        "A0": float(coef[0]),
        "A1c": float(coef[1]),
        "A1s": float(coef[2]),
        "A2c": float(coef[3]),
        "A2s": float(coef[4]),
        "P_k3": float(P[k3]),
        "tail_mean": float(tail),
        "params": {
            "hill_dx": hill_dx,
            "steps": steps,
            "G": G,
            "cap": cap,
            "Rmin": Rmin,
            "shell_bins": shell_bins,
            "bins": bins
        }
    }

# ------------------------------
# random search drivers
# ------------------------------

def tune_halo(budget=120):
    best = None
    t0 = time.time()
    for i in range(budget):
        S0 = RNG.uniform(0.80, 0.92)
        core_r = RNG.uniform(8.0, 14.0)
        halo_in = RNG.uniform(12.0, 20.0)
        halo_out = RNG.uniform(50.0, 80.0)
        S_halo = RNG.uniform(0.985, 0.999)
        S_core = RNG.uniform(0.15, 0.25)
        G_m = 1.0
        G_l = RNG.uniform(2.0, 10.0)
        res = simulate_halo_block(
            S0=S0, core_r=core_r, halo_inner=halo_in, halo_outer=halo_out,
            S_halo=S_halo, S_core=S_core, G_matter=G_m, G_lens=G_l
        )
        score = abs(res["ratio"] - 1.0) + 0.2*abs(res["v_flat"] - 0.30)
        item = (score, res)
        if best is None or score < best[0]:
            best = item
        if (i+1) % 20 == 0:
            print(f"[halo] {i+1}/{budget} current best ratio={best[1]['ratio']:.3f} A_fit(rad)={best[1]['A_fit_rad']:.3g}")
    print(f"[halo] done in {time.time()-t0:.1f}s")
    return best[1]

def tune_void(budget=80):
    best = None
    t0 = time.time()
    for i in range(budget):
        R = RNG.uniform(40.0, 55.0)
        wall_sigma = RNG.uniform(4.0, 9.0)
        steps = int(RNG.integers(900, 1600))
        drag = RNG.uniform(0.02, 0.08)
        launch_bias = RNG.uniform(0.25, 0.55)
        res = simulate_void_block(R=R, wall_sigma=wall_sigma, steps=steps, drag=drag, launch_bias=launch_bias)
        # want boundary_fraction high and inward_bias positive
        # score lowers when boundary_fraction approaches 0.35..0.7 and inward_bias between 0.3..0.8
        bf = res["boundary_fraction"]
        ib = res["inward_bias"]
        loss = (max(0.0, 0.35 - bf) + max(0.0, bf - 0.7)) + (max(0.0, 0.3 - ib) + max(0.0, ib - 0.85))
        item = (loss, res)
        if best is None or loss < best[0]:
            best = item
        if (i+1) % 20 == 0:
            print(f"[void] {i+1}/{budget} best bf={best[1]['boundary_fraction']:.3f} ib={best[1]['inward_bias']:.3f}")
    print(f"[void] done in {time.time()-t0:.1f}s")
    return best[1]

def tune_aniso(budget=100):
    best = None
    t0 = time.time()
    for i in range(budget):
        hill_dx = RNG.uniform(120.0, 180.0)
        steps = int(RNG.integers(360, 520))
        G = RNG.uniform(0.8, 1.2)
        Rmin = RNG.uniform(24.0, 36.0)
        shell_bins = int(RNG.integers(6, 10))
        res = simulate_aniso_block(hill_dx=hill_dx, steps=steps, G=G, Rmin=Rmin, shell_bins=shell_bins)
        # prefer small P_k3 and small tail_mean
        score = 10.0*res["P_k3"] + res["tail_mean"]
        item = (score, res)
        if best is None or score < best[0]:
            best = item
        if (i+1) % 20 == 0:
            print(f"[aniso] {i+1}/{budget} best P_k3={best[1]['P_k3']:.4g} tail={best[1]['tail_mean']:.4g}")
    print(f"[aniso] done in {time.time()-t0:.1f}s")
    return best[1]

# ------------------------------
# main
# ------------------------------

if __name__ == "__main__":
    print("tuning halo...")
    halo_best = tune_halo(budget=120)
    print("tuning void...")
    void_best = tune_void(budget=80)
    print("tuning anisotropy...")
    aniso_best = tune_aniso(budget=100)

    print("\n===== suggested bake in values =====")
    print(f"# halo")
    for k, v in halo_best["params"].items():
        print(f"{k} = {v}")
    print(f"r_flat = {halo_best['r_flat']:.3f}")
    print(f"v_flat = {halo_best['v_flat']:.3f}")
    print(f"M_dyn = {halo_best['M_dyn']:.6g}")
    print(f"A_fit_rad = {halo_best['A_fit_rad']:.6g}")
    print(f"M_lens = {halo_best['M_lens']:.6g}")
    print(f"ratio = {halo_best['ratio']:.6g}")

    print(f"\n# void")
    for k, v in void_best["params"].items():
        print(f"{k} = {v}")
    print(f"boundary_fraction = {void_best['boundary_fraction']:.6g}")
    print(f"inward_bias = {void_best['inward_bias']:.6g}")

    print(f"\n# anisotropy")
    for k, v in aniso_best["params"].items():
        print(f"{k} = {v}")
    print(f"A0 = {aniso_best['A0']:.6g}")
    print(f"A1c = {aniso_best['A1c']:.6g}")
    print(f"A1s = {aniso_best['A1s']:.6g}")
    print(f"A2c = {aniso_best['A2c']:.6g}")
    print(f"A2s = {aniso_best['A2s']:.6g}")
    print(f"P_k3 = {aniso_best['P_k3']:.6g}")
    print(f"tail_mean = {aniso_best['tail_mean']:.6g}")

    print("\ncopy the constants into Simulator.py and lock them.")