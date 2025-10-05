# Sweep of halo parameters to map M_lens/M_dyn ratio and other metrics
import numpy as np
# --- FIX 1: Add missing imports for plotting and fitting ---
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
import math
# -----------------------------------------------------------

# --- FIX 2: Define Missing Global Constants ---
GRID_SIZE = 200        # Standard grid size for the simulation
G = 1.0                # Gravitational constant (G_MATTER)
C = 1.0                # Velocity cap (VEL_CAP or speed_cap)
cx, cy = GRID_SIZE // 2, GRID_SIZE // 2 # Center coordinates
# ----------------------------------------------

halo_radii = [30, 50, 70]       # pixels
halo_strengths = [0.2, 0.4, 0.6, 0.8]  # S level for halo relative to background
bg_entropy = 0.9
core_radius = 10

results = []

def simulate_halo(core_radius, halo_radius, halo_strength):
    # Build entropy and collapse fields
    # NOTE: GRID_SIZE, bg_entropy, cx, cy are now defined globally
    S = np.ones((GRID_SIZE, GRID_SIZE)) * bg_entropy
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if r <= core_radius:
                S[x, y] = 0.2
            elif core_radius < r <= halo_radius:
                frac = (r - core_radius) / (halo_radius - core_radius)
                S[x, y] = bg_entropy + (halo_strength - bg_entropy) * np.cos(frac * math.pi) ** 2
    C_field = 1 - np.clip(S, 0, 1)

    # Matter orbits
    radii_arr = np.linspace(5, 90, 20)
    # NOTE: cx, cy, C (velocity cap) are now defined globally
    orbiters = [{"pos": np.array([cx + r, cy]), "vel": np.array([0.0, 0.3]), "trail": [], "vel_mag": []} for r in radii_arr]
    for _ in range(500):
        for o in orbiters:
            x, y = o["pos"]
            ix, iy = int(round(x)), int(round(y))
            if 1 <= ix < GRID_SIZE-1 and 1 <= iy < GRID_SIZE-1:
                dCdx = (C_field[ix+1, iy] - C_field[ix-1, iy]) / 2.0
                dCdy = (C_field[ix, iy+1] - C_field[ix, iy-1]) / 2.0
                # NOTE: G is now defined globally
                force = np.array([dCdx, dCdy]) * G
            else:
                force = np.zeros(2)
            o["vel"] += force
            speed = np.linalg.norm(o["vel"])
            # WARNING: C is used as the collapse field (C_field) AND the speed cap here.
            # Assuming C is intended to be the speed cap (1.0).
            if speed > C:
                o["vel"] = o["vel"] / speed * C
            o["pos"] += o["vel"]
            o["vel_mag"].append(np.linalg.norm(o["vel"]))

    velocities = np.array([np.mean(o["vel_mag"]) for o in orbiters])
    dv = np.abs(np.diff(velocities))
    threshold = 0.05 * np.max(velocities)
    flatten_index = None
    for i in range(len(dv)):
        if np.all(dv[i:] < threshold):
            flatten_index = i
            break

    # Handle case where no flat index is found
    if flatten_index is None or flatten_index >= len(radii_arr):
        flatten_index = -1

    flatten_radius = radii_arr[flatten_index]
    v_flat = np.mean(velocities[flatten_index:]) if flatten_index < len(velocities) else 0.0
    M_dyn = (v_flat**2) * flatten_radius / G

    # Photon lensing
    impact_params = np.linspace(-40, 40, 11)
    # NOTE: C (speed cap) and cy are defined globally
    photons = [{"pos": np.array([0.0, cy + b]), "vel": np.array([1.0, 0.0]) * C, "trail": []} for b in impact_params]
    for _ in range(300):
        for p in photons:
            x, y = p["pos"]
            ix, iy = int(round(x)), int(round(y))
            if 1 <= ix < GRID_SIZE-1 and 1 <= iy < GRID_SIZE-1:
                dCdx = (C_field[ix+1, iy] - C_field[ix-1, iy]) / 2.0
                dCdy = (C_field[ix, iy+1] - C_field[ix, iy-1]) / 2.0
                # NOTE: G is defined globally
                force = np.array([dCdx, dCdy]) * G
            else:
                force = np.zeros(2)
            p["vel"] += force
            speed = np.linalg.norm(p["vel"])
            if speed > C:
                p["vel"] = p["vel"] / speed * C
            p["pos"] += p["vel"]
            p["trail"].append(p["pos"].copy())

    bending_angles = []
    for p in photons:
        trail = np.array(p["trail"])
        dy = trail[-1, 1] - trail[0, 1]
        dx = trail[-1, 0] - trail[0, 0]
        angle = math.atan2(dy, dx)
        bending_angles.append(angle)

    mask_fit = np.abs(impact_params) > 5

    # NOTE: curve_fit is now imported from scipy.optimize
    popt_gr, _ = curve_fit(lambda b, A: A / b, np.abs(impact_params[mask_fit]), np.abs(np.array(bending_angles)[mask_fit]))
    A_fit = popt_gr[0]
    M_lens = A_fit / 4

    return flatten_radius, v_flat, M_dyn, M_lens, M_lens/M_dyn

# Run sweep
for hr in halo_radii:
    for hs in halo_strengths:
        fr, vf, md, ml, ratio = simulate_halo(core_radius, hr, hs)
        results.append({"halo_radius": hr, "halo_strength": hs, "flatten_radius": fr, 
                        "v_flat": vf, "M_dyn": md, "M_lens": ml, "ratio": ratio})

# Heatmap of mass ratio
# NOTE: plt is now imported from matplotlib.pyplot
ratios_matrix = np.zeros((len(halo_radii), len(halo_strengths)))
for i, hr in enumerate(halo_radii):
    for j, hs in enumerate(halo_strengths):
        # Use a dictionary comprehension for cleaner matching
        match = next(r for r in results if r["halo_radius"]==hr and r["halo_strength"]==hs)
        ratios_matrix[i, j] = match["ratio"]

plt.figure(figsize=(8,6))
im = plt.imshow(ratios_matrix, cmap='viridis', origin='lower', aspect='auto')
plt.xticks(range(len(halo_strengths)), [f"{hs:.1f}" for hs in halo_strengths])
plt.yticks(range(len(halo_radii)), [str(hr) for hr in halo_radii])
plt.colorbar(im, label="M_lens / M_dyn")
plt.xlabel("Halo Strength (S level)")
plt.ylabel("Halo Radius (pixels)")
plt.title("Lensing-to-Dynamical Mass Ratio Across Halo Parameters")
# plt.show() # Run this interactively to see the plot
plt.savefig('darkmatterhalosweep_heatmap.png')
plt.close()
results
