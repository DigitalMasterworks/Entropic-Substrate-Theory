import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# --- Logging Configuration ---
# Function to redirect stdout to a log file
def setup_logging(filename):
    # Backup original stdout
    original_stdout = sys.stdout
    # Open the log file
    log_file = open(filename, 'w')
    # Set stdout to the log file
    sys.stdout = log_file
    print(f"--- Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Script: {filename.replace('.log', '.py')}")
    print("----------------------------------------------------------------")
    return original_stdout, log_file

# Function to restore original stdout
def restore_logging(original_stdout, log_file):
    sys.stdout = original_stdout
    log_file.close()

# --- Main Simulation Logic (wrapped for logging) ---
def run_simulation():
    # Galaxy + High-S Halo Simulation (faithful to entropy substrate logic)
    GRID_SIZE = 200
    C = 1.0
    G = 1.0

    cx, cy = GRID_SIZE // 2, GRID_SIZE // 2

    # --- Create entropy field with smooth halo ---
    def make_entropy_field(core_radius=10, halo_radius=50, halo_strength=0.4, bg_entropy=0.9):
        S = np.ones((GRID_SIZE, GRID_SIZE)) * bg_entropy
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if r <= core_radius:
                    S[x, y] = 0.2  # low-S collapse core
                elif core_radius < r <= halo_radius:
                    # Smooth transition to high-S halo
                    frac = (r - core_radius) / (halo_radius - core_radius)
                    S[x, y] = bg_entropy + (halo_strength - bg_entropy) * np.cos(frac * np.pi) ** 2
        return np.clip(S, 0, 1)

    entropy_field = make_entropy_field()
    collapse_field = 1 - entropy_field

    # --- Matter particle rotation test ---
    num_orbiters = 20
    radii = np.linspace(5, 90, num_orbiters)
    orbiters = []
    for r in radii:
        pos = np.array([cx + r, cy])
        vel = np.array([0.0, 0.3])  # initial guess
        orbiters.append({"pos": pos.copy(), "vel": vel.copy(), "trail": [pos.copy()]})

    # --- Light ray lensing test ---
    num_rays = 11
    impact_params = np.linspace(-40, 40, num_rays)
    photons = []
    for b in impact_params:
        pos = np.array([0.0, cy + b])
        vel = np.array([1.0, 0.0]) * C
        photons.append({"pos": pos.copy(), "vel": vel.copy(), "trail": [pos.copy()]})

    def local_C(ix, iy):
        if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
            return collapse_field[ix, iy]
        return 0.0

    def integrate(objects, steps=500, is_light=False):
        for _ in range(steps):
            for p in objects:
                x, y = p["pos"]
                ix, iy = int(round(x)), int(round(y))
                if 1 <= ix < GRID_SIZE - 1 and 1 <= iy < GRID_SIZE - 1:
                    dCdx = (local_C(ix + 1, iy) - local_C(ix - 1, iy)) / 2.0
                    dCdy = (local_C(ix, iy + 1) - local_C(ix, iy - 1)) / 2.0
                    force = np.array([dCdx, dCdy]) * G
                else:
                    force = np.zeros(2)
                p["vel"] += force
                speed = np.linalg.norm(p["vel"])
                if speed > C:
                    p["vel"] = p["vel"] / speed * C
                p["pos"] += p["vel"]
                p["trail"].append(p["pos"].copy())

    # Run integrations
    integrate(orbiters, steps=500, is_light=False)
    integrate(photons, steps=300, is_light=True)

    # --- Compute orbital speeds ---
    orbital_speeds = []
    for o, r in zip(orbiters, radii):
        speeds = [np.linalg.norm(o["vel"]) for _ in o["trail"]]
        mean_speed = np.mean(speeds) if speeds else 0.0
        orbital_speeds.append(mean_speed)

    print(f"[Results] Mean orbital speed at r={radii[-1]:.1f}: {orbital_speeds[-1]:.4f}")

    # --- Plot 1: Field with orbits and photons (autosave + close) ---
    plt.figure(figsize=(10, 10))
    plt.imshow(entropy_field.T, origin='lower', cmap='plasma', alpha=0.5)
    for o in orbiters:
        trail = np.array(o["trail"])
        plt.plot(trail[:, 0], trail[:, 1], color='white', lw=1)
    for p in photons:
        trail = np.array(p["trail"])
        plt.plot(trail[:, 0], trail[:, 1], color='cyan', lw=1)
    plt.scatter([cx], [cy], color='black', s=50, label="Galaxy Core")
    plt.colorbar(label="Entropy S")
    plt.title("Galaxy Core with High-S Halo: Matter Orbits & Photon Paths")
    plt.legend()
    # MODIFICATION: Save figure instead of showing
    plt.savefig('darkmatterhalo_field_paths.png')
    plt.close()
    print("[Figure] Saved darkmatterhalo_field_paths.png")

    # --- Plot 2: Rotation curve (autosave + close) ---
    plt.figure(figsize=(8, 5))
    plt.plot(radii, orbital_speeds, marker='o')
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Mean Orbital Speed")
    plt.title("Rotation Curve with High-S Halo")
    plt.grid(True)
    # MODIFICATION: Save figure instead of showing
    plt.savefig('darkmatterhalo_rotation_curve.png')
    plt.close()
    print("[Figure] Saved darkmatterhalo_rotation_curve.png")

    # --- Plot 3: Photon bending (autosave + close) ---
    bending_angles = []
    for p, b in zip(photons, impact_params):
        trail = np.array(p["trail"])
        dy = trail[-1, 1] - trail[0, 1]
        dx = trail[-1, 0] - trail[0, 0]
        angle = np.degrees(np.arctan2(dy, dx))
        bending_angles.append(angle - 0)  # relative to initial

    print(f"[Results] Max bending angle: {np.max(np.abs(bending_angles)):.4f} deg")

    plt.figure(figsize=(8, 5))
    plt.plot(impact_params, bending_angles, marker='o', color='magenta')
    plt.xlabel("Impact Parameter (pixels)")
    plt.ylabel("Bending Angle (deg)")
    plt.title("Lensing Deflection vs Impact Parameter")
    plt.grid(True)
    # MODIFICATION: Save figure instead of showing
    plt.savefig('darkmatterhalo_lensing_deflection.png')
    plt.close()
    print("[Figure] Saved darkmatterhalo_lensing_deflection.png")

    print("\n--- Simulation finished successfully. ---")


if __name__ == "__main__":
    # The log file name is derived from the script name, as requested.
    log_filename = "darkmatterhalo.log"

    # Set up logging redirection
    original_stdout, log_file = setup_logging(log_filename)

    try:
        run_simulation()
    except Exception as e:
        print(f"\nFATAL ERROR: The simulation failed with an exception: {e}")
    finally:
        # Restore stdout regardless of success or failure
        restore_logging(original_stdout, log_file)
        # Print confirmation to the actual terminal
        print(f"Log output saved to {log_filename}")
        print("Figures saved to the current directory.")