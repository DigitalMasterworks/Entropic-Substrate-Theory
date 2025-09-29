import numpy as np
import sys

# --- Parameters ---
N = 50              # Grid size (cube: N x N x N)
timesteps = 500     # Number of steps (shorter for testing)
dt = 0.01           # Base timestep
M = 1.0
M0 = 1.0

# --- Physical constants (SI) ---
G = 6.67430e-11      # m^3 / (kg s^2)
c = 2.99792458e8     # m/s
M_sun = 1.988e30     # kg
planck_length = 1.616e-35  # m
planck_mass = 2.176e-8     # kg
planck_time = 5.391e-44    # s

voxel_size = planck_length

# --- Initialize corridor ---
S = np.ones((N, N, N))
C = np.zeros((N, N, N))

def initialize_corridor(S, C, hall_len=20, hall_axis=0, channel_halfwidth=1):
    N = S.shape[0]
    S[:] = 1.0
    C[:] = 0.0
    cx, cy, cz = N//2, N//2, N//2

    if hall_axis == 0:  # corridor along x-axis
        for dx in range(hall_len):
            x = cx - hall_len//2 + dx
            if 1 <= x < N-1:
                # carve entropy channel
                for dy in range(-channel_halfwidth, channel_halfwidth+1):
                    for dz in range(-channel_halfwidth, channel_halfwidth+1):
                        y = cy + dy
                        z = cz + dz
                        if 0 <= y < N and 0 <= z < N:
                            S[x, y, z] = 1.0
                            C[x, y, z] = 0.0
                # walls
                for dy in range(-(channel_halfwidth+1), channel_halfwidth+2):
                    for dz in range(-(channel_halfwidth+1), channel_halfwidth+2):
                        if abs(dy) > channel_halfwidth or abs(dz) > channel_halfwidth:
                            y = cy + dy
                            z = cz + dz
                            if 0 <= y < N and 0 <= z < N:
                                S[x, y, z] = 0.0
                                C[x, y, z] = 1.0
        # sealed stop
        x_stop = cx + hall_len//2
        if 1 <= x_stop < N-1:
            for dy in range(-channel_halfwidth, channel_halfwidth+1):
                for dz in range(-channel_halfwidth, channel_halfwidth+1):
                    y = cy + dy
                    z = cz + dz
                    if 0 <= y < N and 0 <= z < N:
                        S[x_stop, y, z] = 0.0
                        C[x_stop, y, z] = 1.0
    return S, C

S, C = initialize_corridor(S, C, hall_len=20, channel_halfwidth=1)

def print_corridor(S, C, hall_len=5, t=0):
    cx, cy, cz = N//2, N//2, N//2
    x_start = cx - hall_len//2
    lines = [f"t = {t}", "1 1 1"]
    for dx in range(1, hall_len):
        x = x_start + dx
        val = C[x, cy, cz]
        lines.append(f"1 {val:.2f} 1")
    out = "\n".join(lines)
    sys.stdout.write("\033[H\033[J")  # clear screen
    sys.stdout.write(out + "\n")
    sys.stdout.flush()

def collapse_influx(S, C, hall_len=20, channel_halfwidth=1, thickness=1):
    cx, cy, cz = N//2, N//2, N//2
    x_entrance = cx - hall_len//2
    for dy in range(-channel_halfwidth, channel_halfwidth+1):
        for dz in range(-channel_halfwidth, channel_halfwidth+1):
            y = cy + dy
            z = cz + dz
            if 0 <= y < N and 0 <= z < N:
                S[x_entrance:x_entrance+thickness, y, z] = 0.0
                C[x_entrance:x_entrance+thickness, y, z] = 1.0
    return S, C

def laplacian(S):
    return (
        np.pad(S, 1, mode='constant')[1:-1,1:-1,1:-1] * -26
        + sum(np.roll(np.roll(np.roll(S, dx, 0), dy, 1), dz, 2)
              for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
              if not (dx == dy == dz == 0))
    )

def step(S, C):
    deltaS = laplacian(S)
    dt_eff = dt * M / M0 * S
    S_new = S + dt_eff * deltaS
    S_new = np.clip(S_new, 0, 1)    # freeze naturally at 0
    C_new = 1 - S_new
    return S_new, C_new

def check_breakdown(S, C):
    if np.isnan(S).any() or np.isnan(C).any():
        print("Breakdown detected: NaN values!")
        return True
    if np.max(S) > 1.01 or np.min(S) < -0.01:
        print("Breakdown detected: S out of bounds!")
        return True
    return False

# --- Run Simulation ---
for t in range(timesteps):
    S, C = step(S, C)
    S, C = collapse_influx(S, C, hall_len=20, channel_halfwidth=1)
    if t % 50 == 0:
        print_corridor(S, C, hall_len=20, t=t)
    if check_breakdown(S, C):
        print(f"Surgery required at step {t}")
        break

print("Simulation complete.\n")

# --- Horizon detection ---
cx, cy, cz = N//2, N//2, N//2
x_start = cx - 10
channel_vals = [S[x_start+dx, cy, cz] for dx in range(20)]

horizon_index = None
for i, val in enumerate(channel_vals[1:], start=1):  # skip index 0
    if val < 0.5:   # treat S<0.5 as horizon
        horizon_index = i
        break

if horizon_index is not None:
    R_substrate = horizon_index * voxel_size
    R_s = (2 * G * M_sun) / (c**2)
    print(f"Substrate horizon formed at corridor index {horizon_index}")
    print(f"Substrate horizon scale ~ {R_substrate:.3e} m")
    print(f"GR Schwarzschild radius = {R_s:.3e} m")
    print(f"Ratio (substrate/GR) = {R_substrate/R_s:.3e}")

    # Compute voxel size needed for substrate to match GR
    voxel_size_match = R_s / horizon_index
    print(f"Required voxel size for GR match = {voxel_size_match:.3e} m")
    R_s = (2 * G * M_sun) / (c**2)
    horizon_index_planck = R_s / voxel_size
    print(f"If voxels are Planck length ({voxel_size:.3e} m):")
    print(f"  Horizon index needed = {horizon_index_planck:.3e}")

    # --- Substrate S and lightspeed at the horizon ---
    S_horizon = S[x_start + horizon_index, cy, cz]
    v_local = S_horizon * c
    print(f"S value at substrate horizon: {S_horizon:.6f}")
    print(f"Expected local light speed at horizon (S * c): {v_local:.3e} m/s")
    print(f"Physical speed of light: {c:.3e} m/s")
    print(f"Ratio (local/physical): {v_local/c:.6f}")

else:
    print("No frozen horizon detected within corridor length.")

# --- Planck mass self-check ---
print("\n--- Planck Mass from Schwarzschild Radius Self-Check ---")
planck_radius = 2 * planck_length
required_mass = (planck_radius * c**2) / (2 * G)
print(f"Planck mass from Schwarzschild radius (sim): {required_mass:.3e} kg")
print(f"Planck mass (physical):                     {planck_mass:.3e} kg")
print(f"Ratio (sim/physical):                       {required_mass/planck_mass:.6f}")