import numpy as np
import random
import sys

# --- Parameters ---
N = 50              # Grid size (cube: N x N x N)
timesteps = 5000    # Number of steps (shorter for testing)
dt = 0.01           # Base timestep
M = 1.0             # True info speed (arbitrary units)
M0 = 1.0            # Normalization for stability

# --- Initialize Fields ---
S = np.ones((N, N, N))    # Entropy everywhere
C = np.zeros((N, N, N))   # Collapse nowhere

def initialize_corridor(S, C, hall_len=20, hall_axis=0, channel_halfwidth=1):
    N = S.shape[0]
    # reset to entropy everywhere
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
                # set walls just outside the channel
                for dy in range(-(channel_halfwidth+1), channel_halfwidth+2):
                    for dz in range(-(channel_halfwidth+1), channel_halfwidth+2):
                        if abs(dy) > channel_halfwidth or abs(dz) > channel_halfwidth:
                            y = cy + dy
                            z = cz + dz
                            if 0 <= y < N and 0 <= z < N:
                                S[x, y, z] = 0.0
                                C[x, y, z] = 1.0

        # add sealed stop at far end (fill channel with collapse)
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

# Example usage:
S, C = initialize_corridor(S, C, hall_len=20, channel_halfwidth=1)

import sys

def print_corridor(S, C, hall_len=5):
    """
    Print the corridor with walls as 1 and channel values in the middle.
    """
    cx, cy, cz = N//2, N//2, N//2
    x_start = cx - hall_len//2

    lines = []
    # sealed stop
    lines.append("1 1 1")
    # corridor rows
    for dx in range(1, hall_len):
        x = x_start + dx
        val = C[x, cy, cz]
        lines.append(f"1 {val:.2f} 1")

    out = "\n".join(lines)
    sys.stdout.write("\033[H\033[J")  # clear screen
    sys.stdout.write(out + "\n")
    sys.stdout.flush()
    
def collapse_influx(S, C, hall_len=20, channel_halfwidth=1, strength=1.0, thickness=1):
    cx, cy, cz = N//2, N//2, N//2
    x_entrance = cx - hall_len//2  # the open end

    for dy in range(-channel_halfwidth, channel_halfwidth+1):
        for dz in range(-channel_halfwidth, channel_halfwidth+1):
            y = cy + dy
            z = cz + dz
            if 0 <= y < N and 0 <= z < N:
                S[x_entrance:x_entrance+thickness, y, z] = 0.0
                C[x_entrance:x_entrance+thickness, y, z] = 1.0
    return S, C
    
def laplacian(S):
    """Discrete 3D Laplacian with 26 neighbors."""
    return (
        np.pad(S, 1, mode='constant')[1:-1,1:-1,1:-1] * -26
        + sum(np.roll(np.roll(np.roll(S, dx, 0), dy, 1), dz, 2)
              for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
              if not (dx == dy == dz == 0))
    )

def step(S, C):
    """One Ricci flow timestep in the substrate model."""
    deltaS = laplacian(S)
    dt_eff = dt * M / M0 * S
    S_new = S + dt_eff * deltaS
    S_new = np.clip(S_new, 0, 1)    # Enforce physical bounds
    C_new = 1 - S_new
    return S_new, C_new

def check_breakdown(S, C):
    """Check for NaNs or out-of-bound values (substrate 'surgery' detection)."""
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
    print_corridor(S, C, hall_len=20)
    if check_breakdown(S, C):
        print(f"Surgery required at step {t}")
        break

print("Simulation complete.")