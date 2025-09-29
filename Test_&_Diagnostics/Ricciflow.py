import numpy as np
import random

# --- Parameters ---
N = 50            # Grid size (cube: N x N x N)
timesteps = 500   # Number of steps
dt = 0.01         # Time step
M = 1.0           # True info speed (arbitrary units)
M0 = 1.0          # Normalization for stability

# --- Initialize Fields ---
S = np.ones((N, N, N))    # Entropy: everywhere possible
C = np.zeros((N, N, N))   # Collapse: everywhere unresolved

def initialize_dumbbell(S, C, radius1=8, radius2=8, neck_radius=2, neck_length=12):
    """Dumbbell: two spheres with a narrow neck between."""
    cx, cy, cz = N//2, N//2, N//2
    # Sphere 1
    for x in range(N):
        for y in range(N):
            for z in range(N):
                if ((x - cx - neck_length//2)**2 + (y - cy)**2 + (z - cz)**2) < radius1**2:
                    S[x,y,z] = 0.5
                    C[x,y,z] = 0.5
    # Sphere 2
    for x in range(N):
        for y in range(N):
            for z in range(N):
                if ((x - cx + neck_length//2)**2 + (y - cy)**2 + (z - cz)**2) < radius2**2:
                    S[x,y,z] = 0.5
                    C[x,y,z] = 0.5
    # Neck
    for dx in range(-neck_length//2, neck_length//2):
        for y in range(N):
            for z in range(N):
                if ((y - cy)**2 + (z - cz)**2) < neck_radius**2:
                    x = cx + dx
                    if 0 <= x < N:
                        S[x,y,z] = 0.5
                        C[x,y,z] = 0.5
    return S, C

S, C = initialize_dumbbell(S, C)   # Change to other initializations as desired

def laplacian(S):
    """Discrete 3D Laplacian with 26 neighbors."""
    kernel = np.ones((3,3,3))
    kernel[1,1,1] = -26
    return np.pad(S, 1, mode='constant')[1:-1,1:-1,1:-1] * -26 + \
           sum(np.roll(np.roll(np.roll(S, dx, 0), dy, 1), dz, 2)
               for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1]
               if not (dx == dy == dz == 0))

def step(S, C):
    deltaS = laplacian(S)
    c_local = M * S
    dt_eff = dt * M / M0 * S
    S_new = S + dt_eff * deltaS
    S_new = np.clip(S_new, 0, 1)    # Physical limits
    C_new = 1 - S_new
    return S_new, C_new

def entropy_bomb(S, C, n_bombs=3, size=2):
    """Drop n_bombs bombs of random pure entropy (S=1) or pure collapse (S=0)."""
    for _ in range(n_bombs):
        x = random.randint(size, N-size-1)
        y = random.randint(size, N-size-1)
        z = random.randint(size, N-size-1)
        # Randomly pick type of bomb
        if random.random() < 0.5:
            S[x-size:x+size, y-size:y+size, z-size:z+size] = 1.0   # Entropy bomb!
            C[x-size:x+size, y-size:y+size, z-size:z+size] = 0.0
        else:
            S[x-size:x+size, y-size:y+size, z-size:z+size] = 0.0   # Collapse bomb!
            C[x-size:x+size, y-size:y+size, z-size:z+size] = 1.0
    return S, C

# --- Diagnostics ---
def check_breakdown(S, C):
    if np.isnan(S).any() or np.isnan(C).any():
        print("Breakdown detected: NaN values!")
        return True
    if np.max(S) > 1.01 or np.min(S) < -0.01:
        print("Breakdown detected: S out of bounds!")
        return True
    return False

def curvature_energy(S):
    ΔS = laplacian(S)
    return float((ΔS**2).sum())
    
# --- Run Simulation ---
for t in range(timesteps):
    S, C = step(S, C)
    if t % 100 == 0:
        S, C = entropy_bomb(S, C, n_bombs=5, size=3)
    if check_breakdown(S, C):
        print(f"Surgery required at step {t}")
        break
    if t % 50 == 0:
        e = curvature_energy(S)
        print(f"Step {t}: S.mean={S.mean():.4f}  S.min={S.min():.4f}  E={e:.3e}  max_dt_eff={(dt*S).max():.3e}")

print("Simulation complete.")