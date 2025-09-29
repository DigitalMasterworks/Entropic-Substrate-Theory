import numpy as np
import matplotlib.pyplot as plt

# This is a simplified re-implementation of the high-S repulsion ("dark matter") simulation,
# extended to test if lensing (path curvature) is also recovered when high-S halos are present.

GRID_SIZE = 100
NUM_STEPS = 200
C = 1.0  # speed cap
G = 1.0  # coupling

# --- Create entropy field with central low-S mass and surrounding high-S halo ---
entropy_field = np.ones((GRID_SIZE, GRID_SIZE)) * 0.9  # baseline high entropy
cx, cy = GRID_SIZE // 2, GRID_SIZE // 2

# Low-entropy center (collapse well)
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        if dist < 10:
            entropy_field[x, y] = 0.2  # strong collapse center
        elif 15 < dist < 25:
            entropy_field[x, y] = 1.0  # high-S halo (repulsive zone)

collapse_field = 1 - entropy_field

# --- Set up photon-like particles ---
num_rays = 7
impact_params = np.linspace(-15, 15, num_rays)
particles = []
for b in impact_params:
    pos = np.array([0.0, cy + b])
    vel = np.array([1.0, 0.0]) * C  # move to +x
    particles.append({"pos": pos.copy(), "vel": vel.copy(), "trail": [pos.copy()]})

# --- Helper to get local collapse value with bounds check ---
def local_C(ix, iy):
    if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
        return collapse_field[ix, iy]
    return 0.0

# --- Main integration loop ---
for _ in range(NUM_STEPS):
    for p in particles:
        x, y = p["pos"]
        ix, iy = int(round(x)), int(round(y))
        # Compute force from gradient of C
        if 1 <= ix < GRID_SIZE - 1 and 1 <= iy < GRID_SIZE - 1:
            dCdx = (local_C(ix + 1, iy) - local_C(ix - 1, iy)) / 2.0
            dCdy = (local_C(ix, iy + 1) - local_C(ix, iy - 1)) / 2.0
            force = np.array([dCdx, dCdy]) * G
        else:
            force = np.zeros(2)
        # Update velocity and cap speed
        p["vel"] += force
        speed = np.linalg.norm(p["vel"])
        if speed > C:
            p["vel"] = p["vel"] / speed * C
        # Update position
        p["pos"] += p["vel"]
        p["trail"].append(p["pos"].copy())

# --- Plot results ---
plt.figure(figsize=(8, 8))
# Plot entropy field as background
plt.imshow(entropy_field.T, origin='lower', cmap='plasma', alpha=0.5)
# Plot each ray's path
for p in particles:
    trail = np.array(p["trail"])
    plt.plot(trail[:, 0], trail[:, 1], lw=1.5)
# Mark center
plt.scatter([cx], [cy], color='black', marker='o', s=50, label="Central mass")
plt.colorbar(label="Entropy S")
plt.title("Photon Paths in Collapse + High-S Halo Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.show()