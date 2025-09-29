import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Grid size
N = 500
cx, cy = N//2, N//2

# Build a smooth S/C gradient hill with peak offset so we're on the slope
xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
r = np.sqrt((xx - (cx - 120))**2 + (yy - cy)**2)  # big offset so slope curvature is visible
S = 0.5 + 0.5 * (r / r.max())
S = np.clip(S, 0.0, 0.99)
C = 1 - S

# Particle tracers in "our universe" patch
num_p = 400
theta = np.random.rand(num_p) * 2 * np.pi
radius = np.random.uniform(10, 30, num_p)
px = cx + radius * np.cos(theta)
py = cy + radius * np.sin(theta)
vx = np.zeros(num_p)
vy = np.zeros(num_p)

# Gradient function
def gradC(ix, iy):
    ix = np.clip(ix, 1, N-2)
    iy = np.clip(iy, 1, N-2)
    gx = (C[ix+1, iy] - C[ix-1, iy]) * 0.5
    gy = (C[ix, iy+1] - C[ix, iy-1]) * 0.5
    return gx, gy

# Simulation parameters
steps = 300
G = 1.0
speed_cap = 1.0
initial_r = np.sqrt((px - cx)**2 + (py - cy)**2)

# Measure scale factor by direction bins
num_bins = 12  # 30° each
bin_angles = np.linspace(0, 2*np.pi, num_bins+1)
bin_scale = [[] for _ in range(num_bins)]

for step in range(steps):
    # Update velocities
    for i in range(num_p):
        gx, gy = gradC(int(px[i]), int(py[i]))
        vx[i] += gx * G
        vy[i] += gy * G
        speed = np.sqrt(vx[i]**2 + vy[i]**2)
        if speed > speed_cap:
            vx[i] *= speed_cap / speed
            vy[i] *= speed_cap / speed

    # Move particles
    px += vx
    py += vy

    # Measure scale factor in angular bins
    for i in range(num_p):
        dx, dy = px[i] - cx, py[i] - cy
        ang = np.arctan2(dy, dx) % (2*np.pi)
        r_now = np.sqrt(dx**2 + dy**2)
        # find bin
        b = np.searchsorted(bin_angles, ang) - 1
        b = max(0, min(num_bins-1, b))
        if step == 0:
            scale = 1.0
        else:
            scale = r_now / initial_r[i]
        bin_scale[b].append(scale)

# Average scale per bin
avg_scale_by_bin = []
for b in range(num_bins):
    avg_scale_by_bin.append(np.mean(bin_scale[b]))

# Plot S-field with particle positions
fig1 = plt.figure(figsize=(6,6))
plt.imshow(S.T, origin='lower')  # default colormap, no explicit colors
plt.scatter(px, py, s=5)         # default color
plt.title("Final tracer positions on substrate hill")
plt.colorbar(label="S")
out1 = Path("/mnt/data/substrate_hill_tracers.png")
plt.savefig(out1, bbox_inches='tight', dpi=150)
plt.close(fig1)

# Plot expansion by direction
angles_deg = (bin_angles[:-1] + bin_angles[1:]) / 2 * 180/np.pi
fig2 = plt.figure(figsize=(8,4))
plt.bar(angles_deg, avg_scale_by_bin, width=30, align='center')
plt.xlabel("Direction (degrees from 'uphill')")
plt.ylabel("Average scale factor a(t)/a0")
plt.title(f"Directional Expansion After {steps} Steps")
plt.grid(True)
out2 = Path("/mnt/data/directional_expansion_bar.png")
plt.savefig(out2, bbox_inches='tight', dpi=150)
plt.close(fig2)

print("Saved outputs:")
print(f"- {out1}")
print(f"- {out2}")