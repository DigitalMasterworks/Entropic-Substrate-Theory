import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# TEST #4: Time dilation & gravitational redshift in S/C
# =====================================================

# Field: low-S well in otherwise near-vacuum
N = 220
cx, cy = N//2, N//2
x = np.arange(N); y = np.arange(N)
xx, yy = np.meshgrid(x, y, indexing="ij")
r = np.hypot(xx - cx, yy - cy)

# Smooth collapse well
S = 0.98 - 0.78*np.exp(-(r/18.0)**2)   # S in [~0.2, ~0.98]
S = np.clip(S, 0.2, 0.99)
C = 1 - S

# --- Time dilation via "clock particles" ---
# Local proper-time increment dτ = S * dt (operational bridge)
dt = 1.0
T = 1200
radii = np.linspace(3, 80, 16)
clock_tau = np.zeros_like(radii)

for i, rr in enumerate(radii):
    ix = np.clip(int(cx + rr), 0, N-1)
    iy = cy
    local_S = S[ix, iy]
    clock_tau[i] = np.sum(local_S * dt for _ in range(T))  # = T * S

# Normalize vs far-away clock (r ~ 80)
tau_far = clock_tau[-1]
dilation = clock_tau / tau_far  # τ(r)/τ_far

plt.figure(figsize=(7,5))
plt.plot(radii, dilation, marker="o")
plt.xlabel("Radius from core (pixels)")
plt.ylabel("Clock rate vs far clock (τ/τ_far)")
plt.title("Time Dilation from Substrate S(r)")
plt.grid(True)
plt.show()

# --- Gravitational redshift experiment ---
# Emitter at r_e produces local-frequency f0 (in proper time).
# Observer at r_o>> measures received frequency f_obs (in their τ).
# Expect scaling ~ S_e / S_o in this operational mapping.

# Choose emitter radius and observer location
r_e = 10.0
emit_ix = int(cx + r_e); emit_iy = cy
S_e = S[emit_ix, emit_iy]

r_o = 80.0
obs_ix = int(cx + r_o); obs_iy = cy
S_o = S[obs_ix, obs_iy]

# Wave emission in local proper time
f0_local = 0.05  # cycles per unit proper time at emitter
phase_emit = 0.0

# Photon packet modeled as phase fronts moving at speed cap along +x
speed_cap = 1.0
pos = np.array([emit_ix, emit_iy], dtype=float)
vel = np.array([1.0, 0.0]) * speed_cap

# Record observer's received phase vs their local time
obs_times = []
obs_phase_samples = []

# Simulate propagation & ticking
Tsteps = 600
for t in range(Tsteps):
    # Emitter advances phase by dτ_e = S_e * dt
    phase_emit += 2*np.pi * f0_local * (S_e * dt)
    # Move photon
    pos += vel
    ix, iy = int(np.clip(pos[0], 0, N-1)), int(np.clip(pos[1], 0, N-1))
    # When crossing observer x, sample phase in observer local time
    if ix >= obs_ix and len(obs_times) < 200:
        # Observer local time increment
        if not obs_times:
            obs_times.append(0.0)
        else:
            obs_times.append(obs_times[-1] + S_o * dt)
        obs_phase_samples.append(phase_emit)

# Measure observed frequency by linear fit to phase(t_obs)
obs_times = np.array(obs_times)
obs_phase_samples = np.unwrap(np.array(obs_phase_samples))
if len(obs_times) > 5:
    A = np.vstack([obs_times, np.ones_like(obs_times)]).T
    slope, intercept = np.linalg.lstsq(A, obs_phase_samples, rcond=None)[0]
    f_obs = slope / (2*np.pi)  # cycles per local-time unit at observer
else:
    f_obs = np.nan

pred_ratio = S_e / S_o
meas_ratio = f_obs / f0_local if f_obs == f_obs else np.nan  # check for NaN

plt.figure(figsize=(7,5))
plt.plot(obs_times, obs_phase_samples)
plt.xlabel("Observer local time")
plt.ylabel("Unwrapped phase at observer")
plt.title("Gravitational Redshift: Phase vs Observer Time")
plt.grid(True)
plt.show()

print("Time dilation (near core vs far) examples:")
for rr, dil in zip(radii[:4], dilation[:4]):
    print(f"  r≈{rr:.1f} -> τ/τ_far ≈ {dil:.3f}")
print(f"\nRedshift ratio predicted S_e/S_o ≈ {pred_ratio:.4f}")
print(f"Measured f_obs/f0 ≈ {meas_ratio:.4f}")


# =======================================
# TEST #20: High-S Dark Void (repulsion)
# =======================================

# Construct a high-S bubble in a lower-S background to test repulsion and boundary effects.
N2 = 240
cx2, cy2 = N2//2, N2//2
x2 = np.arange(N2); y2 = np.arange(N2)
xx2, yy2 = np.meshgrid(x2, y2, indexing="ij")
r2 = np.hypot(xx2 - cx2, yy2 - cy2)

S_bg = 0.75
S_void = 0.98
R_void = 45.0
# Smooth wall around the bubble
wall = np.exp(-((r2 - R_void)/6.0)**2)
S2 = S_bg + (S_void - S_bg) * (r2 <= R_void) + 0.18*wall*(r2 > R_void)
S2 = np.clip(S2, 0.0, 0.99)
C2 = 1 - S2

# Matter tracers launched toward the void
num_tr = 800
theta = 2*np.pi*np.random.rand(num_tr)
r_start = np.random.uniform(70, 100, size=num_tr)
px = cx2 + r_start*np.cos(theta)
py = cy2 + r_start*np.sin(theta)
# Inward radial bias
vx = -0.4*np.cos(theta)
vy = -0.4*np.sin(theta)

def gradC2(ix, iy):
    ix = np.clip(ix, 1, N2-2); iy = np.clip(iy, 1, N2-2)
    gx = (C2[ix+1, iy] - C2[ix-1, iy])*0.5
    gy = (C2[ix, iy+1] - C2[ix, iy-1])*0.5
    return gx, gy

# Integrate particle motion
dt2 = 1.0; steps2 = 420; G2 = 1.0; cap2 = 1.0; drag2 = 0.02
for _ in range(steps2):
    ix = np.clip(px.astype(int), 1, N2-2)
    iy = np.clip(py.astype(int), 1, N2-2)
    gx = (C2[ix+1, iy] - C2[ix-1, iy])*0.5
    gy = (C2[ix, iy+1] - C2[ix, iy-1])*0.5
    vx += gx*G2; vy += gy*G2
    sp = np.sqrt(vx*vx + vy*vy) + 1e-12
    scale = np.minimum(1.0, cap2/sp)
    vx *= scale*(1 - drag2); vy *= scale*(1 - drag2)
    px += vx*dt2; py += vy*dt2
    px = np.clip(px, 1, N2-2); py = np.clip(py, 1, N2-2)

# Plot final positions over S-field
plt.figure(figsize=(7,7))
plt.imshow(S2.T, origin="lower")
plt.scatter(px, py, s=2)
plt.title("High-S Void Repels Matter to Boundary (Dark Void Dynamics)")
plt.colorbar(label="S")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

# Simple lensing test across void boundary
num_rays = 21
impact = np.linspace(-100, 100, num_rays)
rays = []
for b in impact:
    rays.append({"pos": np.array([0.0, cy2 + b], float), "vel": np.array([1.0, 0.0], float), "trail": []})

for _ in range(500):
    for RY in rays:
        x, y = RY["pos"]
        ix, iy = int(np.clip(x, 1, N2-2)), int(np.clip(y, 1, N2-2))
        gx, gy = gradC2(ix, iy)
        acc = np.array([gx, gy])*1.0
        RY["vel"] += acc
        sp = np.linalg.norm(RY["vel"])
        if sp > 1.0:
            RY["vel"] = RY["vel"]/sp
        RY["pos"] += RY["vel"]
        RY["trail"].append(RY["pos"].copy())

plt.figure(figsize=(8,8))
plt.imshow(S2.T, origin="lower")
for RY in rays:
    tr = np.array(RY["trail"])
    plt.plot(tr[:,0], tr[:,1], lw=0.8)
plt.title("Photon Paths Skirting a High-S Void")
plt.colorbar(label="S")
plt.xlabel("x"); plt.ylabel("y")
plt.show()