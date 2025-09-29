import numpy as np
import matplotlib.pyplot as plt

# =============================================
# TEST #10: Field Analog Motifs (Dipole + Tube)
# =============================================

np.random.seed(0)

N = 240
cx, cy = N//2, N//2
x = np.arange(N); y = np.arange(N)
xx, yy = np.meshgrid(x, y, indexing="ij")

# Build an S-field with two low-S wells (dipole) and a narrow low-S tube between them.
def make_dipole_tube(S_bg=0.95, depth=0.6, sep=50, tube_width=6):
    S = np.ones((N, N)) * S_bg
    # Centers
    c1 = (cx - sep//2, cy)
    c2 = (cx + sep//2, cy)
    # Wells
    r1 = np.hypot(xx - c1[0], yy - c1[1])
    r2 = np.hypot(xx - c2[0], yy - c2[1])
    S -= depth * np.exp(-(r1/12.0)**2)
    S -= depth * np.exp(-(r2/12.0)**2)
    # Tube: carve a corridor of lower S between wells
    tube = (np.abs(yy - cy) < tube_width) & (xx > c1[0]) & (xx < c2[0])
    S[tube] -= 0.35
    return np.clip(S, 0.0, 0.99)

S = make_dipole_tube()
C = 1 - S

# Streamline-like visualization via test particles
num_lines = 200
starts_y = np.linspace(20, N-20, num_lines)
particles = [{"pos": np.array([20.0, sy]), "vel": np.array([1.0, 0.0]) * 0.4, "trail": []} for sy in starts_y]

def gradC(ix, iy):
    ix = np.clip(ix, 1, N-2); iy = np.clip(iy, 1, N-2)
    gx = (C[ix+1, iy] - C[ix-1, iy]) * 0.5
    gy = (C[ix, iy+1] - C[ix, iy-1]) * 0.5
    return gx, gy

steps = 600
for _ in range(steps):
    for p in particles:
        x0, y0 = p["pos"]
        ix, iy = int(round(x0)), int(round(y0))
        gx, gy = gradC(ix, iy)
        acc = np.array([gx, gy])
        p["vel"] += acc
        sp = np.linalg.norm(p["vel"])
        cap = 1.0
        if sp > cap:
            p["vel"] = p["vel"]/sp * cap
        p["pos"] += p["vel"]
        p["pos"][0] = np.clip(p["pos"][0], 1, N-2)
        p["pos"][1] = np.clip(p["pos"][1], 1, N-2)
        p["trail"].append(p["pos"].copy())

plt.figure(figsize=(8,8))
plt.imshow(S.T, origin="lower")
for p in particles[::4]:
    tr = np.array(p["trail"])
    plt.plot(tr[:,0], tr[:,1], lw=0.8)
plt.title("Dipole + Low-S Tube: Streamline-like Particle Paths")
plt.colorbar(label="Entropy S")
plt.xlabel("x"); plt.ylabel("y")
plt.show()

# Show confinement by launching particles inside the tube
tube_particles = []
for y0 in np.linspace(cy-4, cy+4, 15):
    tube_particles.append({"pos": np.array([cx - 20.0, y0]), "vel": np.array([1.0, 0.0])*0.2, "trail": []})

for _ in range(400):
    for p in tube_particles:
        ix, iy = int(round(p["pos"][0])), int(round(p["pos"][1]))
        gx, gy = gradC(ix, iy)
        p["vel"] += np.array([gx, gy])
        sp = np.linalg.norm(p["vel"])
        if sp > 1.0: p["vel"] = p["vel"]/sp
        p["pos"] += p["vel"]
        p["pos"][0] = np.clip(p["pos"][0], 1, N-2)
        p["pos"][1] = np.clip(p["pos"][1], 1, N-2)
        p["trail"].append(p["pos"].copy())

plt.figure(figsize=(8,8))
plt.imshow(S.T, origin="lower")
for p in tube_particles:
    tr = np.array(p["trail"])
    plt.plot(tr[:,0], tr[:,1], lw=1.2)
plt.title("Confinement Along Low-S Tube (Flux-Tube Analogue)")
plt.colorbar(label="Entropy S")
plt.xlabel("x"); plt.ylabel("y")
plt.show()


# ===================================================
# TEST #11: Emergent Near-Conservation in Closed Runs
# ===================================================

# Static field: one smooth low-S well; reflective boundaries; no drag; high cap to avoid clipping
N2 = 220
cx2, cy2 = N2//2, N2//2
xx2, yy2 = np.meshgrid(np.arange(N2), np.arange(N2), indexing="ij")
r2 = np.hypot(xx2 - cx2, yy2 - cy2)

S2 = 0.98 - 0.78*np.exp(-(r2/22.0)**2)
S2 = np.clip(S2, 0.2, 0.99)
C2 = 1 - S2

def gradC2(ix, iy):
    ix = np.clip(ix, 1, N2-2); iy = np.clip(iy, 1, N2-2)
    gx = (C2[ix+1, iy] - C2[ix-1, iy]) * 0.5
    gy = (C2[ix, iy+1] - C2[ix, iy-1]) * 0.5
    return gx, gy

# Initialize many particles with random positions/velocities inside box
m = 600
px = np.random.uniform(20, N2-20, size=m)
py = np.random.uniform(20, N2-20, size=m)
vx = np.random.uniform(-0.3, 0.3, size=m)
vy = np.random.uniform(-0.3, 0.3, size=m)

dt = 1.0; stepsE = 1500
cap = 2.5  # high so we rarely hit it

E_total = []
K_total = []
U_total = []

for t in range(stepsE):
    ix = np.clip(px.astype(int), 1, N2-2)
    iy = np.clip(py.astype(int), 1, N2-2)
    gx = (C2[ix+1, iy] - C2[ix-1, iy]) * 0.5
    gy = (C2[ix, iy+1] - C2[ix, iy-1]) * 0.5
    # Force = +∇C; Conservative potential U = -C so F = -∇U = ∇C
    vx += gx * dt
    vy += gy * dt
    sp = np.sqrt(vx*vx + vy*vy)
    hit = sp > cap
    if np.any(hit):
        vx[hit] = vx[hit]/sp[hit]*cap
        vy[hit] = vy[hit]/sp[hit]*cap
    px += vx * dt
    py += vy * dt
    # Reflective boundaries
    outx_low = px < 1; outx_high = px > N2-2
    outy_low = py < 1; outy_high = py > N2-2
    vx[outx_low | outx_high] *= -1.0
    vy[outy_low | outy_high] *= -1.0
    px = np.clip(px, 1, N2-2); py = np.clip(py, 1, N2-2)
    # Energies
    K = 0.5 * np.mean(vx*vx + vy*vy)
    U = np.mean(-C2[ix, iy])  # U = -C
    K_total.append(K)
    U_total.append(U)
    E_total.append(K + U)

E_total = np.array(E_total); K_total = np.array(K_total); U_total = np.array(U_total)
drift_pct = (E_total[-1] - E_total[0]) / max(1e-12, abs(E_total[0])) * 100.0

plt.figure(figsize=(9,5))
plt.plot(E_total, label="Total E = K + U")
plt.plot(K_total, label="Kinetic (K)")
plt.plot(U_total, label="Potential (U = -C)")
plt.xlabel("Step"); plt.ylabel("Mean energy per particle (sim units)")
plt.title(f"Near-Conservation in Closed System (drift ≈ {drift_pct:.3f}%)")
plt.legend(); plt.grid(True)
plt.show()