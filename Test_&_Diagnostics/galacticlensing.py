# Extend the "reality engine" simulation to include photon lensing through the final halo structure

# Use the final C and S from the last black-hole-driven halo formation test
C_field_final = C.copy()
S_field_final = 1 - C_field_final

# --- Photon setup ---
num_rays = 21
impact_params = np.linspace(-80, 80, num_rays)
photons = []
for b in impact_params:
    photons.append({
        "pos": np.array([0.0, cy + b]),
        "vel": np.array([1.0, 0.0]) * speed_cap,
        "trail": [np.array([0.0, cy + b])]
    })

# Helper to get grad(C) with safe indexing
def grad_C(ix, iy):
    ix = np.clip(ix, 1, GRID-2)
    iy = np.clip(iy, 1, GRID-2)
    dCdx = (C_field_final[ix+1, iy] - C_field_final[ix-1, iy]) * 0.5
    dCdy = (C_field_final[ix, iy+1] - C_field_final[ix, iy-1]) * 0.5
    return dCdx, dCdy

# --- Integrate photon motion ---
steps_rays = 400
for _ in range(steps_rays):
    for p in photons:
        x, y = p["pos"]
        ix, iy = int(round(x)), int(round(y))
        fx, fy = grad_C(ix, iy)
        force = np.array([fx, fy]) * G
        p["vel"] += force
        spd = np.linalg.norm(p["vel"])
        if spd > speed_cap:
            p["vel"] = p["vel"] / spd * speed_cap
        p["pos"] += p["vel"]
        p["trail"].append(p["pos"].copy())

# --- Plot the lensing results over the halo field ---
plt.figure(figsize=(8, 8))
plt.imshow(S_field_final.T, origin='lower', cmap='plasma', alpha=0.5)
for p in photons:
    tr = np.array(p["trail"])
    plt.plot(tr[:, 0], tr[:, 1], color='cyan', lw=0.8)
plt.title("Photon Lensing through Black-Hole-Induced Halo")
plt.colorbar(label="Entropy S")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --- Measure bending angles ---
bending_angles = []
for p in photons:
    tr = np.array(p["trail"])
    dy = tr[-1, 1] - tr[0, 1]
    dx = tr[-1, 0] - tr[0, 0]
    angle = np.degrees(np.arctan2(dy, dx))
    bending_angles.append(angle)

plt.figure(figsize=(8, 5))
plt.plot(impact_params, bending_angles, marker='o', color='magenta')
plt.xlabel("Impact Parameter (pixels)")
plt.ylabel("Bending Angle (deg)")
plt.title("Lensing Deflection vs Impact Parameter (BH Reality Engine Halo)")
plt.grid(True)
plt.show()