# hill_isotropization_test.py
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# -------------------------------
# 1) Build substrate hill
# -------------------------------
N = 600
cx, cy = N//2, N//2

xx, yy = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
# Offset the hill center far to the left so our patch sits on a clear slope
hill_cx = cx - 200
r = np.sqrt((xx - hill_cx)**2 + (yy - cy)**2)

# Smooth S field that increases with distance from hill center
S = 0.45 + 0.55*(r / r.max())
S = np.clip(S, 0.0, 0.99)
C = 1 - S

def gradC(ix, iy):
    ix = np.clip(ix, 1, N-2)
    iy = np.clip(iy, 1, N-2)
    gx = (C[ix+1,iy] - C[ix-1,iy]) * 0.5
    gy = (C[ix,iy+1] - C[ix,iy-1]) * 0.5
    return gx, gy

# -------------------------------
# 2) Tracer patch ("our universe")
# -------------------------------
num_p = 2000
# Small-to-midsize patch so we can see anisotropy emerge over time
r0 = rng.uniform(15, 45, size=num_p)
th0 = rng.uniform(0, 2*np.pi, size=num_p)
px = cx + r0*np.cos(th0)
py = cy + r0*np.sin(th0)
vx = np.zeros(num_p)
vy = np.zeros(num_p)

# Record initial radii and angles relative to observer (cx,cy)
r_init = np.sqrt((px-cx)**2 + (py-cy)**2)
ang_init = (np.arctan2(py-cy, px-cx)) % (2*np.pi)

# -------------------------------
# 3) Evolve under ∇C (the hill)
# -------------------------------
steps = 450
G = 1.0
speed_cap = 1.0

for _ in range(steps):
    # velocity update from ∇C
    ix = np.clip(px.astype(int), 1, N-2)
    iy = np.clip(py.astype(int), 1, N-2)
    gx = (C[ix+1,iy] - C[ix-1,iy]) * 0.5
    gy = (C[ix,iy+1] - C[ix,iy-1]) * 0.5
    vx += G*gx
    vy += G*gy
    # cap speed
    sp = np.sqrt(vx*vx + vy*vy)
    over = sp > speed_cap
    vx[over] = vx[over] / sp[over] * speed_cap
    vy[over] = vy[over] / sp[over] * speed_cap
    # move
    px += vx
    py += vy

# final radii/angles
r_fin = np.sqrt((px-cx)**2 + (py-cy)**2)
ang_fin = (np.arctan2(py-cy, px-cx)) % (2*np.pi)

# Apparent directional scale factor per tracer
scale = r_fin / np.maximum(1e-9, r_init)

# -------------------------------
# 4) Bin by angle: raw anisotropy
# -------------------------------
num_bins = 24  # 15° bins
edges = np.linspace(0, 2*np.pi, num_bins+1)
centers = 0.5*(edges[:-1] + edges[1:])
bin_idx = np.clip(np.searchsorted(edges, ang_init, side='right')-1, 0, num_bins-1)

raw_mean = np.zeros(num_bins)
raw_sem  = np.zeros(num_bins)
for b in range(num_bins):
    sel = (bin_idx == b)
    if np.any(sel):
        raw_mean[b] = np.mean(scale[sel])
        raw_sem[b]  = np.std(scale[sel]) / np.sqrt(np.sum(sel))
    else:
        raw_mean[b] = np.nan
        raw_sem[b]  = np.nan

# -------------------------------
# 5) "Cosmo cleaning" steps
#    a) Cut local region (peculiar velocities)
#    b) Shell-average
#    c) Subtract dipole/quadrupole (bulk flow + shear)
# -------------------------------

# a) Cut local: remove tracers with initial radius below Rmin
Rmin = 25.0  # tune as desired; larger = stronger local cut
far_sel = r_init >= Rmin

# b) Shell-average by final radius to mimic distance binning
# define shells in r_fin (coarse to reduce local clumpiness)
shell_edges = np.linspace(np.percentile(r_fin[far_sel], 5), np.percentile(r_fin[far_sel], 95), 8)
shell_idx = np.clip(np.searchsorted(shell_edges, r_fin, side='right')-1, 0, len(shell_edges)-2)

# within each shell, compute directional mean, then average shells
clean_mean = np.zeros(num_bins)
counts = np.zeros(num_bins)

for s in range(len(shell_edges)-1):
    in_shell = far_sel & (shell_idx == s)
    if not np.any(in_shell): 
        continue
    # direction bins within shell
    b_idx = np.clip(np.searchsorted(edges, ang_init[in_shell], side='right')-1, 0, num_bins-1)
    for b in range(num_bins):
        ss = (b_idx == b)
        if np.any(ss):
            clean_mean[b] += np.mean(scale[in_shell][ss])
            counts[b] += 1

# average across shells
with np.errstate(invalid='ignore'):
    clean_mean = np.where(counts>0, clean_mean/np.maximum(1,counts), np.nan)

# c) Fit and subtract dipole + quadrupole from the angular profile
# Model: f(θ) = A + B cosθ + C sinθ + D cos2θ + E sin2θ
theta = centers
def fit_dip_quad(y):
    M = np.column_stack([
        np.ones_like(theta),
        np.cos(theta), np.sin(theta),
        np.cos(2*theta), np.sin(2*theta)
    ])
    mask = np.isfinite(y)
    coef, *_ = np.linalg.lstsq(M[mask], y[mask], rcond=None)
    fit = M @ coef
    return coef, fit

coef_raw, fit_raw = fit_dip_quad(raw_mean)
coef_cln, fit_cln = fit_dip_quad(clean_mean)

resid_raw = raw_mean - fit_raw
resid_cln = clean_mean - fit_cln

# -------------------------------
# 6) Simple angular "power" metric
#    Discrete Fourier series magnitude (low-l removal should flatten)
# -------------------------------
def ang_power(y):
    # replace nans with mean for FFT stability
    yy = y.copy()
    m = np.isfinite(yy)
    yy[~m] = np.nanmean(yy[m]) if np.any(m) else 0.0
    Y = np.fft.rfft(yy - np.mean(yy))
    P = np.abs(Y)**2
    return P

P_raw  = ang_power(raw_mean)
P_cln  = ang_power(clean_mean)
P_rres = ang_power(resid_raw)
P_cres = ang_power(resid_cln)
k = np.arange(len(P_raw))  # angular harmonic index (0=monopole, 1=dipole, 2=quad,...)

# -------------------------------
# 7) Plots
# -------------------------------
plt.figure(figsize=(6,6))
plt.imshow(S.T, origin='lower', cmap='plasma')
plt.scatter(px, py, s=1, c='white')
plt.title("Final tracer positions on substrate hill")
plt.colorbar(label="S")
plt.tight_layout()
plt.show()

# Raw vs cleaned directional means
deg_centers = centers * 180/np.pi
plt.figure(figsize=(9,4))
plt.plot(deg_centers, raw_mean, 'o-', label='Raw directional scale')
plt.plot(deg_centers, fit_raw,  '--', label='Raw dipole+quad fit')
plt.plot(deg_centers, clean_mean, 's-', label='After local cut + shell avg')
plt.plot(deg_centers, fit_cln,  '--', label='Clean dipole+quad fit')
plt.xlabel("Direction (deg)")
plt.ylabel("⟨ a(θ) ⟩")
plt.title("Directional expansion: raw vs. cosmology-style cleaning")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# Residuals after removing dipole+quadrupole
plt.figure(figsize=(9,4))
plt.plot(deg_centers, resid_raw, 'o-', label='Raw residual (minus l=0..2)')
plt.plot(deg_centers, resid_cln, 's-', label='Clean residual (minus l=0..2)')
plt.axhline(0, color='k', lw=0.8)
plt.xlabel("Direction (deg)")
plt.ylabel("Residual ⟨ a(θ) ⟩")
plt.title("Residual anisotropy after removing monopole/dipole/quadrupole")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# Angular power before/after, and of residuals
plt.figure(figsize=(9,4))
plt.semilogy(k, P_raw,  'o-', label='Raw power')
plt.semilogy(k, P_cln,  's-', label='Clean power')
plt.semilogy(k, P_rres, 'o--', label='Raw residual power')
plt.semilogy(k, P_cres, 's--', label='Clean residual power')
plt.xlabel("Harmonic index k  (k=0 monopole, 1 dipole, 2 quadrupole, ...)")
plt.ylabel("Angular power")
plt.title("Angular spectra: flattening after cosmology-style cleaning")
plt.grid(True, which='both'); plt.legend(); plt.tight_layout(); plt.show()

# Console summary
A_raw, Bc_raw, Bs_raw, D_raw, E_raw = coef_raw
A_cln, Bc_cln, Bs_cln, D_cln, E_cln = coef_cln
print("Dipole amplitude (raw)     ~", np.hypot(Bc_raw, Bs_raw))
print("Quadrupole amplitude (raw) ~", np.hypot(D_raw, E_raw))
print("Dipole amplitude (clean)   ~", np.hypot(Bc_cln, Bs_cln))
print("Quadrupole amplitude (clean)~", np.hypot(D_cln, E_cln))
print("Note: residuals should look much flatter after cleaning; angular power at low k should drop.")