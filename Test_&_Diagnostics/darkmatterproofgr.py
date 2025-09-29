from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Rotation Curve Analysis ---
# Newtonian fit (v ~ r^-0.5) for inner radii
def newtonian_v(r, k):
    return k / np.sqrt(r)

inner_mask = radii_arr < flatten_radius
popt_newton, _ = curve_fit(newtonian_v, radii_arr[inner_mask], velocities[inner_mask])

# Flat fit for outer region
outer_mask = radii_arr >= flatten_radius
v_flat = np.mean(velocities[outer_mask])

# Plot rotation curve
plt.figure(figsize=(8,5))
plt.scatter(radii_arr, velocities, label="Simulated velocities", color="blue")
plt.plot(radii_arr, newtonian_v(radii_arr, *popt_newton), 'r--', label="Newtonian fit (inner)")
plt.axhline(v_flat, color='green', linestyle='--', label=f"Flat fit (outer) ~{v_flat:.3f}")
plt.axvline(flatten_radius, color='gray', linestyle=':', label=f"Flatten radius ~{flatten_radius:.1f}")
plt.xlabel("Radius (pixels)")
plt.ylabel("Orbital velocity (sim units)")
plt.title("Rotation Curve: Inner Newtonian vs Outer Flat")
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Lensing Angle Analysis ---
# GR point-mass profile in sim units: alpha(b) = A / b
def gr_lens(b, A):
    return A / b

# Fit to |impact param| > small radius to avoid singularity
mask_fit = np.abs(impact_params) > 5
popt_gr, _ = curve_fit(gr_lens, np.abs(impact_params[mask_fit]), np.abs(bending_norm[mask_fit]))

A_fit = popt_gr[0]

# Plot bending
b_vals = np.linspace(5, 40, 100)
plt.figure(figsize=(8,5))
plt.scatter(np.abs(impact_params), np.abs(bending_norm), label="Simulated deflection", color="purple")
plt.plot(b_vals, gr_lens(b_vals, A_fit), 'orange', linestyle='--', label=f"GR fit A={A_fit:.2f}")
plt.xlabel("Impact parameter (pixels)")
plt.ylabel("Deflection angle (deg, rel)")
plt.title("Lensing Deflection vs Impact Parameter")
plt.legend()
plt.grid(True)
plt.show()

# --- 3. Effective Mass Comparison ---
# In GR, A = 4GM/c^2 in chosen units; with G=c=1, M_eff = A/4
M_eff_lens = A_fit / 4

# Dynamical mass from flat rotation: v^2 = GM/r => M_eff_dyn = v^2 * r / G
M_eff_dyn = (v_flat**2) * flatten_radius / G

M_eff_lens, M_eff_dyn