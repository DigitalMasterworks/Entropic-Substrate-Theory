# Quantitative analysis of flattening radius and bending angle curve

# Determine velocity flattening radius:
# Flattening is when velocity change < small threshold across increasing radius
velocities = np.array(orbital_speeds)
radii_arr = np.array(radii)
dv = np.abs(np.diff(velocities))
# Find first radius where dv stays below 5% of max velocity for all subsequent points
threshold = 0.05 * np.max(velocities)
flatten_index = None
for i in range(len(dv)):
    if np.all(dv[i:] < threshold):
        flatten_index = i
        break

flatten_radius = radii_arr[flatten_index] if flatten_index is not None else None

# Bending angle curve: normalize relative to farthest impact parameter
bending_angles = np.array(bending_angles)
bending_norm = bending_angles - bending_angles[-1]  # relative deflection

# Summary
flatten_radius, velocities[flatten_index], list(zip(impact_params, bending_norm))[:5]