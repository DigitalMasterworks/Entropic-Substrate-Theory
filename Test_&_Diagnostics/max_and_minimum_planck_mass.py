planck_mass_g = 2.176e-5      # Planck mass (grams)
planck_length_cm = 1.616e-33  # Planck length (cm)
planck_voxel_volume_cm3 = planck_length_cm ** 3

# Physical: 1 cm³ water = 1g
voxels_per_cm3 = 1 / planck_voxel_volume_cm3
cube_mass_g = 1.0  # 1g (water)
mass_per_voxel_water = cube_mass_g / voxels_per_cm3

# At S = 1: pure entropy, min possible
mass_per_voxel_S1 = 0.0

# At S = 0: full collapse, max possible = Planck mass per Planck voxel (physics)
mass_per_voxel_S0 = planck_mass_g

print("\n--- Planck Voxel Mass (All 3 Cases) ---")
print(f"Planck voxel volume: {planck_voxel_volume_cm3:.3e} cm³")
print(f"Planck mass per Planck voxel (S = 0, max):      {mass_per_voxel_S0:.5e} grams")
print(f"Mass per Planck voxel (S = 1, min):             {mass_per_voxel_S1:.5e} grams")
print(f"Mass per Planck voxel (in 1g/cm³ water):        {mass_per_voxel_water:.5e} grams")