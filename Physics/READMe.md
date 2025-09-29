# Physics Substrate: Lean Formalization

This folder contains a minimal, self-contained Lean formalization of substrate-based physics models.  
All files are written in Lean 4 and use the `Physics` namespace throughout.

## Contents

- **Physics.lean**  
  Core definition of substrate fields with two real-valued components `S` and `C`, constrained by `S + C = 1`.

- **Field3D.lean**  
  Definition of a scalar field `S(x, t)` over ℝ³ × ℝ and associated operations.

- **CSEntropy.lean**  
  Minimal entropy substrate structure, algebraic rearrangements, and mixing identities.

- **WeakField.lean**  
  Weak-field construction from a Newtonian potential, field mappings, and gravitational parameter definitions.

- **Kinematics.lean**  
  Acceleration law, gradient structure, and proper time scaling in terms of substrate fields.

- **Metric.lean**  
  Substrate metric: defines a generalized line element and refractive index in terms of field `S`.

- **Voxel.lean**  
  Discrete voxel model with microstate counting, entropy definition, and collapse potential.

- **QuasiStatic.lean**  
  Quasi-static (Poisson) model of the substrate: Laplacian operator, density, and field constraints at fixed time.

## Requirements

- **Lean 4**
- **mathlib4**

## Usage

1. Ensure all files are located in a `Physics/` directory.
2. Build with `lake build` or your preferred Lean 4 project manager.
3. All files are independent modules under the `Physics` namespace.
4. Each file can be imported individually as needed for research or extension.

## Notes

- All dependencies between these files are explicit and local.
- No external (non-mathlib) dependencies are required.
- These files do **not** include reductions to number theory or Clay Millennium Problems.

## Citation

If you use or extend this code, please cite the corresponding ReScience submission or the project’s main publication.
