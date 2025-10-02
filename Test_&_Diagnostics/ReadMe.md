# README: Substrate Physics Diagnostic Pipeline

This repository contains independent simulation and analysis scripts for dark matter halo, void, and anisotropy diagnostics. Follow these instructions for reproducible results and to confirm all model dependencies.

---

## 1. Autotuner.py

Purpose:
- Searches for optimal parameters (constants) for all core diagnostics (halo, void, anisotropy).
- Use this if you wish to re-tune the models or verify parameter selection.

Usage:
1. Run `Autotuner.py`
2. Copy the printed constants for each diagnostic section.
3. Paste these constants into `Simulator3.py` (or wherever model constants are defined).

---

## 2. Simulator3.py

Purpose:
- Runs the three main diagnostics (halo, void, anisotropy) using the current set of constants.
- Produces the expected reference outputs (e.g., rotation curves, mass ratios, boundary fractions, anisotropy metrics).

Usage:
- Ensure all constants in `Simulator3.py` match those provided by `Autotuner.py`.
- Run `Simulator3.py` and record the outputs.

---

## 3. Verification.py

Purpose:
- Checks that diagnostic outputs match the target reference values.
- Provides a PASS/FAIL summary for each diagnostic, based on the same criteria as `Simulator3.py`.

Usage:
- After running `Simulator3.py`, run `Verification.py`.
- Review the PASS/FAIL summary for each diagnostic.

---

## 4. Simulation and Analysis Scripts

- `darkmatterhalo.py` — Main simulation for halo structure, rotation curves, and lensing.
- `galacticlensing.py` — Lensing simulation using a final halo field.
- `darkmatterhalosweep.py` — Parameter sweep to map mass ratios and related metrics.
- `darkmatterproofgr.py` — Analysis and fitting of output arrays for dynamical and lensing mass. (If required, run `darkmatterhalo.py` first to generate data.)
- `csch-and-swapping.py` — Bell/CHSH simulation; standalone.
- `delayed-choicd-mzi.py` — Delayed-choice Mach–Zehnder simulation; standalone.


Usage:
- These scripts can be run independently after confirming parameters as above.
- If a script depends on output from another (e.g., `darkmatterproofgr.py` requires arrays from `darkmatterhalo.py`), run the required script first.

---

## Recommended Order of Execution

- `CSreduction.py` — Derives the invariant manifold S + C = 1, substitutes C = 1 − S, constructs the static isotropic metric ds² = S² c² dt² − S⁻² (dx² + dy² + dz²), auto-generates Christoffel symbols and the null-geodesic ODEs, integrates rays through S(x,y), and fits weak-field bending α(b) ≈ A/b; verifies GR with A ≈ 4GM/c² for S = exp(Φ/c²), and prints PPN expansion (β = 1, γ = 1 for the exponential map).
- `timestepreduction.py` — Derives the emergent evolution law ∂t S = κ ∇·(S ∇S) and local clock Δt_eff ∝ S; shows Σ = S + C → 1 and evolution thereafter on Σ = 1; demonstrates boundedness 0 ≤ S ≤ 1, energy-like decay, and practical no-surgery stability; emits compact logs/plots for stability and convergence checks.
1. `Autotuner.py`  (optional, only if re-tuning)
2. `Simulator3.py`  (to confirm outputs with current constants)
3. `Verification.py`  (to check PASS/FAIL)
4. Any of the simulation or analysis scripts as needed

---

## Notes

- Always ensure that all scripts use the same set of model constants for reproducibility.
- For parameter or model changes, repeat steps 1–3 before further analysis.

# Additional Independent Simulation and Analysis Scripts

The following scripts are fully independent and can be run in any order. None of these scripts depend on output or input from each other or from the main diagnostic pipeline.

---

- **CHSH.py**
  - Runs a lattice simulation of the CHSH (Bell) inequality with quantum correlations and outputs S-values and correlation statistics for standard settings.

- **Gravity.py**
  - Computes metric expansions and parametrized post-Newtonian (PPN) parameters for different mappings between the gravitational potential and the metric, using symbolic algebra. Prints analytic expansions and inferred values for β and γ.

- **PPN.py**
  - Symbolically expands the metric for exponential mappings of the gravitational potential. Outputs PPN metric coefficients and confirms β = 1, γ = 1 as in general relativity.

- **Quibits.py**
  - Analyzes simulated field histories for signatures of qubit, neutrino, quark, and Higgs candidates. Detects oscillatory modes and logs candidates to output files.

---

**Usage:**  
Each script is fully self-contained and can be executed at any time, in any order.  
Review printed or logged output for results and interpretations.  
No data or configuration sharing is required between these scripts.

