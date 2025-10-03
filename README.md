# Entropic Substrate Theory

This repository brings together all components needed to rigorously define, simulate, and empirically test the hypothesis that cosmic structure and gravity emerge from an entropic substrate field. 

The project integrates:
- **Formalized Lean 4 proofs and definitions** of the substrate theory and its predictions,
- **Python pipelines for large-scale data analysis** and comparison with observational void catalogs,
- **Mathematical diagnostics and validation scripts** to ensure correctness and reproducibility.

Our goal is to provide a fully open, transparent, and reproducible platform for substrate-based cosmology research, from first principles to data-driven results.


This repository brings together three main components:

---

## 1. `Physics/` — Lean Substrate Physics Formalization

Contains all Lean 4 code for formalizing the substrate field theory at a mathematical and proof level.
- **Purpose:** Rigorous, computer-verified definitions, theorems, and models for substrate fields, entropy/collapse relations, kinematics, and weak-field gravity.
- All physics is expressed in Lean under the `Physics` namespace, with full formal proofs where possible.

---

## 2. `Prediction_Confirmation/` — Empirical Prediction Validation

Contains Python scripts and supporting data to **test the predictions of the substrate model against real observational catalogs**.
- **Purpose:** Automates analysis of galaxy void data to extract statistics (boundary occupancy, anisotropy, etc.) and compare them with theoretical predictions.
- Includes all scripts needed to replicate the main empirical results in the associated work.

---

## 3. `Test_&_Diagnostics/` — Mathematical & Computational Verification

Contains diagnostic and test scripts for **verifying, cross-checking, and validating all theoretical models and mathematical results** in the project.
- **Purpose:** These scripts run sanity checks, perform independent simulations, sweep parameters, or test mathematical predictions in isolation.
- Used to confirm that the Lean models, Python analysis, and all analytic calculations are correct and internally consistent.

---

## 4. `Turnkey_Replication/` — Automatically Recreate Results with RunAll.sh

Contains all referenced python scripts in the scalar field paper, required bat files and an sh script that automatically downloads the nsa_v1_0_1.fits file and runs all tests, automatically saving logs, test artifacts and images.

---

## Additional Materials

Reference notes, exploratory notebooks, and working results are provided for transparency. These are not part of the core, validated pipeline.

---

**For details on running or using the code in any folder, see the corresponding README in that folder.**
