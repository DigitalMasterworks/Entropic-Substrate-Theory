#!/bin/bash
# Full Replication Script for the Entropic Scalar Field Model (17 Modules)
# Runs all files in the precise order, fixing the dependency issue for the lensing modules.

echo "--- STARTING FULL REPLICATION PIPELINE (17 MODULES) ---"
echo "----------------------------------------------------------------"

# --- CONFIGURATION ---
NSA_FILE="nsa.v0_1_2.fits"
WGET_URL_PRIMARY="https://data.sdss.org/sas/dr17/sdss/atlas/v1/nsa_v1_0_1.fits"

# 1. PHASE 0: SETUP AND DATA ACQUISITION
echo "Phase 0: Environment Setup and Data Download"

# 1a. Dependencies
echo "Installing Python dependencies (numpy, sympy, astropy, matplotlib, scipy)..."
pip install numpy sympy astropy matplotlib scipy || { echo "ERROR: Dependency installation failed. Aborting."; exit 1; }

# 1b. Data Download (2.1 GB file check)
if [ -f "$NSA_FILE" ]; then
    echo "NSA data file already exists: $NSA_FILE. Skipping download."
else
    echo "NSA data file not found. Attempting download from primary URL..."
    # -O saves it as the specified file name
    wget -c -O "$NSA_FILE" "$WGET_URL_PRIMARY"

    if [ $? -ne 0 ]; then
        echo "ERROR: Data download failed. Please manually download $NSA_FILE or try an alternative URL."
        exit 1
    fi
    echo "Data download: PASS."
fi
echo "----------------------------------------------------------------"

# 2. PHASE 1: THEORETICAL & PRELIMINARY ANALYSIS (Modules 1-4)
echo "Phase 1: Running Theoretical and Preliminary Analysis..."

# M1. Cosmological Redshift Analysis
echo "   -> Running redshift.py"
python3 redshift.py
if [ $? -ne 0 ]; then echo "ERROR: redshift.py failed."; exit 1; fi

# M2. PPN Limit (Exponential)
echo "   -> Running PPN.py"
python3 PPN.py
if [ $? -ne 0 ]; then echo "ERROR: PPN.py failed."; exit 1; fi

# M3. PPN Limit (Linear)
echo "   -> Running Gravity.py"
python3 Gravity.py
if [ $? -ne 0 ]; then echo "ERROR: Gravity.py failed."; exit 1; fi

# M4. Geodesic/Ray Tracing Derivation
echo "   -> Running CSreduction.py"
python3 CSreduction.py
if [ $? -ne 0 ]; then echo "ERROR: CSreduction.py failed."; exit 1; fi

echo "Phase 1: THEORETICAL CHECKS PASS."
echo "----------------------------------------------------------------"


# 3. PHASE 2: SIMULATION CORE & DIAGNOSTICS (Modules 5-10)
echo "Phase 2: Running Simulation, Calibration, and Field-Dependent Diagnostics (Order Fixed)..."

# M5. Simulation Core: Autotuner (Must run first for constants)
echo "   -> Running Autotuner.py (Calibration/Tuning)"
python3 Autotuner.py
if [ $? -ne 0 ]; then echo "ERROR: Autotuner.py failed."; exit 1; fi

# M6. Simulation Core: Main Engine (Generates C/S fields)
echo "   -> Running Simulator3.py (Core field evolution)"
python3 Simulator3.py
if [ $? -ne 0 ]; then echo "ERROR: Simulator3.py failed."; exit 1; fi

# M7. **DIAGNOSTIC (FIXED ORDER):** Simplified Lensing Test (Depends on C field from Simulator3.py)
echo "   -> Running darkmatterlensing.py"
python3 darkmatterlensing.py
if [ $? -ne 0 ]; then echo "ERROR: darkmatterlensing.py failed."; exit 1; fi

# M8. **DIAGNOSTIC (FIXED ORDER):** Full Halo Lensing (Depends on C field from Simulator3.py)
echo "   -> Running galacticlensing.py"
python3 galacticlensing.py
if [ $? -ne 0 ]; then echo "ERROR: galacticlensing.py failed."; exit 1; fi

# M9. Verification (Initial check against targets)
echo "   -> Running Verification.py (Initial targets check)"
python3 Verification.py
if [ $? -ne 0 ]; then echo "ERROR: Verification.py failed. Initial targets not met."; exit 1; fi

# M10. Evolution PDE Check
echo "   -> Running timestepreduction.py"
python3 timestepreduction.py
if [ $? -ne 0 ]; then echo "ERROR: timestepreduction.py failed."; exit 1; fi

echo "Phase 2: SIMULATION AND CORE DIAGNOSTICS PASS."
echo "----------------------------------------------------------------"


# 4. PHASE 3: ADVANCED DIAGNOSTICS & FINAL CHECKS (Modules 11-17)
echo "Phase 3: Running Advanced Halo/Void/Quantum Tests..."

# M11. Diagnostic: Dark Matter Halo Model
echo "   -> Running darkmatterhalo.py"
python3 darkmatterhalo.py
if [ $? -ne 0 ]; then echo "ERROR: darkmatterhalo.py failed."; exit 1; fi

# M12. Diagnostic: Halo Parameter Sweep
echo "   -> Running darkmatterhalosweep.py"
python3 darkmatterhalosweep.py
if [ $? -ne 0 ]; then echo "ERROR: darkmatterhalosweep.py failed."; exit 1; fi

# M13. Diagnostic: Predictions Tester
echo "   -> Running PredictionsTester.py"
python3 PredictionsTester.py
if [ $? -ne 0 ]; then echo "ERROR: PredictionsTester.py failed."; exit 1; fi

# M14. Diagnostic: Void Deep Tests
echo "   -> Running VoidDeepTests.py"
python3 VoidDeepTests.py
if [ $? -ne 0 ]; then echo "ERROR: VoidDeepTests.py failed."; exit 1; fi

# M15. Diagnostic: Void Anisotropy Test
echo "   -> Running VoidAnisoTest.py"
python3 VoidAnisoTest.py
if [ $? -ne 0 ]; then echo "ERROR: VoidAnisoTest.py failed."; exit 1; fi

# M16. Final Analogy: CHSH Test
echo "   -> Running CHSH.py"
python3 CHSH.py
if [ $? -ne 0 ]; then echo "ERROR: CHSH.py failed."; exit 1; fi

# M17. Final Analogy: MZI Test
echo "   -> Running delayed-choice-mzi.py"
python3 delayed-choice-mzi.py
if [ $? -ne 0 ]; then echo "ERROR: delayed-choice-mzi.py failed."; exit 1; fi

echo "----------------------------------------------------------------"
echo "✅ Full Replication Pipeline Completed Successfully (17/17 Modules Verified)."
