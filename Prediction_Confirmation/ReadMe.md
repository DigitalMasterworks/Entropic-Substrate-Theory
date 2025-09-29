# Confirming Predictions: Void Catalog Analysis Pipeline

This folder contains the complete pipeline for analyzing galaxy void catalogs and confirming substrate-based predictions.  
All scripts are self-contained and work together to process observational void data, extract key statistics, and test model predictions.

---

## Required Scripts

- **fitsloader.py**  
  Loads NSA galaxy positions from the `nsa_v1_0_1.fits` catalog for position lookup.

- **PredictionsTester.py**  
  Main entry point for reproducing the void boundary-occupancy results and global statistics.  
  Discovers available catalogs in the current folder and processes each one, outputting CSVs and summary files.

- **VoidDeepTests.py**  
  Computes deep diagnostics for each catalog, including stacked radial profiles, robustness splits, anisotropy dipole, and null controls.  
  Produces CSV and TXT summary files per catalog.

- **VoidAnisoTest.py**  
  Focused anisotropy diagnostic, extracting harmonic amplitudes and power at k=3, both globally and per-void, with null controls.  
  Outputs per-void CSVs and summary TXT files.

---

## Required Data Files (per catalog)

For each void catalog to be analyzed, ensure the following files are present in the same directory:

- `<label>_zobovoids.dat`      (void centers, radii, redshifts; columns: RA, DEC, z, R)
- `<label>_zonevoids.dat`      (mapping of zones to voids; columns: zone_id, void_id)
- `<label>_galzones.dat`       (mapping of galaxies to zones; columns: gal_id, zone_id)
- `nsa_v1_0_1.fits`            (NASA-Sloan Atlas galaxy catalog; required for position lookup)

**Note:**  
- Scripts auto-detect available catalogs using the `*_zobovoids.dat` naming pattern.
- All necessary column formats are auto-detected by the scripts.

---

## Usage

1. Place all four scripts and required data files in the same working directory.
2. Install required Python packages: `numpy`, `astropy` (for FITS), etc.
3. Run the main pipeline:
   - For void boundary and global statistics:  
     ```bash
     python3 PredictionsTester.py
     ```
   - For detailed profile and anisotropy analysis:  
     ```bash
     python3 VoidDeepTests.py
     python3 VoidAnisoTest.py
     ```

## Required Data

The `nsa_v1_0_1.fits` catalog (~2.5GB) is required but **not included in this repository**.
Download it from:
https://www.sdss.org/dr17/data_access/value-added-catalogs/?vac_id=nasa-sloan-atlas
Place it in the same directory as the analysis scripts.

4. Output CSVs and summaries will be generated for each catalog and analysis.

---

## Outputs

- Boundary occupancy statistics (CSV)
- Anisotropy summaries (TXT, CSV)
- Stacked radial profiles and robustness splits (CSV)
- Global and per-void statistics for model confirmation (TXT)

---

## Citation

If you use or extend this pipeline, please cite the corresponding ReScience submission or the main project publication.
