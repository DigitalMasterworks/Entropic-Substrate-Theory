# Substrate g−2: One-shot Digest
_Generated: 2025-10-03T07:20:49.390725Z_

## Required vs. Realistic Contributions

- **Target** δa_μ: 250.0 × 1e-11  (2.500e-09)
- **Required** ⟨ln S⟩: 2.144e-06  (Φ_required = 1.927e+11 J/kg)
- **Earth** ⟨ln S⟩: -6.961e-10  → implied δa_μ (×1e-11) = -0.081
- **From `S_real.npy`** ⟨ln S⟩: -6.961e-10 (should match Earth value)

## QED Vacuum Estimate (Low-field EH)

- (B/B_cr)^2 = 1.079e-19,  7(E/E_cr)^2 = 4.017e-24,  α/(45π) = 5.162e-05
- **⟨ln S⟩_QED ≈ −(n−1)** = -5.570e-24

## Measured (latest runs)

| file | mode | κμ | κp | S_mean | ΔlnR | δaμ (×1e-11) | within room? |
|---|---|---:|---:|---:|---:|---:|:---:|
| precession_run_1759473503_summary.json | uniform | 1.0 | 0.0 | 0.999999999305 | -7.217e-10 | -0.084 | ✅" |
| precession_run_1759473505_summary.json | uniform | 1.0 | 1.0 | 0.999999999305 | -2.670e-11 | -0.003 | ✅" |
| precession_run_1759473507_summary.json | radial | 1.0 | 0.0 | 0.999999999593 | -4.338e-10 | -0.051 | ✅" |
| precession_run_1759473508_summary.json | uniform | 1.0 | 0.0 | 0.999999999305 | -7.217e-10 | -0.084 | ✅" |
| precession_run_1759475072_summary.json | file | 1.0 | 0.0 | 0.999999999304 | -7.228e-10 | -0.084 | ✅" |
| precession_run_1759475074_summary.json | file | 1.0 | 1.0 | 0.999999999304 | -2.670e-11 | -0.003 | ✅" |
