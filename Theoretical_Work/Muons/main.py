#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Substrate g−2 Program — Full Pipeline

Implements two test routes:
  (A) Substrate-modified BMT precession (primary, fully implemented here)
  (B) Hadronic Vacuum Polarization window route (pluggable; requires inputs)

References (for methodology/parameters):
- E989 (Fermilab) measurement & ring parameters: magic momentum, ω_a extraction, ω_p reference.  # :contentReference[oaicite:4]{index=4}
- SM consensus baseline / uncertainty budgeting & "room" for new physics.                                   # :contentReference[oaicite:5]{index=5}
- Substrate theory couplings: dτ = S dt; single-metric optical form; weak-field S ≈ 1+Φ/c².                  # :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}

Usage examples:
  python -m substrate_g2.main precession --mode uniform --S0 0.9999999993 --kappa-mu 1.0 --kappa-p 0.0
  python -m substrate_g2.main precession --mode radial --S-center 0.999999999 --dS-dr -1e-12 --ring-radius 7.112 --kappa-mu 1.0 --kappa-p 0.0
  python -m substrate_g2.main precession --mode file --s-field-path ./S_field_ring.npy --kappa-mu 1.0 --kappa-p 0.5
  python -m substrate_g2.main hvp --rs-file ./Rs_data.csv --kernel built_in

All outputs are printed and saved under ./outputs/ with CSV and JSON summaries.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Callable, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

# ---------------------------
# Constants and experiment refs
# ---------------------------

ALPHA = 7.2973525693e-3             # fine-structure constant (CODATA 2018-level precision)
QE = 1.602176634e-19                # C, exact by SI
C = 299792458.0                     # m/s, exact by SI
HBAR = 1.054571817e-34              # J s
ME = 9.1093837015e-31               # kg
MMU_MEV = 105.6583745               # MeV/c^2 (CODATA-like)
GE = 2.00231930436256               # electron g-factor (useful if you want exact E989 extraction form)  # :contentReference[oaicite:8]{index=8}

# Convert muon mass to kg
EV = 1.602176634e-19                # J/eV
MMU = (MMU_MEV * 1e6 * EV) / (C**2)  # kg

# Muon magnetic anomaly (baseline reference for generating synthetic ω_a):
A_MU_BASELINE = 116592059e-11  # dimensionless ~ 1.16592059e-3 (close to legacy central value)
# (Use WP25 value 116592033e-11 as your SM yardstick when comparing outputs in reports.)  # :contentReference[oaicite:9]{index=9}
A_MU_SM_WP25 = 116592033e-11   # dimensionless  # :contentReference[oaicite:10]{index=10}
A_MU_SM_SIGMA = 62e-11         # 1σ theory uncertainty  # :contentReference[oaicite:11]{index=11}
NEW_PHYS_BAND_1SIG = 63e-11    # ~ current exp−SM room (magnitude)  # :contentReference[oaicite:12]{index=12}

# E989 ring / magic settings (for synthetic spectra)
B_RING = 1.45                   # Tesla (storage ring field)  # :contentReference[oaicite:13]{index=13}
P_MAGIC_GEV = 3.094             # GeV/c (magic momentum)     # :contentReference[oaicite:14]{index=14}
TAU_MUON_NS = 2197.03           # muon proper lifetime ~2.197 μs (ns here), for lab lifetime use gamma scale
ASYMMETRY = 0.35                # typical analyzing power A in E989 positron spectrum fits (ballpark)
N0_COUNTS = 1.0e6               # initial counts scale for synthetic spectrum
CBO_FREQ_KHZ = 400.0            # coherent betatron oscillation ~O(100–500 kHz); we keep it off by default
RING_RADIUS_M = 7.112           # E989 ring radius ~7.112 m  # :contentReference[oaicite:15]{index=15}

# Output directory
OUTDIR = os.path.abspath("./outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# Data Classes
# ---------------------------

@dataclass
class PrecessionConfig:
    mode: str = "uniform"  # "uniform" | "radial" | "file"
    S0: float = 1.0        # for uniform
    S_center: float = 1.0  # for radial: S(r) = S_center + (dS_dr) * r
    dS_dr: float = 0.0
    ring_radius_m: float = RING_RADIUS_M

    s_field_path: Optional[str] = None  # for "file": expects 1D numpy array of S at ring azimuth sampling

    kappa_mu: float = 1.0   # substrate tick exponent for muon spin coupling
    kappa_p: float = 0.0    # substrate tick exponent for proton NMR reference (0 means cancels in ratio)

    # synthetic spectrum controls
    t_max_us: float = 600.0
    dt_us: float = 0.1
    noise_sigma: float = 300.0
    include_cbo: bool = False
    cbo_amp: float = 0.01
    cbo_phase: float = 0.0

    # new
    rng_seed: int = 42           # deterministic noise control
    save_ideal: bool = False     # also save the noiseless spectrum we generate

@dataclass
class PrecessionResults:
    omega_a_std_rad_s: float
    omega_p_std_rad_s: float
    omega_a_meas_rad_s: float
    omega_p_meas_rad_s: float
    ratio_std: float
    ratio_meas: float
    delta_ratio: float
    delta_ln_ratio: float
    a_mu_delta_linear: float  # ≈ a_baseline * delta_ln_ratio
    a_mu_delta_exactK: Optional[float]  # if you supply exact K, this is K*delta_ratio
    within_current_room_1sigma: bool
    S_stats: Dict[str, float]
    fit_cov_status: str
    output_files: Dict[str, str]

# ---------------------------
# Helpers
# ---------------------------

def gamma_from_p(p_GeV: float = P_MAGIC_GEV, m_mu_MeV: float = MMU_MEV) -> float:
    """Relativistic gamma for the muon at momentum p (GeV/c)."""
    p = p_GeV    # GeV
    m = m_mu_MeV/1000.0  # GeV
    E = math.sqrt(p*p + m*m)  # GeV
    return E / m

def omega_a_standard(B: float = B_RING, a_mu: float = A_MU_BASELINE) -> float:
    """
    Standard ω_a ignoring E-field corrections at magic momentum:
      ω_a = (q/m) * a_mu * B
    Returns rad/s.
    """
    q_over_m = QE / MMU  # C/kg
    return q_over_m * a_mu * B  # s^-1 (rad/s)

def omega_p_standard(B: float = B_RING, gamma_p: float = 2.6752218744e8) -> float:
    """
    Proton Larmor ω_p = γ_p * B (γ_p is proton gyromagnetic ratio in rad·s^-1·T^-1).
    CODATA-like γ_p ≈ 2.6752218744e8 rad/(s·T).
    """
    return gamma_p * B

def muon_lab_lifetime_us() -> float:
    """Lab lifetime ~ γ * τ_muon."""
    gamma = gamma_from_p()
    return (gamma * TAU_MUON_NS) / 1000.0  # μs

def ring_azimuth_S(config: PrecessionConfig, n_points: int = 3600) -> NDArray[np.float64]:
    """
    Build S(θ) along the ring.
    - uniform: constant S0
    - radial: S(r)=S_center + dS_dr * r  (use ring_radius_m)
    - file: load 1D array of S(θ) (length n_points or will be interpolated)
    """
    theta = np.linspace(0.0, 2*np.pi, n_points, endpoint=False)
    if config.mode == "uniform":
        S = np.full_like(theta, fill_value=config.S0, dtype=np.float64)
    elif config.mode == "radial":
        r = np.full_like(theta, fill_value=config.ring_radius_m, dtype=np.float64)
        S = config.S_center + config.dS_dr * r
    elif config.mode == "file":
        if not config.s_field_path or not os.path.exists(config.s_field_path):
            raise FileNotFoundError("S-field file not found. Provide --s-field-path to a .npy 1D array.")
        arr = np.load(config.s_field_path)
        if arr.ndim != 1:
            raise ValueError("S-field .npy must be 1D with S along azimuth.")
        # resample to n_points if needed
        S = np.interp(np.linspace(0, len(arr), n_points, endpoint=False), np.arange(len(arr)), arr)
    else:
        raise ValueError("config.mode must be one of {'uniform','radial','file'}")
    return S

def apply_substrate_scaling(omega: float, S_local: NDArray[np.float64], kappa: float) -> float:
    """
    Scale a base frequency by the substrate tick factor averaged around the ring.
    Measurement occurs in lab time t; with dτ = S dt, a spin-specific coupling is modeled as ω_obs = ω_std * <S^κ>.
    """
    if kappa == 0.0:
        return omega
    factor = float(np.mean(np.power(S_local, kappa)))
    return omega * factor

# Synthetic E989-like positron time spectrum
def positron_spectrum_model(t, N0, A, omega, phi, tau, y0, cbo_amp=0.0, cbo_freq=2*np.pi*1e3, cbo_phi=0.0):
    base = N0 * np.exp(-t / tau) * (1.0 + A*np.cos(omega*t + phi)) + y0
    if cbo_amp != 0.0:
        base *= (1.0 + cbo_amp*np.cos(cbo_freq*t + cbo_phi))
    return base

def fit_omega_from_counts(time_array, counts, init_guess, include_cbo=False, cbo_freq=None):
    if include_cbo:
        def model(t, N0, A, omega, phi, tau, y0, cbo_amp, cbo_phi):
            return positron_spectrum_model(t, N0, A, omega, phi, tau, y0, cbo_amp=cbo_amp,
                                           cbo_freq=cbo_freq, cbo_phi=cbo_phi)
        p0 = (*init_guess, 0.01, 0.0)
        bounds = (0, [np.inf, 1.0, np.inf, 2*np.pi, np.inf, np.inf, 0.1, 2*np.pi])
    else:
        def model(t, N0, A, omega, phi, tau, y0):
            return positron_spectrum_model(t, N0, A, omega, phi, tau, y0)
        p0 = init_guess
        bounds = (0, [np.inf, 1.0, np.inf, 2*np.pi, np.inf, np.inf])

    popt, pcov = curve_fit(model, time_array, counts, p0=p0, bounds=bounds, maxfev=20000)
    omega_fit = popt[2]
    return omega_fit, popt, pcov

def compute_delta_a_from_ratios(r_std: float, r_meas: float, a_baseline: float = A_MU_SM_WP25,
                                K_exact: Optional[float] = None) -> Tuple[float, Optional[float]]:
    """
    Map ratio change to delta a_mu:
      linearized: δa ≈ a_baseline * ln(R_meas/R_std)
      exact-K:   δa = K * (R_meas - R_std)   (only if you supply K)
    """
    delta_ln = math.log(r_meas / r_std)
    delta_a_linear = a_baseline * delta_ln
    delta_a_exact = None if K_exact is None else K_exact * (r_meas - r_std)
    return delta_a_linear, delta_a_exact

def predict_delta_from_S(S: NDArray[np.float64],
                         kappa_mu: float,
                         kappa_p: float,
                         a_baseline: float = A_MU_SM_WP25) -> Dict[str, float]:
    """
    Analytic prediction for ΔlnR and δa_μ from an S(θ) array.
    """
    km = float(kappa_mu); kp = float(kappa_p)
    mean_km = float(np.mean(np.power(S, km)))
    mean_kp = float(np.mean(np.power(S, kp)))
    delta_lnR = math.log(mean_km) - math.log(mean_kp)
    delta_a = a_baseline * delta_lnR
    return {
        "delta_lnR": delta_lnR,
        "delta_a_mu": delta_a,
        "delta_a_mu_x1e11": delta_a * 1e11,
        "mean_Skmu": mean_km,
        "mean_Skp": mean_kp
    }

def fit_omega_in_windows(t: NDArray[np.float64],
                         y: NDArray[np.float64],
                         init_guess: Tuple[float, float, float, float, float, float],
                         include_cbo: bool,
                         cbo_freq: Optional[float],
                         windows: Tuple[Tuple[float, float], ...]):
    """
    Fit ω in fractional time windows; windows are tuples of (start_frac, end_frac) within [0,1].
    Returns list of dicts with 'win', 'omega_fit'.
    """
    out = []
    t0, t1 = float(t[0]), float(t[-1])
    T = t1 - t0
    for (a, b) in windows:
        a = max(0.0, min(1.0, a)); b = max(0.0, min(1.0, b))
        if b <= a: 
            continue
        ta, tb = t0 + a*T, t0 + b*T
        mask = (t >= ta) & (t <= tb)
        if mask.sum() < 100:
            out.append({"win": (a, b), "omega_fit": float("nan")})
            continue
        omega_fit, _, _ = fit_omega_from_counts(
            t[mask], y[mask], init_guess, include_cbo=include_cbo, cbo_freq=cbo_freq
        )
        out.append({"win": (a, b), "omega_fit": float(omega_fit)})
    return out
    
# ---------------------------
# Route A: Precession pipeline
# ---------------------------

def run_precession(config: PrecessionConfig) -> PrecessionResults:
    # Build S along ring (θ sampling)
    S_theta = ring_azimuth_S(config, n_points=3600)
    S_stats = {
        "S_min": float(np.min(S_theta)),
        "S_max": float(np.max(S_theta)),
        "S_mean": float(np.mean(S_theta)),
        "S_std": float(np.std(S_theta)),
    }

    # Standard ω's
    omega_a_std = omega_a_standard(B=B_RING, a_mu=A_MU_BASELINE)      # rad/s
    omega_p_std = omega_p_standard(B=B_RING)                           # rad/s

    # Apply substrate tick scaling with exponents κμ, κp
    omega_a_obs = apply_substrate_scaling(omega_a_std, S_theta, config.kappa_mu)
    omega_p_obs = apply_substrate_scaling(omega_p_std, S_theta, config.kappa_p)

    # Synthetic spectrum build & fit ω_a
    tmax = config.t_max_us * 1e-6
    dt = config.dt_us * 1e-6
    t = np.arange(0.0, tmax, dt, dtype=np.float64)

    # Lab lifetime
    tau_lab = muon_lab_lifetime_us() * 1e-6  # seconds

    # Make "standard" and "observed" spectra to verify the fitter can see δ
    # We'll fit only the observed (meas) to extract omega_a_meas
    rng = np.random.default_rng(config.rng_seed)
    counts_ideal = positron_spectrum_model(
        t, N0=N0_COUNTS, A=ASYMMETRY, omega=omega_a_obs, phi=0.25*np.pi, tau=tau_lab, y0=0.0,
        cbo_amp=(config.cbo_amp if config.include_cbo else 0.0),
        cbo_freq=(2*np.pi*CBO_FREQ_KHZ*1e3) if config.include_cbo else None,
        cbo_phi=config.cbo_phase if config.include_cbo else 0.0
    )
    # Poisson/stat noise
    noisy = counts_ideal + rng.normal(0.0, config.noise_sigma, size=counts_ideal.shape)

    init_guess = (max(noisy), 0.3, omega_a_std, 0.0, tau_lab, 0.0)  # start near standard
    if config.include_cbo:
        omega_fit, popt, pcov = fit_omega_from_counts(
            t, noisy, init_guess, include_cbo=True, cbo_freq=2*np.pi*CBO_FREQ_KHZ*1e3
        )
    else:
        omega_fit, popt, pcov = fit_omega_from_counts(t, noisy, init_guess, include_cbo=False)

    # Ratios
    ratio_std = omega_a_std / omega_p_std
    ratio_meas = omega_fit / omega_p_obs
    delta_ratio = ratio_meas - ratio_std
    delta_ln_ratio = math.log(ratio_meas / ratio_std)

    # Map to δa_μ (two ways)
    delta_a_linear, delta_a_exact = compute_delta_a_from_ratios(
        r_std=ratio_std, r_meas=ratio_meas, a_baseline=A_MU_SM_WP25, K_exact=None  # supply if desired
    )
    within_room = abs(delta_a_linear) <= NEW_PHYS_BAND_1SIG

    # Save summary & CSV
    ts = int(time.time())
    prefix = os.path.join(OUTDIR, f"precession_run_{ts}")
    csv_path = prefix + "_timeseries.csv"
    if config.save_ideal:
        ideal_path = prefix + "_ideal_timeseries.csv"
        with open(ideal_path, "w") as f:
            f.write("t_s,counts_ideal\n")
            for ti, yi in zip(t, counts_ideal):
                f.write(f"{ti:.9e},{yi:.6f}\n")
    
    # always save the measured/noisy spectrum so window refits can load it
    with open(csv_path, "w") as f:
        f.write("t_s,counts\n")
        for ti, yi in zip(t, noisy):
            f.write(f"{ti:.9e},{yi:.6f}\n")
            
    summary = {
        "config": asdict(config),
        "S_stats": S_stats,
        "omega_a_std_rad_s": omega_a_std,
        "omega_p_std_rad_s": omega_p_std,
        "omega_a_meas_fit_rad_s": float(omega_fit),
        "omega_a_obs_true_rad_s": omega_a_obs,
        "omega_p_obs_rad_s": omega_p_obs,
        "ratio_std": ratio_std,
        "ratio_meas": ratio_meas,
        "delta_ratio": delta_ratio,
        "delta_ln_ratio": delta_ln_ratio,
        "delta_a_linear": delta_a_linear,
        "delta_a_linear_x1e11": delta_a_linear * 1e11,  # convenient units
        "room_1sigma_x1e11": NEW_PHYS_BAND_1SIG * 1e11,
        "within_current_room_1sigma": within_room,
    }
    json_path = prefix + "_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # Fit covariance status
    fit_cov_status = "ok" if np.all(np.isfinite(pcov)) else "warn"

    out_files = {"timeseries_csv": csv_path, "summary_json": json_path}
    if config.save_ideal:
        out_files["timeseries_ideal_csv"] = ideal_path

    return PrecessionResults(
        omega_a_std_rad_s=omega_a_std,
        omega_p_std_rad_s=omega_p_std,
        omega_a_meas_rad_s=float(omega_fit),
        omega_p_meas_rad_s=omega_p_obs,
        ratio_std=ratio_std,
        ratio_meas=ratio_meas,
        delta_ratio=delta_ratio,
        delta_ln_ratio=delta_ln_ratio,
        a_mu_delta_linear=delta_a_linear,
        a_mu_delta_exactK=delta_a_exact,
        within_current_room_1sigma=within_room,
        S_stats=S_stats,
        fit_cov_status=fit_cov_status,
        output_files=out_files,
    )

# ---------------------------
# Route B: HVP (pluggable)
# ---------------------------

def built_in_kernel_Khat(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    A mild, placeholder kernel shape that emphasizes low-s (this is NOT the exact standard kernel).
    Replace with your exact K̂(s) if desired. This is kept differentiable & positive.
    """
    m_mu2 = (MMU * C**2 / EV)**2  # (eV)^2 for shape; we only need a smooth decaying weight
    x = s / (m_mu2 + 1.0)
    return 1.0 / (1.0 + x)  # soft low-energy emphasis

def integrate_hvp_rs(
    s_grid: NDArray[np.float64],
    R_of_s: NDArray[np.float64],
    kernel: Callable[[NDArray[np.float64]], NDArray[np.float64]] = built_in_kernel_Khat
) -> float:
    """
    Toy dispersive integral:
        a_mu^(HVP,LO) ~ (alpha^2 / (3π^2)) ∫ ds [ K̂(s) R(s) / s ]
    Replace kernel with exact one as needed and feed a realistic R(s).
    Units here are formal; treat result as "relative" unless you provide calibrated inputs.
    """
    s = np.asarray(s_grid, dtype=np.float64)
    R = np.asarray(R_of_s, dtype=np.float64)
    Khat = kernel(s)
    integrand = np.where(s > 0, Khat * R / s, 0.0)
    # trapezoid
    val = np.trapz(integrand, s)
    return (ALPHA**2 / (3.0 * math.pi**2)) * val

def run_hvp(rs_file: str, kernel_name: str = "built_in") -> Dict[str, float]:
    """
    Load R(s) CSV and integrate. CSV format required:
      s, R
      0.100, 0.0
      0.105, 12.34
      ...
    If you have a vacuum polarization Π(Q^2) instead, adapt this function or add a --pi-file route.
    """
    data = np.genfromtxt(rs_file, delimiter=",", names=True)
    s = data[data.dtype.names[0]]  # first column
    R = data[data.dtype.names[1]]  # second column

    kernel = built_in_kernel_Khat
    if kernel_name != "built_in":
        raise NotImplementedError("Provide your kernel function or extend the dispatcher.")

    val = integrate_hvp_rs(s, R, kernel=kernel)
    ts = int(time.time())
    out_json = os.path.join(OUTDIR, f"hvp_result_{ts}.json")
    with open(out_json, "w") as f:
        json.dump({"a_mu_HVP_like": val}, f, indent=2, sort_keys=True)
    return {"a_mu_HVP_like": val, "output_json": out_json}

def parse_windows_spec(spec: str):
    wins = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            label, rng = token.split(":", 1)
        else:
            label, rng = token, token
        a, b = rng.split("-", 1)
        a = float(a)
        b = float("inf") if b.strip().lower() == "inf" else float(b)
        wins.append((label, a, b))
    return wins

def run_hvp_windows(rs_file: str, windows_spec: str):
    data = np.genfromtxt(rs_file, delimiter=",", names=True)
    s = data[data.dtype.names[0]]  # s (GeV^2)
    R = data[data.dtype.names[1]]  # R(s)
    wins = parse_windows_spec(windows_spec)
    out = {}
    for label, a, b in wins:
        mask = (s >= a) & ((s <= b) if np.isfinite(b) else np.ones_like(s, dtype=bool))
        if mask.sum() < 2:
            out[label] = float("nan")
            continue
        out[label] = float(integrate_hvp_rs(s[mask], R[mask]))
    return out
    
# ---------------------------
# CLI
# ---------------------------

def build_argparser():
    p = argparse.ArgumentParser(prog="substrate_g2")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Precession
    pp = sub.add_parser("precession", help="Run substrate-modified BMT precession pipeline.")
    pp.add_argument("--mode", choices=["uniform", "radial", "file"], default="uniform")
    pp.add_argument("--S0", type=float, default=1.0)
    pp.add_argument("--S-center", dest="S_center", type=float, default=1.0)
    pp.add_argument("--dS-dr", dest="dS_dr", type=float, default=0.0)
    pp.add_argument("--ring-radius", dest="ring_radius_m", type=float, default=RING_RADIUS_M)
    pp.add_argument("--s-field-path", dest="s_field_path", type=str, default=None)

    pp.add_argument("--kappa-mu", dest="kappa_mu", type=float, default=1.0)
    pp.add_argument("--kappa-p", dest="kappa_p", type=float, default=0.0)

    pp.add_argument("--t-max-us", dest="t_max_us", type=float, default=600.0)
    pp.add_argument("--dt-us", dest="dt_us", type=float, default=0.1)
    pp.add_argument("--noise-sigma", type=float, default=300.0)

    pp.add_argument("--include-cbo", action="store_true")
    pp.add_argument("--cbo-amp", type=float, default=0.01)
    pp.add_argument("--cbo-phase", type=float, default=0.0)
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument("--save-ideal", action="store_true")

    # HVP
    ph = sub.add_parser("hvp", help="Integrate an R(s) CSV with a kernel (pluggable).")
    ph.add_argument("--rs-file", type=str, required=True, help="CSV with header 's,R'")
    ph.add_argument("--kernel", type=str, default="built_in", help="Kernel name; 'built_in' provided here.")
    # Predict-only (analytic)
    ppred = sub.add_parser("predict", help="Predict ΔlnR and δaμ from an S(θ) .npy without fitting.")
    ppred.add_argument("--s-field-path", required=True, type=str)
    ppred.add_argument("--kappa-mu", dest="kappa_mu", type=float, default=1.0)
    ppred.add_argument("--kappa-p", dest="kappa_p", type=float, default=0.0)

    # Precession window fits (stability)
    pwin = sub.add_parser("precession-windows", help="Run precession and refit ω in fractional time windows.")
    pwin.add_argument("--mode", choices=["uniform", "radial", "file"], default="uniform")
    pwin.add_argument("--S0", type=float, default=1.0)
    pwin.add_argument("--S-center", dest="S_center", type=float, default=1.0)
    pwin.add_argument("--dS-dr", dest="dS_dr", type=float, default=0.0)
    pwin.add_argument("--ring-radius", dest="ring_radius_m", type=float, default=RING_RADIUS_M)
    pwin.add_argument("--s-field-path", dest="s_field_path", type=str, default=None)
    pwin.add_argument("--kappa-mu", dest="kappa_mu", type=float, default=1.0)
    pwin.add_argument("--kappa-p", dest="kappa_p", type=float, default=0.0)
    pwin.add_argument("--t-max-us", dest="t_max_us", type=float, default=600.0)
    pwin.add_argument("--dt-us", dest="dt_us", type=float, default=0.1)
    pwin.add_argument("--noise-sigma", type=float, default=300.0)
    pwin.add_argument("--include-cbo", action="store_true")
    pwin.add_argument("--cbo-amp", type=float, default=0.01)
    pwin.add_argument("--cbo-phase", type=float, default=0.0)
    pwin.add_argument("--seed", type=int, default=42)
    pwin.add_argument("--windows", type=str, default="0.0-0.2,0.2-0.6,0.6-1.0",
                      help="Comma list of start-end fractions within [0,1], e.g. '0.0-0.2,0.2-0.6,0.6-1.0'")

    # HVP window integrals (toy)
    phw = sub.add_parser("hvp-windows", help="Integrate R(s) in windows (e.g., SD/W/LD) with the toy kernel.")
    phw.add_argument("--rs-file", type=str, required=True)
    phw.add_argument("--windows", type=str, default="SD:0-1.0,W:1.0-9.0,LD:9.0-inf",
                     help="label:start-end in GeV^2, comma-separated; use 'inf' for open end.")
    return p

def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "precession":
        cfg = PrecessionConfig(
            mode=args.mode,
            S0=args.S0,
            S_center=args.S_center,
            dS_dr=args.dS_dr,
            ring_radius_m=args.ring_radius_m,
            s_field_path=args.s_field_path,
            kappa_mu=args.kappa_mu,
            kappa_p=args.kappa_p,
            t_max_us=args.t_max_us,
            dt_us=args.dt_us,
            noise_sigma=args.noise_sigma,
            include_cbo=args.include_cbo,
            cbo_amp=args.cbo_amp,
            cbo_phase=args.cbo_phase,
            rng_seed=getattr(args, "seed", 42),
            save_ideal=getattr(args, "save_ideal", False),
        )
        res = run_precession(cfg)

        print("\n=== Substrate g−2 Precession Results ===")
        print(f"S : mean={res.S_stats['S_mean']:.12f}  (min={res.S_stats['S_min']:.12f}, max={res.S_stats['S_max']:.12f})")
        print(f"ω_a std (rad/s):   {res.omega_a_std_rad_s:.9e}")
        print(f"ω_p std (rad/s):   {res.omega_p_std_rad_s:.9e}")
        print(f"ω_a meas (fit):    {res.omega_a_meas_rad_s:.9e}")
        print(f"ω_p meas (scaled): {res.omega_p_meas_rad_s:.9e}")
        print(f"R_std = ωa/ωp:     {res.ratio_std:.12e}")
        print(f"R_meas:            {res.ratio_meas:.12e}")
        print(f"ΔR:                {res.delta_ratio:.12e}")
        print(f"Δ ln R:            {res.delta_ln_ratio:.12e}")
        print(f"δa_μ (linear)      {res.a_mu_delta_linear:.12e}  (×1e11 = {res.a_mu_delta_linear*1e11:.3f})")
        print(f"Room (1σ, ×1e11):  {NEW_PHYS_BAND_1SIG*1e11:.3f}  => within? {res.within_current_room_1sigma}")
        print(f"Saved: {res.output_files}")
        
    elif args.cmd == "predict":
        S = np.load(args.s_field_path)
        pred = predict_delta_from_S(S, args.kappa_mu, args.kappa_p, a_baseline=A_MU_SM_WP25)
        print("\n=== Predict-only ===")
        print(f"Δ ln R: {pred['delta_lnR']:.12e}")
        print(f"δa_μ  : {pred['delta_a_mu']:.12e}  (×1e11={pred['delta_a_mu_x1e11']:.3f})")
        print(f"<S^κμ>={pred['mean_Skmu']:.12e}  <S^κp>={pred['mean_Skp']:.12e}")

    elif args.cmd == "precession-windows":
        cfg = PrecessionConfig(
            mode=args.mode,
            S0=args.S0,
            S_center=args.S_center,
            dS_dr=args.dS_dr,
            ring_radius_m=args.ring_radius_m,
            s_field_path=args.s_field_path,
            kappa_mu=args.kappa_mu,
            kappa_p=args.kappa_p,
            t_max_us=args.t_max_us,
            dt_us=args.dt_us,
            noise_sigma=args.noise_sigma,
            include_cbo=args.include_cbo,
            cbo_amp=args.cbo_amp,
            cbo_phase=args.cbo_phase,
            rng_seed=getattr(args, "seed", 42),
            save_ideal=True
        )
        res = run_precession(cfg)
        csv_path = res.output_files.get("timeseries_csv")
        use_ideal = False
        if not (csv_path and os.path.exists(csv_path)):
            csv_path = res.output_files.get("timeseries_ideal_csv")
            use_ideal = True
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        t = data["t_s"]; y = data["counts_ideal"] if use_ideal else data["counts"]
        init_guess = (max(y), 0.3, res.omega_a_std_rad_s, 0.0, muon_lab_lifetime_us()*1e-6, 0.0)
        wins = []
        for token in args.windows.split(","):
            a, b = token.split("-"); wins.append((float(a), float(b)))
        subfits = fit_omega_in_windows(
            t, y, init_guess, include_cbo=args.include_cbo,
            cbo_freq=(2*np.pi*CBO_FREQ_KHZ*1e3) if args.include_cbo else None,
            windows=tuple(wins)
        )
        print("\n=== Window fits ===")
        for d in subfits:
            a,b = d["win"]; print(f"{a:.2f}-{b:.2f}: ω_a = {d['omega_fit']:.9e} rad/s")

    elif args.cmd == "hvp-windows":
        out = run_hvp_windows(args.rs_file, args.windows)
        print("\n=== HVP window integrals (toy) ===")
        for k,v in out.items():
            print(f"{k:>6}: {v:.6e}")
            
    elif args.cmd == "hvp":
        out = run_hvp(args.rs_file, kernel_name=args.kernel)
        print("\n=== HVP Integration (toy) ===")
        print(out)

if __name__ == "__main__":
    main()