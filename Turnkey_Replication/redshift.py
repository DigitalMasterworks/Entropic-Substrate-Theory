import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# --- Logging Configuration ---
def setup_logging(filename):
    original_stdout = sys.stdout
    log_file = open(filename, 'w')
    sys.stdout = log_file
    print(f"--- Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Script: {filename.replace('.log', '.py')}")
    print("----------------------------------------------------------------")
    return original_stdout, log_file

def restore_logging(original_stdout, log_file):
    sys.stdout = original_stdout
    log_file.close()

# --- Main Simulation Logic (wrapped for logging) ---
def run_simulation():
    # =====================================================
    # TEST #4: Time dilation & gravitational redshift in S/C
    # =====================================================
    print("--- Running Test #4: Time Dilation & Gravitational Redshift ---")

    # Field: low-S well in otherwise near-vacuum
    N = 220
    cx, cy = N//2, N//2
    x = np.arange(N); y = np.arange(N)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r = np.hypot(xx - cx, yy - cy)

    # Smooth collapse well
    S = 0.98 - 0.78*np.exp(-(r/18.0)**2)   # S in [~0.2, ~0.98]
    S = np.clip(S, 0.2, 0.99)
    C = 1 - S

    # --- Time dilation via "clock particles" ---
    dt = 1.0
    T = 1200
    radii = np.linspace(3, 80, 16)
    clock_tau = np.zeros_like(radii)

    for i, rr in enumerate(radii):
        ix = np.clip(int(cx + rr), 0, N-1)
        iy = cy
        local_S = S[ix, iy]
        clock_tau[i] = np.sum(local_S * dt for _ in range(T))  # = T * S

    # Normalize vs far-away clock (r ~ 80)
    tau_far = clock_tau[-1]
    dilation = clock_tau / tau_far  # τ(r)/τ_far

    # Plot 1: Time Dilation
    plt.figure(figsize=(7,5))
    plt.plot(radii, dilation, marker="o")
    plt.xlabel("Radius from core (pixels)")
    plt.ylabel("Clock rate vs far clock (τ/τ_far)")
    plt.title("Time Dilation from Substrate S(r)")
    plt.grid(True)
    # MODIFICATION: Save figure instead of showing
    plt.savefig('redshift_time_dilation.png')
    plt.close()
    print("[Figure] Saved redshift_time_dilation.png")


    # --- Gravitational redshift experiment ---
    r_e = 10.0
    emit_ix = int(cx + r_e); emit_iy = cy
    S_e = S[emit_ix, emit_iy]

    r_o = 80.0
    obs_ix = int(cx + r_o); obs_iy = cy
    S_o = S[obs_ix, obs_iy]

    f0_local = 0.05
    phase_emit = 0.0
    speed_cap = 1.0
    pos = np.array([emit_ix, emit_iy], dtype=float)
    vel = np.array([1.0, 0.0]) * speed_cap

    obs_times = []
    obs_phase_samples = []
    Tsteps = 600

    for t in range(Tsteps):
        phase_emit += 2*np.pi * f0_local * (S_e * dt)
        pos += vel
        ix, iy = int(np.clip(pos[0], 0, N-1)), int(np.clip(pos[1], 0, N-1))

        # When crossing observer x, sample phase in observer local time
        if ix >= obs_ix and len(obs_times) < 200:
            if not obs_times:
                obs_times.append(0.0)
            else:
                obs_times.append(obs_times[-1] + S_o * dt)
            obs_phase_samples.append(phase_emit)

    # Measure observed frequency by linear fit to phase(t_obs)
    obs_times = np.array(obs_times)
    obs_phase_samples = np.unwrap(np.array(obs_phase_samples))
    if len(obs_times) > 5:
        A = np.vstack([obs_times, np.ones_like(obs_times)]).T
        slope, intercept = np.linalg.lstsq(A, obs_phase_samples, rcond=None)[0]
        f_obs = slope / (2*np.pi)
    else:
        f_obs = np.nan

    pred_ratio = S_e / S_o
    meas_ratio = f_obs / f0_local if np.isfinite(f_obs) else np.nan

    # Plot 2: Redshift Phase vs Time
    plt.figure(figsize=(7,5))
    plt.plot(obs_times, obs_phase_samples)
    plt.xlabel("Observer local time")
    plt.ylabel("Unwrapped phase at observer")
    plt.title("Gravitational Redshift: Phase vs Observer Time")
    plt.grid(True)
    # MODIFICATION: Save figure instead of showing
    plt.savefig('redshift_phase_fit.png')
    plt.close()
    print("[Figure] Saved redshift_phase_fit.png")

    print("\nTime dilation (near core vs far) examples:")
    for rr, dil in zip(radii[:4], dilation[:4]):
        print(f"  r≈{rr:.1f} -> τ/τ_far ≈ {dil:.3f}")
    print(f"\nRedshift ratio predicted S_e/S_o ≈ {pred_ratio:.4f}")
    print(f"Measured f_obs/f0 ≈ {meas_ratio:.4f}")


    # =======================================
    # TEST #20: High-S Dark Void (repulsion)
    # =======================================
    print("\n--- Running Test #20: High-S Dark Void (Repulsion) ---")

    N2 = 240
    cx2, cy2 = N2//2, N2//2
    x2 = np.arange(N2); y2 = np.arange(N2)
    xx2, yy2 = np.meshgrid(x2, y2, indexing="ij")
    r2 = np.hypot(xx2 - cx2, yy2 - cy2)

    S_bg = 0.75
    S_void = 0.98
    R_void = 45.0
    wall = np.exp(-((r2 - R_void)/6.0)**2)
    S2 = S_bg + (S_void - S_bg) * (r2 <= R_void) + 0.18*wall*(r2 > R_void)
    S2 = np.clip(S2, 0.0, 0.99)
    C2 = 1 - S2

    # Matter tracers launched toward the void
    num_tr = 800
    theta = 2*np.pi*np.random.rand(num_tr)
    r_start = np.random.uniform(70, 100, size=num_tr)
    px = cx2 + r_start*np.cos(theta)
    py = cy2 + r_start*np.sin(theta)

    # Store initial positions for reference in logging
    r_init_mean = np.mean(r_start)

    # Inward radial bias
    vx = -0.4*np.cos(theta)
    vy = -0.4*np.sin(theta)
    print(f"[Void Dynamics] Initialized {num_tr} tracers. Mean starting radius: {r_init_mean:.2f}")


    def gradC2(ix, iy):
        ix = np.clip(ix, 1, N2-2); iy = np.clip(iy, 1, N2-2)
        gx = (C2[ix+1, iy] - C2[ix-1, iy])*0.5
        gy = (C2[ix, iy+1] - C2[ix, iy-1])*0.5
        return gx, gy

    # Integrate particle motion
    dt2 = 1.0; steps2 = 420; G2 = 1.0; cap2 = 1.0; drag2 = 0.02
    for _ in range(steps2):
        ix = np.clip(px.astype(int), 1, N2-2)
        iy = np.clip(py.astype(int), 1, N2-2)
        gx = (C2[ix+1, iy] - C2[ix-1, iy])*0.5
        gy = (C2[ix, iy+1] - C2[ix, iy-1])*0.5
        vx += gx*G2; vy += gy*G2
        sp = np.sqrt(vx*vx + vy*vy) + 1e-12
        scale = np.minimum(1.0, cap2/sp)
        vx *= scale*(1 - drag2); vy *= scale*(1 - drag2)
        px += vx*dt2; py += vy*dt2
        px = np.clip(px, 1, N2-2); py = np.clip(py, 1, N2-2)

    r_final_mean = np.mean(np.hypot(px - cx2, py - cy2))
    print(f"[Void Dynamics] Final mean radius: {r_final_mean:.2f}. Repulsion observed: {r_final_mean > r_init_mean}")

    # Plot 3: Void Dynamics (Repulsion)
    plt.figure(figsize=(7,7))
    plt.imshow(S2.T, origin="lower", cmap='viridis')
    plt.scatter(px, py, s=2, color='white', alpha=0.8)
    plt.title("High-S Void Repels Matter to Boundary (Dark Void Dynamics)")
    plt.colorbar(label="S")
    plt.xlabel("x"); plt.ylabel("y")
    # MODIFICATION: Save figure instead of showing
    plt.savefig('redshift_void_dynamics.png')
    plt.close()
    print("[Figure] Saved redshift_void_dynamics.png")


    # Simple lensing test across void boundary
    num_rays = 21
    impact = np.linspace(-100, 100, num_rays)
    rays = []
    for b in impact:
        rays.append({"pos": np.array([0.0, cy2 + b], float), "vel": np.array([1.0, 0.0], float), "trail": []})

    for _ in range(500):
        for RY in rays:
            x, y = RY["pos"]
            ix, iy = int(np.clip(x, 1, N2-2)), int(np.clip(y, 1, N2-2))
            gx, gy = gradC2(ix, iy)
            acc = np.array([gx, gy])*1.0
            RY["vel"] += acc
            sp = np.linalg.norm(RY["vel"])
            if sp > 1.0:
                RY["vel"] = RY["vel"]/sp
            RY["pos"] += RY["vel"]
            RY["trail"].append(RY["pos"].copy())

    print(f"[Void Lensing] Propagated {num_rays} light rays.")

    # Plot 4 (Original was a second void plot): Photon Paths
    plt.figure(figsize=(8,8))
    plt.imshow(S2.T, origin="lower", cmap='viridis')
    for RY in rays:
        tr = np.array(RY["trail"])
        plt.plot(tr[:,0], tr[:,1], lw=0.8, color='cyan')
    plt.title("Photon Paths Skirting a High-S Void (Repulsive Lensing)")
    plt.colorbar(label="S")
    plt.xlabel("x"); plt.ylabel("y")
    # MODIFICATION: Save figure instead of showing (renamed for clarity)
    plt.savefig('redshift_void_lensing.png')
    plt.close()
    print("[Figure] Saved redshift_void_lensing.png")

    print("\n--- All tests finished successfully. ---")


if __name__ == "__main__":
    log_filename = "redshift.log"

    # Set up logging redirection
    original_stdout, log_file = setup_logging(log_filename)

    try:
        run_simulation()
    except Exception as e:
        print(f"\nFATAL ERROR: The simulation failed with an exception: {e}")
    finally:
        # Restore stdout regardless of success or failure
        restore_logging(original_stdout, log_file)
        # Print confirmation to the actual terminal
        print(f"Log output saved to {log_filename}")
        print("Figures saved to the current directory.")