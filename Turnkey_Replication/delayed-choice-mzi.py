import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# --- Logging Configuration ---
# Function to redirect stdout to a log file
def setup_logging(filename):
    original_stdout = sys.stdout
    log_file = open(filename, 'w')
    sys.stdout = log_file
    print(f"--- Log started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    print(f"Script: {filename.replace('.log', '.py')}")
    print("----------------------------------------------------------------")
    return original_stdout, log_file

# Function to restore original stdout
def restore_logging(original_stdout, log_file):
    sys.stdout = original_stdout
    log_file.close()
# -----------------------------

# Random number generator initialized once
rng = np.random.default_rng(0)

def bs_matrix():
    c = 1/np.sqrt(2)
    return np.array([[c, 1j*c],
                     [1j*c, c]], dtype=np.complex128)

BS = bs_matrix()

def propagate(phi):
    return np.array([[1.0, 0.0],
                     [0.0, np.exp(1j*phi)]], dtype=np.complex128)

def run_trials(n_trials=5000, phi=0.0, mode="BS2_on"):
    in_state = np.array([1.0+0j, 0.0+0j])
    clicks_D1 = 0
    clicks_D2 = 0
    for _ in range(n_trials):
        after_bs1 = BS @ in_state
        after_phase = propagate(phi) @ after_bs1

        if mode == "BS2_on":
            out = BS @ after_phase
            p1 = np.abs(out[0])**2
            if rng.random() < p1:
                clicks_D1 += 1
            else:
                clicks_D2 += 1

        elif mode == "BS2_off":
            pA = np.abs(after_phase[0])**2
            if rng.random() < pA:
                clicks_D1 += 1
            else:
                clicks_D2 += 1

        elif mode == "delayed_choice":
            # NOTE: The original script effectively ran separate BS2_on/BS2_off for the sweep plot
            # This is a general Delayed-Choice MZI trial which selects a measurement type randomly
            if rng.random() < 0.5: # 50/50 chance to recombine (interference)
                out = BS @ after_phase
                p1 = np.abs(out[0])**2
                if rng.random() < p1: clicks_D1 += 1
                else: clicks_D2 += 1
            else: # 50/50 chance to check path (which-path)
                pA = np.abs(after_phase[0])**2
                if rng.random() < pA: clicks_D1 += 1
                else: clicks_D2 += 1
        else:
            raise ValueError("Unknown mode")
    return clicks_D1, clicks_D2

# --- Main Simulation Logic (wrapped for logging) ---
def run_simulation():
    phis = np.linspace(0, 2*np.pi, 25)
    n = 4000

    print(f"[Setup] Running sweep with N={n} trials per point, {len(phis)} phase points.")

    D1_on = []; D1_off = []; D1_delayed_recombined = []; D1_delayed_whichpath = []
    for phi in phis:
        # BS2 ON (Interference)
        c1, _ = run_trials(n_trials=n, phi=phi, mode="BS2_on")
        D1_on.append(c1/n)

        # BS2 OFF (Which-Path)
        c1o, _ = run_trials(n_trials=n, phi=phi, mode="BS2_off")
        D1_off.append(c1o/n)

        # Delayed-Choice visualization setup
        c1r, _ = run_trials(n_trials=n, phi=phi, mode="BS2_on")
        c1w, _ = run_trials(n_trials=n, phi=phi, mode="BS2_off")
        D1_delayed_recombined.append(c1r/n)
        D1_delayed_whichpath.append(c1w/n)

    # --- Plot results (Autosave + Close) ---
    plt.figure(figsize=(8,5))
    plt.plot(phis, D1_on, marker='o', label="BS2 ON (interference)")
    plt.plot(phis, D1_off, marker='s', label="BS2 OFF (which-path)")
    plt.plot(phis, D1_delayed_recombined, '--', label="Delayed → recombine")
    plt.plot(phis, D1_delayed_whichpath, '--', label="Delayed → which-path")
    plt.xlabel("Phase φ"); plt.ylabel("P(D1 click)")
    plt.title("Delayed-Choice Mach–Zehnder Simulation")
    plt.legend(); plt.grid(True)

    # MODIFICATION: Save figure instead of showing
    plt.savefig('delayed-choice-mzi_plot.png')
    plt.close()
    print("[Figure] Saved delayed-choice-mzi_plot.png")

    # Final run for specific phase angle
    phi0 = np.pi/2
    n_final = 20000
    c1_on, c2_on = run_trials(n_final, phi0, "BS2_on")
    c1_off, c2_off = run_trials(n_final, phi0, "BS2_off")

    # Log the final results
    print(f"\n[Results: N={n_final}, φ=π/2]")
    print(f"BS2 ON (Interference) → P(D1) = {c1_on/(c1_on+c2_on):.3f}")
    print(f"BS2 OFF (Which-Path)  → P(D1) = {c1_off/(c1_off+c2_off):.3f}")

    print("\n--- Simulation finished successfully. ---")


if __name__ == "__main__":
    log_filename = "delayed-choice-mzi.log"

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
        print("Figure saved to delayed-choice-mzi_plot.png")