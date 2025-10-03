import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- LOGGING SETUP ---
LOG_FILENAME = "CHSH.log"
LOG_FILE = None
_original_print = print

def log_print(*args, **kwargs):
    # Robustly handle the 'file' keyword argument to prevent errors
    kwargs_for_stdout = dict(kwargs)
    kwargs_for_stdout.pop('file', None) 

    text = kwargs.get('sep', ' ').join(map(str, args))
    end_char = kwargs.get('end', '\n')

    # Print to console (always sys.stdout)
    _original_print(text, **kwargs_for_stdout, file=sys.stdout) 

    # Log to file
    if LOG_FILE:
        LOG_FILE.write(text + end_char)

print = log_print
# --- END LOGGING SETUP ---

# Parameters
N_trials = 100_000
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Standard CHSH set
sites = [100, 200]  # Positions A and B in the substrate array

def evolve_substrate(S, steps=10):
    """Deterministic substrate evolution: Laplacian smoothing."""
    for _ in range(steps):
        lap = np.roll(S, 1) + np.roll(S, -1) - 2*S
        S += 0.01 * lap
        S = np.clip(S, 0, 1)
    return S

def substrate_measurement(S, site, angle):
    """Deterministic local measurement rule. No quantum probability."""
    # Example: threshold on local S + angle-dependent offset
    value = S[site] + 0.1*np.cos(angle)
    return 1 if value > 0.5 else -1

def quantum_measurement(angle_A, angle_B):
    """Ideal quantum correlation (singlet state) benchmark."""
    delta = angle_A - angle_B
    prob_same = np.cos(delta / 2) ** 2
    # Determine the result based on probability (anti-correlated for singlet)
    # E = -cos(delta)
    # The result A*B must be 1 with probability (1+E)/2, and -1 with probability (1-E)/2.
    # (1+E)/2 = (1 - cos(delta))/2 = sin^2(delta/2)
    # Let's use the standard E = -cos(delta) calculation
    E_ab = -np.cos(delta)
    if np.random.rand() < (1 + E_ab) / 2:
        return 1 # A*B = 1
    else:
        return -1 # A*B = -1

def random_measurement():
    return np.random.choice([1, -1])

def run_bell_experiment(mode="substrate"):
    """Runs Bell test with the given measurement mode."""
    results = { (a, b): [] for a in angles for b in angles }
    for _ in range(N_trials):
        # New substrate per trial
        S = np.ones(300)
        S[150-10:150+10] = 0.5  # Seed some structure
        S = evolve_substrate(S, steps=20)
        for a in angles:
            for b in angles:
                if mode == "substrate":
                    A = substrate_measurement(S, sites[0], a)
                    B = substrate_measurement(S, sites[1], b)
                    A_B = A * B
                elif mode == "quantum":
                    # For quantum, we calculate the product directly for a perfect singlet E=-cos(delta)
                    # We rely on the product being calculated within the function for simplicity, 
                    # as true A and B separately are tricky for a single-trial simulation.
                    # The function quantum_measurement returns the product A*B directly.
                    A_B = quantum_measurement(a, b)
                elif mode == "random":
                    A = random_measurement()
                    B = random_measurement()
                    A_B = A * B

                results[(a, b)].append(A_B)

    # Compute <A*B> for each angle pair
    E = { (a, b): np.mean(results[(a, b)]) for a in angles for b in angles }
    # Compute CHSH S-value
    S_val = abs(E[(angles[0], angles[2])] - E[(angles[0], angles[3])] +
                E[(angles[1], angles[2])] + E[(angles[1], angles[3])])
    return E, S_val

def main():
    # Run substrate-only Bell
    print("\n[Substrate-Only Bell Test]")
    E_sub, S_sub = run_bell_experiment("substrate")
    print("CHSH S (substrate):", S_sub)
    for (a, b), v in E_sub.items():
        print(f"E({a:.2f}, {b:.2f}) = {v:.3f}")

    # Run quantum benchmark (for reference)
    print("\n[Quantum Benchmark Bell Test]")
    E_q, S_q = run_bell_experiment("quantum")
    print("CHSH S (quantum):", S_q)
    for (a, b), v in E_q.items():
        print(f"E({a:.2f}, {b:.2f}) = {v:.3f}")

    # Run null/random control
    print("\n[Null/Random Bell Test]")
    E_r, S_r = run_bell_experiment("random")
    print("CHSH S (random):", S_r)
    for (a, b), v in E_r.items():
        print(f"E({a:.2f}, {b:.2f}) = {v:.3f}")

    # Plot all three
    fig, ax = plt.subplots(figsize=(7, 5))
    pairs = [ (a, b) for a in angles for b in angles ]
    x = np.arange(len(pairs))
    ax.plot(x, [E_sub[p] for p in pairs], "o-", label="Substrate Only")
    ax.plot(x, [E_q[p] for p in pairs], "s--", label="Quantum (benchmark)")
    ax.plot(x, [E_r[p] for p in pairs], "x:", label="Random/null")
    ax.set_xticks(x)
    ax.set_xticklabels([f"({a:.2f},{b:.2f})" for (a, b) in pairs], rotation=45)
    ax.set_ylabel("E(a,b)")
    ax.legend()
    plt.title("Bell Test: <A*B> for All Angle Pairs")
    plt.tight_layout()

    # --- FIX: Save and Close Plot ---
    plt.savefig('bell_test_results.png')
    plt.close()
    # --------------------------------

if __name__ == "__main__":
    try:
        f = open(LOG_FILENAME, 'w')
        LOG_FILE = f
        main()
    except Exception as e:
        _original_print(f"Fatal error during execution: {e}", file=sys.stderr)
    finally:
        if LOG_FILE:
            LOG_FILE.close()
            LOG_FILE = None