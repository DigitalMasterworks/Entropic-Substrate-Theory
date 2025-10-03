import numpy as np
import matplotlib.pyplot as plt

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
            if rng.random() < 0.5:
                out = BS @ after_phase
                p1 = np.abs(out[0])**2
                if rng.random() < p1: clicks_D1 += 1
                else: clicks_D2 += 1
            else:
                pA = np.abs(after_phase[0])**2
                if rng.random() < pA: clicks_D1 += 1
                else: clicks_D2 += 1
        else:
            raise ValueError("Unknown mode")
    return clicks_D1, clicks_D2

if __name__ == "__main__":
    phis = np.linspace(0, 2*np.pi, 25)
    D1_on = []; D1_off = []; D1_delayed_recombined = []; D1_delayed_whichpath = []
    for phi in phis:
        n=4000
        c1, _ = run_trials(n_trials=n, phi=phi, mode="BS2_on")
        D1_on.append(c1/n)
        c1o, _ = run_trials(n_trials=n, phi=phi, mode="BS2_off")
        D1_off.append(c1o/n)
        c1r, _ = run_trials(n_trials=n, phi=phi, mode="BS2_on")
        c1w, _ = run_trials(n_trials=n, phi=phi, mode="BS2_off")
        D1_delayed_recombined.append(c1r/n)
        D1_delayed_whichpath.append(c1w/n)

    plt.figure(figsize=(8,5))
    plt.plot(phis, D1_on, marker='o', label="BS2 ON (interference)")
    plt.plot(phis, D1_off, marker='s', label="BS2 OFF (which-path)")
    plt.plot(phis, D1_delayed_recombined, '--', label="Delayed → recombine")
    plt.plot(phis, D1_delayed_whichpath, '--', label="Delayed → which-path")
    plt.xlabel("Phase φ"); plt.ylabel("P(D1 click)")
    plt.title("Delayed-Choice Mach–Zehnder")
    plt.legend(); plt.grid(True); plt.show()

    phi0 = np.pi/2
    c1_on, c2_on = run_trials(20000, phi0, "BS2_on")
    c1_off, c2_off = run_trials(20000, phi0, "BS2_off")
    print(f"φ=π/2: BS2 ON → {c1_on/(c1_on+c2_on):.3f}, BS2 OFF → {c1_off/(c1_off+c2_off):.3f}")
