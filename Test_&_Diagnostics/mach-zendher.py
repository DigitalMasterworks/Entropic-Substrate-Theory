import numpy as np
import matplotlib.pyplot as plt

# Delayed-Choice Mach–Zehnder Interferometer (amplitude model)
# We simulate single-photon trials with two paths (A,B).
# BS1 splits amplitude 50/50; path phase difference φ accrued;
# BS2 can be present or absent. "Delayed choice" means we decide BS2 at the
# very last step (after BS1 already acted). We record detector clicks.

rng = np.random.default_rng(0)

def bs_matrix(theta=np.pi/4):
    # 50/50 beam splitter with phase convention:
    # [outA]   [ 1/sqrt(2)   i/np.sqrt(2)] [inA]
    # [outB] = [ i/np.sqrt(2) 1/np.sqrt(2)] [inB]
    c = 1/np.sqrt(2)
    return np.array([[c, 1j*c],
                     [1j*c, c]], dtype=np.complex128)

BS = bs_matrix()

def propagate(phi):
    # path phase accumulation matrix: A gets 0, B gets phi
    return np.array([[1.0, 0.0],
                     [0.0, np.exp(1j*phi)]], dtype=np.complex128)

def run_trials(n_trials=5000, phi=0.0, mode="BS2_on"):
    # Input state: |1,0> (photon enters port A)
    in_state = np.array([1.0+0j, 0.0+0j])
    clicks_D1 = 0
    clicks_D2 = 0
    for _ in range(n_trials):
        # BS1 acts
        after_bs1 = BS @ in_state
        # phase accrual
        after_phase = propagate(phi) @ after_bs1
        
        if mode == "BS2_on":
            out = BS @ after_phase
            p1 = np.abs(out[0])**2
            # stochastic single click
            if rng.random() < p1:
                clicks_D1 += 1
            else:
                clicks_D2 += 1
        
        elif mode == "BS2_off":
            # No recombination; detectors sit on each arm directly
            # Choose path by Born rule on path amplitudes
            pA = np.abs(after_phase[0])**2
            if rng.random() < pA:
                clicks_D1 += 1  # detector on arm A
            else:
                clicks_D2 += 1  # detector on arm B
        
        elif mode == "delayed_choice":
            # Decide presence of BS2 AFTER BS1 and phase accrual
            # (in practice, this is identical to "on" vs "off" applied at last step,
            #  but we keep it explicit to show timing.)
            if rng.random() < 0.5:
                # choose recombine
                out = BS @ after_phase
                p1 = np.abs(out[0])**2
                if rng.random() < p1: clicks_D1 += 1
                else: clicks_D2 += 1
            else:
                # choose path detection
                pA = np.abs(after_phase[0])**2
                if rng.random() < pA: clicks_D1 += 1
                else: clicks_D2 += 1
        else:
            raise ValueError("Unknown mode")
    return clicks_D1, clicks_D2

# Sweep φ to show fringes (BS2 on) and no fringes (BS2 off). Include delayed choice.
phis = np.linspace(0, 2*np.pi, 25)
D1_on = []; D1_off = []; D1_delayed_recombined = []; D1_delayed_whichpath = []

# For delayed choice, we split outcomes by which branch was chosen at the end.
for phi in phis:
    n=4000
    c1, c2 = run_trials(n_trials=n, phi=phi, mode="BS2_on")
    D1_on.append(c1/n)
    c1o, c2o = run_trials(n_trials=n, phi=phi, mode="BS2_off")
    D1_off.append(c1o/n)
    
    # For delayed, run and record separately by forcing branch in two runs
    c1r, c2r = run_trials(n_trials=n, phi=phi, mode="BS2_on")
    c1w, c2w = run_trials(n_trials=n, phi=phi, mode="BS2_off")
    D1_delayed_recombined.append(c1r/n)
    D1_delayed_whichpath.append(c1w/n)

plt.figure(figsize=(8,5))
plt.plot(phis, D1_on, marker='o', label="BS2 ON (interference)")
plt.plot(phis, D1_off, marker='s', label="BS2 OFF (which-path)")
plt.plot(phis, D1_delayed_recombined, linestyle='--', label="Delayed → recombine subset")
plt.plot(phis, D1_delayed_whichpath, linestyle='--', label="Delayed → which-path subset")
plt.xlabel("Phase difference φ")
plt.ylabel("P(D1 click)")
plt.title("Delayed-Choice Mach–Zehnder: Fringes vs Which-Path")
plt.legend()
plt.grid(True)
plt.show()

# Report a representative point
phi0 = np.pi/2
c1_on, c2_on = run_trials(20000, phi0, "BS2_on")
c1_off, c2_off = run_trials(20000, phi0, "BS2_off")
print(f"At φ=π/2: BS2 ON → P(D1)≈{c1_on/(c1_on+c2_on):.3f}; BS2 OFF → P(D1)≈{c1_off/(c1_off+c2_off):.3f}")