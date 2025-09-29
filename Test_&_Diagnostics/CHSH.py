import numpy as np

# Lattice and simulation parameters
N = 2000
timesteps = 1000
dt = 0.02
c = 1.0
sep = 80
centerA, centerB = N//2 - sep//2, N//2 + sep//2

# CHSH angles
angles = [0, np.pi/4, np.pi/8, 3*np.pi/8]
a, a_prime = angles[0], angles[2]
b, b_prime = angles[1], angles[3]
tests = [(a, b), (a, b_prime), (a_prime, b), (a_prime, b_prime)]

NTRIALS = 200

def sign(x): return 1 if x >= 0 else -1

def quantum_outcomes(angle_A, angle_B):
    # Quantum rule: P(same) = 0.5 * sin^2((a-b)/2), P(diff) = 0.5 * cos^2((a-b)/2)
    delta = angle_A - angle_B
    p_same = 0.5 * (np.sin(delta / 2))**2
    # Randomly select "which" way it goes (same/diff)
    if np.random.rand() < p_same:
        r = np.random.choice([-1, 1])
        return r, r
    else:
        r = np.random.choice([-1, 1])
        return r, -r

def run_trial(angle_A, angle_B):
    S = np.ones(N)
    C = np.zeros(N)
    measured_A = measured_B = False
    outcome_A = outcome_B = 0
    t_A = t_B = None

    # At t=10, measure A, propagate to B at c=1
    for t in range(timesteps):
        lap = np.roll(S, 1) + np.roll(S, -1) - 2 * S
        S_new = S + dt * c * lap
        S_new = np.clip(S_new, 0, 1)
        C_new = 1 - S_new
        S, C = S_new, C_new

        if t == 10 and not measured_A:
            # When A measured, immediately determine *both* quantum outcomes (since entangled)
            oA, oB = quantum_outcomes(angle_A, angle_B)
            outcome_A = oA
            # Only A's region is collapsed immediately
            S[centerA-1:centerA+2] = 0.0
            C[centerA-1:centerA+2] = 1.0
            measured_A = True
            t_A = t
            quantum_B_outcome = oB  # Save B's predetermined outcome (quantum)

        if measured_A and not measured_B:
            if t - t_A >= abs(centerA - centerB):
                outcome_B = quantum_B_outcome
                S[centerB-1:centerB+2] = 0.0
                C[centerB-1:centerB+2] = 1.0
                measured_B = True
                t_B = t
                break

    return outcome_A, outcome_B

def run_CHSH():
    stats = []
    for (angA, angB) in tests:
        results = [run_trial(angA, angB) for _ in range(NTRIALS)]
        A = np.array([r[0] for r in results])
        B = np.array([r[1] for r in results])
        corr = np.mean(A * B)
        stats.append(corr)
        print(f"Settings (A={angA*180/np.pi:.1f}°, B={angB*180/np.pi:.1f}°): Correlation = {corr:.3f}")

    S_val = abs(stats[0] - stats[1] + stats[2] + stats[3])
    print(f"\nCHSH S value: {S_val:.3f}")
    print("Classical bound: S <= 2   |   Quantum (Tsirelson) bound: S <= 2.828...")
    if S_val > 2 and S_val <= 2.828:
        print("Result: Quantum regime! (violates classical, respects Tsirelson)")
    elif S_val > 2.828:
        print("Result: Superquantum (not physical)")
    else:
        print("Result: Classical regime")

if __name__ == "__main__":
    run_CHSH()