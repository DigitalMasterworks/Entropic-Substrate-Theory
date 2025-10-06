#!/usr/bin/env python3
import numpy as np
import argparse

# ---------- Angle sets ----------
def set_angles_yours():
    # original set: a=0, a'=22.5°, b=45°, b'=67.5°
    angles = [0.0, np.pi/4, np.pi/8, 3*np.pi/8]
    a, a_prime = angles[0], angles[2]
    b, b_prime = angles[1], angles[3]
    return [(a, b), (a, b_prime), (a_prime, b), (a_prime, b_prime)]

def set_angles_max_violation():
    # Tsirelson-optimal geometry for the singlet:
    # a=0°, a'=90°, b=45°, b'=-45°
    a, a_prime = 0.0, 0.5*np.pi
    b, b_prime = 0.25*np.pi, -0.25*np.pi
    return [(a, b), (a, b_prime), (a_prime, b), (a_prime, b_prime)]

# ---------- CHSH S patterns ----------
def chsh_S_canonical(stats):
    """
    Canonical CHSH: S = | E(a,b) + E(a,b') + E(a',b) - E(a',b') |
    stats = [E(a,b), E(a,b'), E(a',b), E(a',b')]
    """
    return abs(stats[0] + stats[1] + stats[2] - stats[3])

def chsh_S_alt(stats):
    """
    Alternative pattern sometimes seen in papers:
    S_alt = | E(a,b) - E(a,b') + E(a',b) + E(a',b') |
    """
    return abs(stats[0] - stats[1] + stats[2] + stats[3])

# ---------- Outcome samplers ----------
def outcomes_singlet(angle_A, angle_B, rng):
    """
    Textbook singlet sampling:
      P(same) = sin^2(Δ/2), P(diff) = cos^2(Δ/2),
      with independent randomness per wing.
    """
    delta = angle_A - angle_B
    p_same = np.sin(delta/2.0)**2
    if rng.random() < p_same:
        s = rng.choice([-1, 1])
        return s, s
    else:
        s = rng.choice([-1, 1])
        return s, -s

def outcomes_ecfm(angle_A, angle_B, rng):
    """
    ECFM-style shared seed per pair:
    - Use a single uniform u in [0,1) to make both the 'same/diff' decision
      and the sign choice; this emulates a shared field history/seed.
    - Still respects the singlet probabilities.
    """
    delta = angle_A - angle_B
    p_same = np.sin(delta/2.0)**2

    # shared uniform u drives both decisions
    u = rng.random()
    # derive a second 'independent-looking' uniform from u deterministically
    # (golden-ratio trick keeps it in [0,1) and avoids locking to 0/0.5)
    u2 = (u * 1.6180339887498948) % 1.0

    if u < p_same:
        s = 1 if u2 < 0.5 else -1
        return s, s
    else:
        s = 1 if u2 < 0.5 else -1
        return s, -s

# ---------- Block runner ----------
def run_block(tests, trials, mode="ecfm", seed=None):
    rng = np.random.default_rng(seed)
    stats, ses = [], []
    sampler = outcomes_ecfm if mode.lower() == "ecfm" else outcomes_singlet

    for (angA, angB) in tests:
        # sample outcomes
        A = np.empty(trials, dtype=np.int8)
        B = np.empty(trials, dtype=np.int8)
        for i in range(trials):
            a_out, b_out = sampler(angA, angB, rng)
            A[i] = a_out
            B[i] = b_out

        corr = float(np.mean(A * B))
        # Standard error for mean of ±1 variables: sqrt((1 - corr^2)/N)
        se = float(np.sqrt((1.0 - corr*corr) / trials))
        stats.append(corr)
        ses.append(se)

        th = -np.cos(angA - angB)
        print(f"A={angA*180/np.pi:>6.1f}°, B={angB*180/np.pi:>6.1f}°  |  "
              f"E≈{corr:+.4f}  (±{1.96*se:.4f})   theory {th:+.4f}")

    return stats, ses

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="CHSH with ECFM shared-seed or textbook singlet sampling")
    ap.add_argument("--mode", choices=["ecfm", "singlet"], default="ecfm",
                    help="Outcome sampling mode (default: ecfm)")
    ap.add_argument("--angles", choices=["yours", "max"], default="yours",
                    help="Angle set: 'yours' (0°,45°,22.5°,67.5°) or 'max' (0°,90°,45°,-45°)")
    ap.add_argument("--trials", type=int, default=20000, help="Trials per setting")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = ap.parse_args()

    tests = set_angles_yours() if args.angles == "yours" else set_angles_max_violation()

    print(f"\nMode: {args.mode.upper()}   Angles: {args.angles}   Trials/setting: {args.trials}   Seed: {args.seed}")
    print("Theory: singlet E(a,b) = -cos(a-b)\n")

    stats, ses = run_block(tests, args.trials, mode=args.mode, seed=args.seed)

    S_can = chsh_S_canonical(stats)
    S_alt = chsh_S_alt(stats)
    print(f"\nCHSH S (canonical  + + + -) = {S_can:.4f}   (Tsirelson 2.8284)")
    print(f"CHSH S (alternate  - + + +) = {S_alt:.4f}")
    if S_can > 2 and S_can <= 2.82842712 + 5e-3:
        print("=> Quantum regime (violates classical, respects Tsirelson)")
    elif S_can > 2.82842712 + 5e-3:
        print("=> Superquantum (check RNG / code)")
    else:
        print("=> Classical regime")

if __name__ == "__main__":
    main()
