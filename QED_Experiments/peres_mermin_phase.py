# peres_mermin_phase.py
# Phase-only Peres–Mermin square (no grids, no PDEs). Millisecond runs.

import math
from math import cos, sin, pi

TAU = 2*pi

def wrap(a: float) -> float:
    a = (a + pi) % TAU
    if a <= 0.0: a += TAU
    return a - pi

def sgn(x: float) -> int:
    return 1 if x >= 0.0 else -1

# 3x3 observables indexed 0..8:
# 0 1 2
# 3 4 5
# 6 7 8
# PM contexts: rows +1, columns +1 except last column -1
CONTEXTS = [
    ("R1", [0,1,2], +1),
    ("R2", [3,4,5], +1),
    ("R3", [6,7,8], +1),
    ("C1", [0,3,6], +1),
    ("C2", [1,4,7], +1),
    ("C3", [2,5,8], -1),
]
PHI = {+1: 0.0, -1: pi}  # desired parity → phase target

# Energy and gradient for a set of contexts (triplet cos-type)
def energy(phi, contexts, gamma=1.0):
    E = 0.0
    for _, idxs, sign in contexts:
        s = sum(phi[i] for i in idxs)
        E += -gamma * cos(s - PHI[sign])
    return E

def grad(phi, contexts, gamma=1.0):
    g = [0.0]*len(phi)
    for _, idxs, sign in contexts:
        s = sum(phi[i] for i in idxs)
        t = gamma * sin(s - PHI[sign])  # derivative of -cos is +sin
        for i in idxs:
            g[i] += t
    return g

def solve(phi0, contexts, step=0.25, iters=80, restarts=2, gamma=1.0):
    best_phi, best_E = None, float('inf')
    seeds = [phi0[:], [0.0]*len(phi0)]
    while len(seeds) < restarts:
        eps = 1e-3 * (len(seeds))
        seeds.append([wrap(eps) for _ in phi0])
    for seed in seeds[:restarts]:
        phi = seed[:]
        for _ in range(iters):
            g = grad(phi, contexts, gamma)
            for i in range(len(phi)):
                phi[i] = wrap(phi[i] - step*g[i])
        E = energy(phi, contexts, gamma)
        if E < best_E:
            best_E, best_phi = E, phi[:]
    return best_phi, best_E

def context_parity(phi, idxs, sign):
    s = sum(phi[i] for i in idxs)
    return sgn(cos(s - PHI[sign]))

if __name__ == "__main__":
    # (i) Solve each context individually (should match its target parity)
    print("=== Peres–Mermin: single-context satisfiability ===")
    for name, idxs, sign in CONTEXTS:
        phi0 = [0.0]*9
        phi, E = solve(phi0, [(name, idxs, sign)], step=0.3, iters=60, restarts=2, gamma=1.0)
        p = context_parity(phi, idxs, sign)
        print(f"{name}: parity={p:+d}  target={sign:+d}   E={E:+.6f}   indices={idxs}")

    # (ii) Try to satisfy all six simultaneously (contextuality ⇒ at least one fails)
    print("\n=== Peres–Mermin: all-context optimization (contextuality test) ===")
    phi0 = [0.0]*9
    phi, E = solve(phi0, CONTEXTS, step=0.2, iters=200, restarts=3, gamma=1.0)
    print(f"Total energy E={E:+.6f}")
    failures = 0
    for name, idxs, sign in CONTEXTS:
        p = context_parity(phi, idxs, sign)
        ok = "MATCH" if p == sign else "MISMATCH"
        if p != sign: failures += 1
        print(f"{name}: parity={p:+d}  target={sign:+d}   {ok}   indices={idxs}")
    print(f"\nResult: {failures} context(s) violated (nonzero ⇒ state-independent contextuality).")