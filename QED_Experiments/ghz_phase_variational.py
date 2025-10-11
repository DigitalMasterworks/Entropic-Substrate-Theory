# ghz_phase_variational.py
# Minimal phase-only GHZ substrate model (no grids, no PDEs)
# Solves a 3-angle variational problem and prints GHZ products.

import math
from math import pi, sin, cos, copysign

# ---------------- Parameters (safe defaults) ----------------
ALPHA = 0.20   # measurement-lock weight (weak)
BETA  = 0.05   # pairwise glue (small)
GAMMA = 1.00   # triplet term (dominant)
PHI0  = pi     # global GHZ phase offset

STEP  = 0.20   # gradient descent step
ITERS = 60     # iterations (tiny)
RESTARTS = 2   # a couple of restarts (near θ and zero)

# Handedness per site (use -1 if your Y basis is mirrored by array geometry)
HANDED = (-1.0, -1.0, -1.0)

# Define bases
X_ANGLE = 0.0
Y_ANGLE = -pi/2  # choose -π/2 by convention; flip sign if needed


# ---------------- Utilities ----------------
TAU = 2.0 * pi
def wrap(a: float) -> float:
    """wrap to (-π, π]"""
    a = (a + pi) % TAU
    if a <= 0.0:
        a += TAU
    return a - pi

def sgn(x: float) -> int:
    return 1 if x >= 0.0 else -1


# ---------------- Energy & Gradient ----------------
def energy(phi, theta, alpha=ALPHA, beta=BETA, gamma=GAMMA, PHI=PHI0):
    # phi = (φA, φB, φC), theta = (θA, θB, θC)
    a, b, c = phi
    ta, tb, tc = theta
    E_meas = -alpha * (cos(a - ta) + cos(b - tb) + cos(c - tc))
    E_pair = -beta  * (cos(a - b) + cos(b - c) + cos(c - a))  # complete triangle
    E_trip = -gamma * cos(a + b + c - PHI)
    return E_meas + E_pair + E_trip

def grad(phi, theta, alpha=ALPHA, beta=BETA, gamma=GAMMA, PHI=PHI0):
    a, b, c = phi
    ta, tb, tc = theta
    # measurement terms
    ga =  alpha * sin(a - ta)
    gb =  alpha * sin(b - tb)
    gc =  alpha * sin(c - tc)
    # pairwise (derive per-edge and distribute)
    ga += beta * ( sin(a - b) - sin(c - a) )  # from -β cos(a-b) and -β cos(c-a)
    gb += beta * ( sin(b - c) - sin(a - b) )
    gc += beta * ( sin(c - a) - sin(b - c) )
    # triplet
    u = a + b + c - PHI
    trip = gamma * sin(u)
    ga += trip
    gb += trip
    gc += trip
    return (ga, gb, gc)


# ---------------- Solver ----------------
def solve_phases(theta, alpha=ALPHA, beta=BETA, gamma=GAMMA, PHI=PHI0,
                 step=STEP, iters=ITERS, restarts=RESTARTS):
    best_phi = None
    best_E = float('inf')

    seeds = [
        tuple(wrap(t) for t in theta),         # start near the requested angles
        (0.0, 0.0, 0.0),                       # start at zero
    ]
    # add tiny perturbations if more restarts requested
    while len(seeds) < restarts:
        eps = 1e-3 * (len(seeds))
        seeds.append(tuple(wrap(t + eps) for t in theta))

    for seed in seeds[:restarts]:
        phi = list(seed)
        for _ in range(iters):
            ga, gb, gc = grad(phi, theta, alpha, beta, gamma, PHI)
            phi[0] = wrap(phi[0] - step * ga)
            phi[1] = wrap(phi[1] - step * gb)
            phi[2] = wrap(phi[2] - step * gc)
        E = energy(phi, theta, alpha, beta, gamma, PHI)
        if E < best_E:
            best_E = E
            best_phi = tuple(phi)
    return best_phi, best_E

def measure_product(phi):
    # GHZ product is set by the triplet phase: sgn(cos((φA+φB+φC) - Φ))
    a, b, c = phi
    return sgn(cos((a + b + c) - PHI0))


# ---------------- GHZ Harness ----------------
def ghz_angles(setting, y_angle=Y_ANGLE, handed=HANDED):
    # Apply handedness h_s to the requested basis angles
    if setting == "XXX":
        θ = (0.0, 0.0, 0.0)
    elif setting == "XYY":
        θ = (0.0, y_angle, y_angle)
    elif setting == "YXY":
        θ = (y_angle, 0.0, y_angle)
    elif setting == "YYX":
        θ = (y_angle, y_angle, 0.0)
    else:
        raise ValueError("Unknown setting")
    return tuple(h * t for h, t in zip(handed, θ))

def run_ghz_suite():
    # 1) Calibrate a single global X sign so XXX → +1
    theta_xxx = ghz_angles("XXX")
    phi_xxx, _ = solve_phases(theta_xxx)
    prod_xxx = measure_product(phi_xxx)

    global_flip = 1
    if prod_xxx != +1:
        global_flip = -1  # one-time global flip to set XXX → +1

    def evaluate(tag):
        theta = ghz_angles(tag)
        phi, E = solve_phases(theta)
        prod = global_flip * measure_product(phi)
        qm = +1.0 if tag == "XXX" else -1.0
        status = "MATCH" if abs(prod - qm) < 1e-6 else "MISMATCH"
        print(f"{tag:>3} → product={prod:+d}  (QM: {qm:+.0f})  {status}   "
              f"[φ*=({phi[0]:+.3f},{phi[1]:+.3f},{phi[2]:+.3f}), E={E:+.5f}]")

    print("== Phase-only GHZ (variational, raw math) ==")
    print(f"params: α={ALPHA}, β={BETA}, γ={GAMMA}, Φ={PHI0:.3f}, handed={HANDED}, y={Y_ANGLE:+.3f}")
    print(f"solver: step={STEP}, iters={ITERS}, restarts={RESTARTS}")
    print("\nCalibrating global X sign so XXX → +1 ...")
    print(f"XXX calibration product (pre-flip) = {prod_xxx:+d} → global_flip={global_flip:+d}\n")

    for tag in ("XXX", "XYY", "YXY", "YYX"):
        evaluate(tag)


if __name__ == "__main__":
    run_ghz_suite()