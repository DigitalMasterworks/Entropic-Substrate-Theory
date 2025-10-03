#!/usr/bin/env python3
# CSreduction.py
# Geodesics in ds^2 = S^2 c^2 dt^2 - S^{-2}(dx^2+dy^2+dz^2)
# Static 2D reduction + null constraint + RK4 lensing demo (A/b fit)

import argparse
import math
import numpy as np
import sys
import os

# --- LOGGING SETUP ---
LOG_FILENAME = "CHreduction.log"
LOG_FILE = None
_original_print = print

def log_print(*args, **kwargs):
    text = kwargs.get('sep', ' ').join(map(str, args))
    end_char = kwargs.get('end', '\n')
    _original_print(text, **kwargs, file=sys.stdout)
    if LOG_FILE:
        LOG_FILE.write(text + end_char)

print = log_print
# --- END LOGGING SETUP ---

try:
    import sympy as sp
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

# ---- numeric safety helpers ----
SAFE_MIN = 1e-300
SAFE_MAX = 1e300

def safe_hypot2(x, y):
    """Overflow/underflow-safe r^2 = x^2 + y^2."""
    r = math.hypot(x, y)
    r = min(r, SAFE_MAX**0.5)
    r2 = r*r
    return r2 + 0.0 if r2 > SAFE_MIN else SAFE_MIN

def capped_exp(z, cap=50.0):
    """exp(z) with symmetric cap on the exponent to avoid overflow."""
    if z > cap:   z = cap
    if z < -cap:  z = -cap
    return math.exp(z)

# ===========================
# 1) Symbolic derivation
# ===========================
# --- Closed-form Christoffels (static 2D, S=S(x,y)) from the manuscript ---
def christoffels_closed(Sval, Sx, Sy, cval):
    """
    Return Gamma^mu_{nu rho} for mu,nu,rho in {0:t,1:x,2:y},
    using the exact expressions in the paper (static case).
    Non-zeros:
      Γ^t_{t x} = Sx/S,  Γ^t_{t y} = Sy/S
      Γ^x_{t t} = S^3 c^2 Sx,  Γ^y_{t t} = S^3 c^2 Sy
      Γ^x_{x x} = -Sx/S,  Γ^x_{y y} =  Sx/S,  Γ^x_{x y} = Γ^x_{y x} = -Sy/S
      Γ^y_{y y} = -Sy/S,  Γ^y_{x x} =  Sy/S,  Γ^y_{x y} = Γ^y_{y x} = -Sx/S
    """
    invS = 1.0 / Sval
    S3c2 = (Sval**3) * (cval**2)

    # build a dense 3x3x3 filled with zeros then set entries
    G = [[[0.0 for _ in range(3)] for _ in range(3)] for _ in range(3)]

    # t-row (mu=0)
    G[0][0][1] =  Sx*invS
    G[0][1][0] =  Sx*invS
    G[0][0][2] =  Sy*invS
    G[0][2][0] =  Sy*invS

    # x-row (mu=1)
    G[1][0][0] =  S3c2 * Sx
    G[1][1][1] = -Sx*invS
    G[1][2][2] =  Sx*invS
    G[1][1][2] = -Sy*invS
    G[1][2][1] = -Sy*invS

    # y-row (mu=2)
    G[2][0][0] =  S3c2 * Sy
    G[2][2][2] = -Sy*invS
    G[2][1][1] =  Sy*invS
    G[2][1][2] = -Sx*invS
    G[2][2][1] = -Sx*invS

    return G


def geodesic_rhs_closed(lam, U, S_provider, pars):
    """
    Exact null geodesic RHS using closed-form Christoffels.
    U = [t, x, y, tp, xp, yp]
    """
    t0, x0, y0, tp, xp, yp = U
    # clamp extreme positions to avoid downstream overflow
    if x0 > 1e150: x0 = 1e150
    if x0 < -1e150: x0 = -1e150
    if y0 > 1e150: y0 = 1e150
    if y0 < -1e150: y0 = -1e150
    Sval, Sx, Sy = S_provider(x0, y0, **pars)
    G = christoffels_closed(Sval, Sx, Sy, pars['cval'])

    u = [tp, xp, yp]
    a = [0.0, 0.0, 0.0]
    for mu in range(3):
        s = 0.0
        for nu in range(3):
            for rho in range(3):
                s += G[mu][nu][rho] * u[nu] * u[rho]
        a[mu] = -s

    # return dU/dλ
    return np.array([tp, xp, yp, a[0], a[1], a[2]], dtype=float)

def derive_symbolics():
    """
    Build Christoffels and geodesics, then reduce to static-2D + null
    and print the compact ray ODEs.
    """
    if not HAVE_SYMPY:
        print("[warn] sympy not available; skipping symbolic derivation.")
        return

    # symbols
    t, x, y, z, c = sp.symbols('t x y z c', positive=True)
    S = sp.Function('S')(t, x, y, z)
    coords = (t, x, y, z)

    # metric
    g = sp.diag(S**2*c**2, -S**(-2), -S**(-2), -S**(-2))
    g_inv = g.inv()

    # Christoffels
    Gamma = [[[
        sp.simplify(
            sp.Rational(1,2) * sum(
                g_inv[mu,alpha]*(
                    sp.diff(g[alpha, rho], coords[nu]) +
                    sp.diff(g[alpha, nu], coords[rho]) -
                    sp.diff(g[nu,  rho], coords[alpha])
                )
                for alpha in range(4)
            )
        )
        for rho in range(4)] for nu in range(4)] for mu in range(4)]

    lam = sp.symbols('lam')
    T = sp.Function('T')(lam)
    X = sp.Function('X')(lam)
    Y = sp.Function('Y')(lam)
    Z = sp.Function('Z')(lam)
    Xs = [T, X, Y, Z]
    dXs = [sp.diff(q, lam) for q in Xs]
    ddXs = [sp.diff(v, lam) for v in dXs]

    # full geodesics
    geos = []
    for mu in range(4):
        acc = ddXs[mu]
        for nu in range(4):
            for rho in range(4):
                acc += Gamma[mu][nu][rho].subs({t:T, x:X, y:Y, z:Z}) * dXs[nu] * dXs[rho]
        geos.append(sp.simplify(acc))

    # Static 2D reduction: S = S(x,y); ∂_t S = ∂_z S = 0; z'=0
    stat2d = {
        sp.diff(S, t): 0,
        sp.diff(S, z): 0,
        Z: 0, sp.diff(Z, lam): 0, sp.diff(Z, (lam,2)): 0
    }
    geos_stat2d = [sp.simplify(g.subs(stat2d)) for g in geos]

    print("\n=== Static-2D geodesics (before null) ===")
    for i, gexpr in enumerate(geos_stat2d):
        print(f"coord {i}:")
        print(sp.simplify(gexpr), "\n")

    # Null constraint: S^2 c^2 T'^2 = S^{-2}(X'^2 + Y'^2)
    # We'll replace S^3 c^2 T'^2 with (X'^2+Y'^2)/S
    # To show the compact final form, derive the known reduced equations:
    # From manuscript algebra (and direct substitution) one obtains:
    #   X'' = -2 (S_x/S) X'^2 - 2 (S_y/S) X' Y'
    #   Y'' = -2 (S_y/S) Y'^2 - 2 (S_x/S) X' Y'
    # where S_x = ∂S/∂x, S_y = ∂S/∂y.
    Sx = sp.Function('Sx')(X, Y)
    Sy = sp.Function('Sy')(X, Y)

    print("=== Null-reduced ray ODEs (compact form; S=S(x,y)) ===")
    print("X'' = -2 (S_x/S) * X'^2 - 2 (S_y/S) * X' * Y'")
    print("Y'' = -2 (S_y/S) * Y'^2 - 2 (S_x/S) * X' * Y'")
    print("with S_x = ∂S/∂x, S_y = ∂S/∂y, evaluated along the ray.\n")


# =====================================
# 2) Numeric ray integrator (RK4)
# =====================================
def S_pointmass(x, y, GM=1.0, soft=1.0, cval=1.0, exact_exp=True):
    """
    Weak-field toy: Phi = -GM / sqrt(x^2+y^2+soft^2).
    S ≈ exp(Phi/c^2) (exact_exp=True) or 1 + Phi/c^2 (linearized).
    (overflow-safe)
    """
    r2 = safe_hypot2(x, y) + soft*soft
    r  = math.sqrt(r2)
    Phi = -GM / r
    if exact_exp:
        return capped_exp(Phi/(cval*cval))
    else:
        return 1.0 + Phi/(cval*cval)


def S_grad_pointmass(x, y, GM=1.0, soft=1.0, cval=1.0, exact_exp=True):
    """
    Gradients of S for the same model (overflow-safe).
    If exact_exp: S = exp(Phi/c^2) => ∇S = S * (∇Phi)/c^2
    Phi = -GM / r  => ∇Phi = GM * (x,y) / r^3
    """
    r2 = safe_hypot2(x, y) + soft*soft
    r  = math.sqrt(r2)
    Phi = -GM / r
    if exact_exp:
        Sval = capped_exp(Phi/(cval*cval))
        fac  = Sval/(cval*cval)
        dPhidx = GM * x / (r**3)
        dPhidy = GM * y / (r**3)
        Sx = fac * dPhidx
        Sy = fac * dPhidy
    else:
        Sval = 1.0 + Phi/(cval*cval)
        dPhidx = GM * x / (r**3)
        dPhidy = GM * y / (r**3)
        Sx = dPhidx / (cval*cval)
        Sy = dPhidy / (cval*cval)
    return Sval, Sx, Sy


def ray_rhs_compact(x, y, vx, vy, Sval, Sx, Sy):
    """
    Compact null-reduced ODEs (static S):
      x'' = -2 (Sx/S) vx^2 - 2 (Sy/S) vx*vy
      y'' = -2 (Sy/S) vy^2 - 2 (Sx/S) vx*vy
    """
    invS = 1.0 / Sval
    ax = -2.0 * (Sx*invS) * vx*vx - 2.0 * (Sy*invS) * vx*vy
    ay = -2.0 * (Sy*invS) * vy*vy - 2.0 * (Sx*invS) * vx*vy
    return ax, ay


def rk4_step(state, h, rhs, pars):
    k1 = rhs(0.0, state, pars)
    k2 = rhs(0.0, state + 0.5*h*k1, pars)
    k3 = rhs(0.0, state + 0.5*h*k2, pars)
    k4 = rhs(0.0, state + h*k3, pars)
    return state + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def shoot_null_ray(b, Xmax, h, pars, S_provider,
                   rhs_mode="closed", store_path=False):
    t0 = 0.0
    x0 = -Xmax
    y0 = float(b)
    xp0, yp0 = 1.0, 0.0

    Sval, Sx, Sy = S_provider(x0, y0, **pars)
    tp0 = math.sqrt((xp0*xp0 + yp0*yp0) / ((pars['cval']**2)*(Sval**4)))

    U = np.array([t0, x0, y0, tp0, xp0, yp0], dtype=float)
    path = [(x0, y0)] if store_path else None

    # choose RHS
    if rhs_mode == "closed":
        rhs = lambda lam, U, P: geodesic_rhs_closed(lam, U, S_provider, P)
    else:
        rhs = lambda lam, U, P: geodesic_rhs_lambdified(lam, U, S_provider, P)

    N = int((2*Xmax)/h)
    for _ in range(N):
        U = rk4_step(U, h, rhs, pars)
        if store_path:
            path.append((U[1], U[2]))

    theta_out = math.atan2(U[5], U[4])
    alpha = theta_out
    return alpha, path


def shoot_ray(b, Xmax=400.0, h=0.1, par=None):
    """
    Launch from x=-Xmax, y=b with velocity to the +x direction.
    Return terminal outgoing angle.
    """
    if par is None:
        par = dict(GM=1.0, soft=1.0, c=1.0, exact_exp=True)

    # initial condition: head rightward
    x0, y0 = -Xmax, float(b)
    vx0, vy0 = 1.0, 0.0

    s = np.array([x0, y0, vx0, vy0], dtype=float)
    xprev, yprev = x0, y0

    N = int((2*Xmax)/h)
    for _ in range(N):
        s = rk4_step(s, h, par)
    x, y, vx, vy = s

    # incoming and outgoing angles
    theta_in  = math.atan2(0.0, 1.0)
    theta_out = math.atan2(vy, vx)
    # deflection, signed
    alpha = theta_out - theta_in
    return alpha, s


def fit_A_over_b(bs, alphas, tail_frac=0.5):
    """
    Fit alpha ≈ A/b in the weak-field tail.
    tail_frac=0.5 uses the largest 50% of b values (asymptotic regime).
    Uses a through-the-origin regression of alpha vs 1/b:
        minimize ||alpha - A*(1/b)||^2  =>  A = <alpha*(1/b)> / <(1/b)^2>
    """
    b = np.asarray(bs, dtype=float)
    a = np.asarray(alphas, dtype=float)
    # sort by b (just in case) and take the largest tail
    idx = np.argsort(b)
    b_sorted = b[idx]; a_sorted = a[idx]
    k0 = int(round((1.0 - tail_frac) * len(b_sorted)))
    b_tail = b_sorted[k0:]; a_tail = a_sorted[k0:]
    invb = 1.0 / b_tail
    num = np.sum(a_tail * invb)
    den = np.sum(invb * invb) + 1e-18
    return float(num / den)

def geodesic_rhs_lambdified(lam, U, S_provider, pars):
    raise NotImplementedError("lambdified Γ path not wired yet; use --closed")
# ===========================
# 3) CLI / main
# ===========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sympy", action="store_true", help="skip the symbolic derivation printout")
    ap.add_argument("--demo", action="store_true", help="run lensing demo and A/b fit")
    ap.add_argument("--GM", type=float, default=1.0, help="point-mass GM")
    ap.add_argument("--soft", type=float, default=1.0, help="softening length to avoid r=0 singularity")
    ap.add_argument("--cval", type=float, default=1.0, help="speed of light (units)")
    ap.add_argument("--linear", action="store_true", help="use S ≈ 1 + Phi/c^2 instead of exp(Phi/c^2)")
    ap.add_argument("--Xmax", type=float, default=400.0, help="domain half-width")
    ap.add_argument("--h", type=float, default=0.1, help="RK4 stepsize")
    ap.add_argument("--bmin", type=float, default=5.0, help="min impact parameter")
    ap.add_argument("--bmax", type=float, default=40.0, help="max impact parameter")
    ap.add_argument("--nb", type=int, default=10, help="number of b samples")
    ap.add_argument("--plot", action="store_true", help="plot a few trajectories")
    ap.add_argument("--closed", action="store_true",
                    help="use closed-form Christoffels (paper formulas)")
    ap.add_argument("--sympy-gamma", action="store_true",
                    help="use lambdified SymPy Christoffels (your current path)")
    args = ap.parse_args()

    if not args.skip_sympy:
        derive_symbolics()

    if args.demo:
        # use full null geodesics; build params with cval
        pars = dict(GM=args.GM, soft=args.soft, cval=args.cval, exact_exp=(not args.linear))

        # choose RHS mode
        mode = "closed" if args.closed else ("sympy" if args.sympy_gamma else "closed")

        bs = np.linspace(args.bmin, args.bmax, args.nb)
        alphas = []
        for b in bs:
            a, _ = shoot_null_ray(
                b, Xmax=args.Xmax, h=args.h,
                pars=pars,
                S_provider=S_grad_pointmass,
                rhs_mode=mode, store_path=False
            )
            alphas.append(a)

        A = fit_A_over_b(bs, alphas)
        print("\n=== Lensing demo (exact geodesics) ===")
        for b, a in zip(bs, alphas):
            print(f"b={b:7.3f}  alpha={a: .6e}  |alpha|={abs(a): .6e}  A_est≈{a*b: .6e}")
        print(f"\nFitted A (tail)   ≈ {A:.6e}")
        print(f"|Fitted A| (tail) ≈ {abs(A):.6e}")
        print(f"GR A (4GM/c^2)    ≈ {4*args.GM/(args.cval*args.cval):.6e}")

        if args.plot:
            try:
                import matplotlib.pyplot as plt
            except Exception:
                print("[warn] matplotlib not available, skipping plot.")
            else:
                for b in [args.bmin, 0.5*(args.bmin+args.bmax), args.bmax]:
                    _, path = shoot_null_ray(
                        b, Xmax=args.Xmax, h=args.h,
                        pars=pars,
                        S_provider=S_grad_pointmass,
                        rhs_mode=mode, store_path=True
                    )
                    xs, ys = zip(*path)
                    plt.plot(xs, ys, label=f"b={b:.1f}")
                plt.axhline(0, color='k', lw=0.4); plt.axvline(0, color='k', lw=0.4)
                plt.gca().set_aspect('equal', 'box')
                plt.xlabel("x"); plt.ylabel("y"); plt.title("Null rays (full geodesics)")
                plt.legend(); plt.tight_layout(); plt.savefig("null_rays.png", dpi=150)
                plt.close()


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