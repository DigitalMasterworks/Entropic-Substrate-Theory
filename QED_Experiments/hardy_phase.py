# hardy_phase.py
# Hardy with strict constraints: three zeros exact, entanglement bounded away from 0,
# and nontrivial marginals for primed settings. Pure Python.

import math, random
from math import sin, cos, pi

rnd = random.Random(11)

def wrap(a):
    TAU = 2*pi
    a = (a + pi) % TAU
    if a <= 0.0: a += TAU
    return a - pi

def plus(th):   return (math.cos(th), math.sin(th))
def minus(th):  return (math.sin(th), -math.cos(th))

def amp_schmidt(zeta, vA, vB):
    c, s = math.cos(zeta), math.sin(zeta)
    return c*(vA[0]*vB[0]) + s*(vA[1]*vB[1])

def prob_joint(zeta, vA, vB):
    A = amp_schmidt(zeta, vA, vB)
    return A*A

def marg_Aplus(zeta, a_th):
    # P(A'=+) when only A' measured (B traced out)
    c, s = math.cos(zeta), math.sin(zeta)
    ca, sa = math.cos(a_th), math.sin(a_th)
    return (c*c)*(ca*ca) + (s*s)*(sa*sa)

def marg_Bplus(zeta, b_th):
    c, s = math.cos(zeta), math.sin(zeta)
    cb, sb = math.cos(b_th), math.sin(b_th)
    return (c*c)*(cb*cb) + (s*s)*(sb*sb)

def hardy_probs(zeta, a, ap, b, bp):
    Ppp   = prob_joint(zeta, plus(a),  plus(b))    # success
    Papbp = prob_joint(zeta, plus(ap), plus(bp))   # must 0
    Papbm = prob_joint(zeta, plus(ap), minus(b))   # must 0
    Pambp = prob_joint(zeta, minus(a), plus(bp))   # must 0
    return Ppp, Papbp, Papbm, Pambp

def objective(zeta, a, ap, b, bp, lam_zero=1e10, lam_marg=1e3, delta=0.05):
    Ppp, Z1, Z2, Z3 = hardy_probs(zeta, a, ap, b, bp)
    # zero penalties
    pen_zero = Z1 + Z2 + Z3
    # marginal penalties (enforce nontrivial primed use)
    mA = marg_Aplus(zeta, ap)
    mB = marg_Bplus(zeta, bp)
    pen_marg = max(0.0, delta - mA) + max(0.0, delta - mB)
    # we minimize: force zeros and marginals, maximize Ppp
    obj = (5e10)*pen_zero + (2e2)*pen_marg - Ppp + 1e-8*(a*a+ap*ap+b*b+bp*bp)
    return obj, (Ppp, Z1, Z2, Z3, mA, mB)

def search(N=30000, zmin=pi/24, zmax=pi/6, delta=0.05):
    best, best_obj = None, 1e300
    for _ in range(N):
        zeta = rnd.uniform(zmin, zmax)
        a    = rnd.uniform(-pi/2, pi/2)
        ap   = rnd.uniform(-pi/2, pi/2)
        b    = rnd.uniform(-pi/2, pi/2)
        bp   = rnd.uniform(-pi/2, pi/2)
        step = 0.10
        obj, vals = objective(zeta, a, ap, b, bp, delta=delta)

        for _ in range(200):
            improved = False
            for var in range(5):
                for sgn in (+1, -1):
                    zz, aa, aap, bb, bbp = zeta, a, ap, b, bp
                    if var==0:
                        zz = min(max(zmin, zeta + sgn*step*0.2), zmax)
                    elif var==1:
                        aa = wrap(a + sgn*step)
                    elif var==2:
                        aap = wrap(ap + sgn*step)
                    elif var==3:
                        bb = wrap(b + sgn*step)
                    else:
                        bbp = wrap(bp + sgn*step)
                    cand_obj, cand_vals = objective(zz, aa, aap, bb, bbp, delta=delta)
                    if cand_obj < obj:
                        zeta, a, ap, b, bp = zz, aa, aap, bb, bbp
                        obj, vals = cand_obj, cand_vals
                        improved = True
            step = max(0.5*step, 0.003)
            if not improved and step <= 0.004:
                break

        if obj < best_obj:
            best_obj, best = obj, (zeta, a, ap, b, bp, vals)
    return best

def fmt(rad): return f"{rad:+.5f} rad ({rad*180/math.pi:+.2f}°)"

if __name__ == "__main__":
    print("== Hardy (strict: zeros exact, entanglement & marginals constrained) ==")
    zeta, a, ap, b, bp, (Ppp, Z1, Z2, Z3, mA, mB) = search(N=20000, zmin=pi/24, zmax=pi/6, delta=0.03)

    print("\nAngles:")
    print("  zeta :", fmt(zeta))
    print("  a    :", fmt(a))
    print("  a'   :", fmt(ap))
    print("  b    :", fmt(b))
    print("  b'   :", fmt(bp))

    print("\nProbabilities:")
    print(f"  P(A=+, B=+)         = {Ppp:.6f}   (target: > 0)")
    print(f"  P(A'=+, B'=+)       = {Z1:.3e} (target: 0)")
    print(f"  P(A'=+, B=-)        = {Z2:.3e} (target: 0)")
    print(f"  P(A=- , B'=+)       = {Z3:.3e} (target: 0)")
    print(f"  P(A'=+) marginal    = {mA:.6f}   (>= 0.05)")
    print(f"  P(B'=+) marginal    = {mB:.6f}   (>= 0.05)")

    zeros_ok = (Z1<1e-9 and Z2<1e-9 and Z3<1e-9)
    print("\nSummary:")
    print(f"  success P(A=+,B=+)  ≈ {Ppp:.6f}")
    print(f"  zero constraints ok  = {zeros_ok}")
    print("  reference optimum ≈ 0.090170")