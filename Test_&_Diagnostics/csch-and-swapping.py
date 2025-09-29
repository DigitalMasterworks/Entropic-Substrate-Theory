import numpy as np
rng = np.random.default_rng(3)

# (A) Local hidden-variable model
def chsh_local(n=200000):
    angles = [0, np.pi/4, np.pi/8, -np.pi/8]
    a, a2, b, b2 = angles
    lam = rng.uniform(0, 2*np.pi, size=n)
    def outcome(theta):
        return np.sign(np.cos(theta - lam))
    A = outcome(a); A2 = outcome(a2)
    B = outcome(b); B2 = outcome(b2)
    def E(X,Y): return np.mean(X*Y)
    S = E(A,B) + E(A,B2) + E(A2,B) - E(A2,B2)
    return S

# (B) Encoded quantum-cosine correlations
def chsh_quantum(n=200000):
    angles = [0, np.pi/4, np.pi/8, -np.pi/8]
    a, a2, b, b2 = angles
    def corr(a,b, size):
        Δ = np.abs(a-b)
        p_same = np.cos(Δ/2)**2
        u = rng.random(size)
        same = (u < p_same)
        r = np.where(rng.random(size)<0.5, 1, -1)
        A = r
        B = np.where(same, r, -r)
        return A, B
    def Eab(aa, bb):
        A,B = corr(aa, bb, n)
        return np.mean(A*B)
    S = Eab(a,b) + Eab(a,b2) + Eab(a2,b) - Eab(a2,b2)
    return S

# Delayed entanglement swapping
def run_entanglement_swapping(n=50000, thetaA=0.0, thetaD=np.pi/4):
    lam1 = rng.uniform(0, 2*np.pi, size=n)
    lam2 = rng.uniform(0, 2*np.pi, size=n)
    def corr_angles(thetaL, lam):
        return np.where(np.cos(thetaL - lam) >= 0, 1, -1)
    A = corr_angles(thetaA, lam1)
    D = corr_angles(thetaD, lam2)
    s = np.where(np.cos((lam1 - lam2)) >= 0, 1, -1)
    corr_total = np.mean(A*D)
    sel_plus = (s==1)
    sel_minus = (s==-1)
    corr_s_plus = np.mean(A[sel_plus]*D[sel_plus]) if np.any(sel_plus) else 0.0
    corr_s_minus = np.mean(A[sel_minus]*D[sel_minus]) if np.any(sel_minus) else 0.0
    return corr_total, corr_s_plus, corr_s_minus, np.mean(sel_plus), np.mean(sel_minus)

if __name__ == "__main__":
    print(f"CHSH (local): S ≈ {chsh_local():.3f}")
    print(f"CHSH (quantum-like): S ≈ {chsh_quantum():.3f}")
    corr_total, corr_s_plus, corr_s_minus, p_plus, p_minus = run_entanglement_swapping()
    print(f"Entanglement swapping uncond: {corr_total:.3f}")
    print(f"  Condition (+): {corr_s_plus:.3f} ({p_plus:.2f} fraction)")
    print(f"  Condition (-): {corr_s_minus:.3f} ({p_minus:.2f} fraction)")