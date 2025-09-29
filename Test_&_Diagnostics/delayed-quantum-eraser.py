import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

c = 1/np.sqrt(2)
BS = np.array([[c, 1j*c],[1j*c, c]], dtype=np.complex128)

def phase(phi):
    return np.array([[1.0, 0.0],[0.0, np.exp(1j*phi)]], dtype=np.complex128)

def tag_marker(state):
    M = np.zeros((2,2), dtype=np.complex128)
    M[0,0] = state[0]
    M[1,1] = state[1]
    return M

def erase_tags(M, do_erase=True):
    if not do_erase:
        return M
    U = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=np.complex128)
    return M @ U.T

def run_eraser(n=5000, phi=0.0):
    in_state = np.array([1+0j, 0+0j])
    bucket = [0,0]; bucket_hits = [0,0]
    for _ in range(n):
        after_bs1 = BS @ in_state
        after_phase = phase(phi) @ after_bs1
        M = tag_marker(after_phase)
        M = erase_tags(M, do_erase=True)
        for k in (0,1):
            out = BS @ M[:,k]
            p1 = np.abs(out[0])**2
            if rng.random() < p1: bucket[k] += 1
            bucket_hits[k] += 1
    P_cond = [bucket[k]/bucket_hits[k] for k in (0,1)]
    return P_cond

if __name__ == "__main__":
    phis = np.linspace(0, 2*np.pi, 25)
    P_plus = []; P_minus = []
    for phi in phis:
        P_cond = run_eraser(n=3000, phi=phi)
        P_plus.append(P_cond[0])
        P_minus.append(P_cond[1])

    plt.figure(figsize=(8,5))
    plt.plot(phis, P_plus, '--', label="Condition on |+> tag")
    plt.plot(phis, P_minus, '--', label="Condition on |-> tag")
    plt.xlabel("Phase φ"); plt.ylabel("P(D1)")
    plt.title("Delayed Quantum Eraser: Conditioned Fringes")
    plt.legend(); plt.grid(True); plt.show()