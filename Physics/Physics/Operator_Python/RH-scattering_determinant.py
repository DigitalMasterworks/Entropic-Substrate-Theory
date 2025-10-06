

import numpy as np, math
from scipy.integrate import solve_ivp
from numpy.linalg import norm

def radial_ode(u, y, lam, m):
 return [y[1], -(2.0*y[1] + (m*m - lam)*y[0])]

def fit_AB_at_cusp(u, psi, s, k_max=80, ridge=1e-8):
 k = min(k_max, len(u)); uu = u[:k]; yy = psi[:k]
 u0 = uu[0]; us = uu - u0
 a1 = s*us; a2 = (1.0 - s)*us
 a1 -= np.max(a1); a2 -= np.max(a2)
 c1 = np.exp(a1); c2 = np.exp(a2)
 n1 = float(norm(c1))+1e-30; n2 = float(norm(c2))+1e-30
 C = np.vstack([c1/n1, c2/n2]).T
 U,S,VT = np.linalg.svd(C, full_matrices=False)
 coeffs = VT.T @ ((S/(S*S+ridge)) * (U.T @ yy))
 A = coeffs[0]/n1; B = coeffs[1]/n2
 return complex(A), complex(B)


_SOLVES = {}

def precompute_solution(m, lam_pick, span, n_eval=300, rtol=3e-7, atol=5e-9):
 key = (m, span, lam_pick, n_eval, rtol, atol)
 if key in _SOLVES:
 return _SOLVES[key]
 u0,u1 = span
 u = np.linspace(u0,u1,n_eval)
 y0 = [1.0, 1.0]
 sol = solve_ivp(
 radial_ode, (u0,u1), y0, t_eval=u,
 args=(lam_pick, m), method="RK45", rtol=rtol, atol=atol
 )
 _SOLVES[key] = (u, sol.y[0])
 return _SOLVES[key]

def S_m_of_s_cached(m, s, lam_pick=None, spans=((-12.0,-4.0),)):
 if lam_pick is None:
 lam_pick = m*m + 100.0
 vals=[]
 for span in spans:
 u, psi = precompute_solution(m, lam_pick, span)
 A,B = fit_AB_at_cusp(u, psi, s)
 if abs(B) > 1e-14:
 vals.append(A/B)
 if not vals:
 return complex('nan')
 return np.median(vals)

def phi_of_t_cached(t, M=12, lam_pick=None, spans=((-12.0,-4.0),)):
 s = 0.5 + 1j*t
 prod = 1.0+0j
 for m in range(-M, M+1):
 Sm = S_m_of_s_cached(m, s, lam_pick=lam_pick, spans=spans)
 if np.isfinite(Sm.real) and np.isfinite(Sm.imag):
 prod *= Sm
 return prod

def scan_phi_cached(t_grid, M=12, lam_pick=None, spans=((-12.0,-4.0),)):
 Phi = np.array([phi_of_t_cached(t, M=M, lam_pick=lam_pick, spans=spans)
 for t in t_grid], dtype=complex)
 mod = np.abs(Phi)
 return Phi, mod

if __name__ == "__main__":
 ts = np.linspace(-5,5,21)
 Phi, mod = scan_phi_cached(ts, M=12, spans=((-12.0,-4.0),))
 print("[phi] |Phi| mean, std:", float(mod.mean()), float(mod.std()))
 phase = np.unwrap(np.angle(Phi))
 A = np.vstack([ts, np.ones_like(ts)]).T
 m,b = np.linalg.lstsq(A, phase, rcond=None)[0]
 print(f"[phi] phase ~ {m:.4f} t + {b:.4f}")