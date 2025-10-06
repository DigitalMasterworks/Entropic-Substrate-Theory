#!/usr/bin/env python3





import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def radial_ode(r, y, lam, m):
 """
 Radial ODE:
 (r^2 R')' + [lam*r^2 - m^2] R = 0
 Written as first-order system for y = [R, R'].
 """
 R, Rp = y
 dR = Rp
 dRp = (m**2 - lam*r**2) * R / (r**2) - 2*Rp/r
 return [dR, dRp]

def jost_solution(mu, m, lam, r_span=(1e-3, 1.0), r_match=0.1):
 """
 Integrate radial ODE with Frobenius-like initial condition near r=0:
 R(r) ~ r^(1/2 + i mu)
 """
 r0 = r_span[0]

 alpha = 0.5 + 1j*mu
 R0 = r0**alpha
 Rp0 = alpha * r0**(alpha-1)
 y0 = [R0, Rp0]

 sol = solve_ivp(radial_ode, r_span, y0, t_eval=[r_match], args=(lam, m),
 rtol=1e-10, atol=1e-12)
 return sol.y[0, -1]

def scattering_ratio(mu, m, lam, r_match=0.1):
 Rin = jost_solution(mu, m, lam, r_match=r_match)
 Rout = jost_solution(-mu, m, lam, r_match=r_match)
 return Rin / Rout

def check_symmetry(m=0, lam=1.0, mu_vals=None, r_match=0.1):
 if mu_vals is None:
 mu_vals = np.linspace(0.5, 10.0, 20)
 results = []
 for mu in mu_vals:
 Splus = scattering_ratio(mu, m, lam, r_match)
 Sminus = scattering_ratio(-mu, m, lam, r_match)
 prod = Splus * Sminus
 results.append((mu, Splus, Sminus, prod))
 return results

if __name__ == "__main__":
 m = 0
 lam = 1.0
 mu_vals = np.linspace(0.5, 10, 20)

 results = check_symmetry(m, lam, mu_vals)

 print(" Scattering symmetry test (m=0)")
 print(" mu S(μ) S(-μ) Product")
 for mu, Splus, Sminus, prod in results:
 print(f"{mu:5.2f} {Splus:.6f} {Sminus:.6f} {prod:.6f}")


 prods = [abs(prod) for _,_,_,prod in results]
 plt.figure(figsize=(6,4))
 plt.plot(mu_vals, prods, "o-")
 plt.axhline(1.0, color="r", linestyle="--", label="theory 1")
 plt.xlabel("μ")
 plt.ylabel("|S(μ) S(-μ)|")
 plt.title("Scattering symmetry check")
 plt.legend()
 plt.tight_layout()
 plt.savefig("scattering_symmetry.png", dpi=150)
 print("Saved: scattering_symmetry.png")





