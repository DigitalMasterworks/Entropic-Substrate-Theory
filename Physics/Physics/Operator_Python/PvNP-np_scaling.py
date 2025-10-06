#!/usr/bin/env python3






import numpy as np
import time, argparse
import matplotlib.pyplot as plt
from numpy.linalg import lstsq


def random_3sat(n_vars, n_clauses, rng):
 clauses = []
 for _ in range(n_clauses):
 vars_ = rng.choice(np.arange(1, n_vars+1), size=3, replace=False)
 signs = rng.choice([-1,1], size=3)
 clause = list(signs * vars_)
 clauses.append(clause)
 return clauses

def verify_assignment(clauses, assignment):
 for clause in clauses:
 if not any(((lit > 0 and assignment[abs(lit)-1]==1) or
 (lit < 0 and assignment[abs(lit)-1]==0)) for lit in clause):
 return False
 return True

def energy(clauses, assignment):
 return sum(0 if verify_assignment([c], assignment) else 1 for c in clauses)

def substrate_search(clauses, n_steps=20000, T0=1.0, rng=None):
 n_vars = max(abs(lit) for c in clauses for lit in c)
 assign = rng.integers(0,2,size=n_vars)
 E = energy(clauses, assign)
 T = T0
 for step in range(n_steps):
 if E == 0: return step, assign
 var = rng.integers(0,n_vars)
 new_assign = assign.copy()
 new_assign[var] ^= 1
 E_new = energy(clauses, new_assign)
 dE = E_new - E
 if dE <= 0 or rng.random() < np.exp(-dE/T):
 assign, E = new_assign, E_new
 T = T0 * (1 - step/n_steps)
 return n_steps, assign


def run_scaling(max_vars=30, clauses_per_var=4, trials=10, rng=None):
 results = []
 for n in range(10, max_vars+1, 2):
 n_clauses = clauses_per_var * n
 search_times, verify_times = [], []
 for _ in range(trials):
 clauses = random_3sat(n, n_clauses, rng)
 t0 = time.time()
 steps, sol = substrate_search(clauses, rng=rng)
 t1 = time.time()
 search_times.append(t1-t0)
 t0 = time.time()
 ok = verify_assignment(clauses, sol)
 t1 = time.time()
 verify_times.append(t1-t0)
 results.append((n, np.mean(search_times), np.mean(verify_times)))
 print(f"n={n:2d}, search={np.mean(search_times):.4f}s, verify={np.mean(verify_times):.6f}s")
 return results


def fit_models(n_vals, search_times):
 n_vals = np.array(n_vals, float)
 y = np.array(search_times, float)


 Xp = np.column_stack([np.log(n_vals), np.ones_like(n_vals)])
 ap, bp = lstsq(Xp, np.log(y), rcond=None)[0]
 poly_pred = np.exp(Xp @ [ap, bp])


 Xe = np.column_stack([n_vals, np.ones_like(n_vals)])
 ae, be = lstsq(Xe, np.log(y), rcond=None)[0]
 exp_pred = np.exp(Xe @ [ae, be])

 def r2(y, pred): return 1 - np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
 return (ap, r2(y, poly_pred)), (ae, r2(y, exp_pred))

if __name__ == "__main__":
 parser = argparse.ArgumentParser()
 parser.add_argument("--max_vars", type=int, default=30)
 parser.add_argument("--clauses_per_var", type=int, default=4)
 parser.add_argument("--trials", type=int, default=10)
 args = parser.parse_args()

 rng = np.random.default_rng(0)
 results = run_scaling(args.max_vars, args.clauses_per_var, args.trials, rng)

 n_vals, search_times, verify_times = zip(*results)
 poly_fit, exp_fit = fit_models(n_vals, search_times)

 print("\n--- Fit results ---")
 print(f"Polynomial fit exponent ~ {poly_fit[0]:.3f}, R²={poly_fit[1]:.3f}")
 print(f"Exponential fit rate ~ {exp_fit[0]:.3f}, R²={exp_fit[1]:.3f}")

 plt.figure(figsize=(6,4))
 plt.plot(n_vals, search_times, "o-", label="substrate search")
 plt.plot(n_vals, verify_times, "s-", label="verification")
 plt.yscale("log")
 plt.xlabel("n variables")
 plt.ylabel("runtime (s, log scale)")
 plt.title("Scaling of substrate search vs verification")
 plt.legend()
 plt.tight_layout()
 plt.savefig("np_scaling.png", dpi=150)
 print("Saved: np_scaling.png")