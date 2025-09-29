import sympy as sp

# Symbols
Phi, c = sp.symbols('Phi c')
U = Phi  # gravitational potential

# --- Exponential mapping ---
S = sp.exp(Phi/c**2)

# Metric components in terms of S
g_tt = -(S**2).expand()
g_rr = (S**-2).series(Phi, 0, 2).removeO()

print("Metric component expansions:")
print("g_tt:", sp.series(g_tt, Phi, 0, 3))
print("g_rr:", g_rr)

# Compare with standard PPN form:
# g_tt = -1 + 2U/c^2 - 2β U^2/c^4
# g_rr = 1 + 2γ U/c^2

# For exponential mapping, β = 1, γ = 1 (as in GR)
print("PPN beta = 1, gamma = 1")