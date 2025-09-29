import sympy as sp

# symbols
Phi, c = sp.symbols('Phi c')
U = Phi  # identify Newtonian potential with U

# --- Case 1: linear S = 1 + Phi/c^2 ---
S1 = 1 + Phi/c**2
g_tt1 = -(S1**2).expand()
g_rr1 = (S1**-2).series(Phi, 0, 2).removeO()

print("Case 1: S = 1 + Phi/c^2")
print("g_tt expansion:", sp.series(g_tt1, Phi, 0, 3))
print("g_rr expansion:", g_rr1)

# compare with PPN form:
# g_tt = -1 + 2U/c^2 - 2β U^2/c^4
# g_rr = 1 + 2γ U/c^2
beta1 = sp.Rational(1,2)  # matches coefficient found
gamma1 = 1
print("=> PPN beta =", beta1, ", gamma =", gamma1)
print()

# --- Case 2: exponential S = exp(Phi/c^2) ---
S2 = sp.exp(Phi/c**2)
g_tt2 = -(S2**2).expand()
g_rr2 = (S2**-2).series(Phi, 0, 2).removeO()

print("Case 2: S = exp(Phi/c^2)")
print("g_tt expansion:", sp.series(g_tt2, Phi, 0, 3))
print("g_rr expansion:", g_rr2)

# compare with PPN form
# g_tt = -1 + 2U/c^2 - 2β U^2/c^4
# g_rr = 1 + 2γ U/c^2
beta2 = 1
gamma2 = 1
print("=> PPN beta =", beta2, ", gamma =", gamma2)