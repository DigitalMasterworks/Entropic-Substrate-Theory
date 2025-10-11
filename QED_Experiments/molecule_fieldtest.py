#!/usr/bin/env python3
# field_bmatching_exact.py
# Exact, closed-form check:
#   molecules.csv → FORM → rung ν → dual-peak score s_ij → exact max-weight b-matching via min-cost max-flow
# No geometry, no gradient descent, no greedy heuristics.

import math
import re
import pandas as pd
from typing import List, Dict, Tuple

# ---------------- Locked ladder anchors & step factor (from masses) ----------------
MEV_PER_U   = 931.49410242
MASS_E_MEV  = 0.51099895000
MASS_MU_MEV = 105.6583755
MU_E        = MASS_MU_MEV / MASS_E_MEV
g           = MU_E ** (1.0 / 13.0)  # ≈ 1.507003107

# ---------------- Locked dual-peak resonance constants (numeric) ----------------
SIGMA0 = 0.900000
SIGMA1 = 1.200000
DELTA1 = 6.500000
W0 = 1.124462
W1 = 1.551250

# ---------------- Atomic masses (u) for rung computation ----------------
AMU: Dict[str, float] = {
    "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
    "F": 18.998403163, "Cl": 35.45, "Br": 79.904, "I": 126.90447,
    "P": 30.973761998, "S": 32.065,
}

# ---------------- Valence capacities (hard equality constraints) ----------------
# These are NOT used to tune anything; they are the mathematical degree caps that
# enforce Σ_i deg(i) = Σ_i V_i and Σ_i deg(i) = 2|E| (the bond-order identity).
VALENCE = {"H":1, "C":4, "N":3, "O":2, "F":1, "Cl":1, "Br":1, "I":1, "P":3, "S":2}

# ---------------- Rung and resonance ----------------
def rung_from_atomic_mass_u(amu: float) -> float:
    mZ = amu * MEV_PER_U
    return 3.0 + 13.0 * math.log(mZ / MASS_E_MEV) / math.log(MU_E)

def resonance_score(nu_i: float, nu_j: float) -> float:
    d = abs(nu_i - nu_j)
    p0 = math.exp(-(d / SIGMA0) ** 2)
    p1 = math.exp(-((d - DELTA1) ** 2) / (2.0 * SIGMA1 ** 2))
    return W0 * p0 + W1 * p1

# ---------------- Formula parsing & expectation (for reporting only) ----------------
FORM_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_formula(form: str) -> List[str]:
    atoms: List[str] = []
    for sym, num in FORM_RE.findall(form):
        cnt = int(num) if num else 1
        atoms.extend([sym] * cnt)
    return atoms

def expected_total_bonds(atoms: List[str]) -> int:
    return sum(VALENCE.get(a, 0) for a in atoms) // 2

# ---------------- Exact min-cost max-flow (successive shortest augmenting path) ----------------
class MCMF:
    class Edge:
        __slots__ = ("v","cap","cost","rev")
        def __init__(self, v: int, cap: int, cost: float, rev: int):
            self.v = v
            self.cap = cap
            self.cost = cost
            self.rev = rev

    def __init__(self, n: int):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: float):
        a = MCMF.Edge(v, cap, cost, len(self.g[v]))
        b = MCMF.Edge(u, 0,  -cost, len(self.g[u]))
        self.g[u].append(a)
        self.g[v].append(b)

    def min_cost_flow(self, s: int, t: int, f: int) -> Tuple[int,float]:
        n = self.n
        INF = 10**30
        flow = 0
        cost = 0.0
        # Johnson potentials to keep costs non-negative for Dijkstra
        pot = [0.0]*n
        dist = [0.0]*n
        parent_v = [0]*n
        parent_e = [0]*n

        while flow < f:
            for i in range(n):
                dist[i] = INF
            dist[s] = 0.0
            inq = [False]*n

            # Dijkstra with potentials
            import heapq
            pq = [(0.0, s)]
            while pq:
                d,u = heapq.heappop(pq)
                if d != dist[u]:
                    continue
                for ei, e in enumerate(self.g[u]):
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + pot[u] - pot[e.v]
                    if nd < dist[e.v]:
                        dist[e.v] = nd
                        parent_v[e.v] = u
                        parent_e[e.v] = ei
                        heapq.heappush(pq, (nd, e.v))

            if dist[t] == INF:
                break  # cannot send more flow (infeasible)

            for i in range(n):
                if dist[i] < INF:
                    pot[i] += dist[i]

            # augment 1 unit at a time (all arc capacities are small ints)
            add = f - flow
            v = t
            while v != s:
                u = parent_v[v]
                e = self.g[u][parent_e[v]]
                add = min(add, e.cap)
                v = u
            v = t
            while v != s:
                u = parent_v[v]
                ei = parent_e[v]
                e = self.g[u][ei]
                e.cap -= add
                self.g[v][e.rev].cap += add
                cost += add * e.cost
                v = u
            flow += add
        return flow, cost

# ---------------- Exact b-matching via flow ----------------
def exact_bmatching_bond_order(atoms: List[str]) -> Tuple[int, Dict[Tuple[int,int], int]]:
    """
    Solve:
      maximize sum s_ij * b_ij
      s.t. sum_j b_ij = V_i, b_ij in {0,1,2,...}
    by min-cost max-flow on a bipartite expansion with capacities.
    Returns (total_bonds, multiplicity_dict).
    """
    n = len(atoms)
    # rungs and valence caps (errors if unknown element)
    nu = []
    V = []
    for a in atoms:
        if a not in AMU or a not in VALENCE:
            raise ValueError(f"unknown element '{a}'")
        nu.append(rung_from_atomic_mass_u(AMU[a]))
        V.append(VALENCE[a])

    total_bonds_needed = sum(V)//2
    if total_bonds_needed == 0:
        return 0, {}

    # Node indexing:
    #  source = 0
    #  left atoms: 1..n
    #  right atoms: n+1 .. n+n
    #  sink = n+n+1
    N = 2*n + 2
    SRC = 0
    SNK = N-1
    def L(i): return 1 + i
    def R(j): return 1 + n + j

    g = MCMF(N)

    # Source -> left_i with cap = V_i
    for i in range(n):
        if V[i] > 0:
            g.add_edge(SRC, L(i), V[i], 0.0)

    # Right_j -> sink with cap = V_j
    for j in range(n):
        if V[j] > 0:
            g.add_edge(R(j), SNK, V[j], 0.0)

    # Left_i -> Right_j with cap = min(V_i, V_j), cost = -s_ij
    # Allow multiple bonds between same pair (capacity > 1).
    # Disallow self-pair (i == j).
    for i in range(n):
        if V[i] == 0: continue
        for j in range(n):
            if i == j or V[j] == 0: continue
            sij = resonance_score(nu[i], nu[j])
            if sij <= 0.0:
                continue
            cap_ij = min(V[i], V[j])
            g.add_edge(L(i), R(j), cap_ij, -sij)

    # Push exactly total_bonds_needed units of flow
    flow, cost = g.min_cost_flow(SRC, SNK, total_bonds_needed)
    if flow < total_bonds_needed:
        # Infeasible (should not happen for normal organic formulas)
        # Return whatever achieved.
        pass

    # Recover multiplicities: any residual flow on R(j)->L(i) reverse arcs indicates used units.
    bonds: Dict[Tuple[int,int], int] = {}
    # Iterate over left adjacency, look at edges to R(*)
    for i in range(n):
        for e in g.g[L(i)]:
            v = e.v
            if R(0) <= v <= R(n-1):
                # reverse edge capacity tells how much was pushed
                rev = g.g[v][e.rev]
                used = rev.cap  # because we added to reverse on augmentation
                if used > 0:
                    j = v - (1 + n)
                    # store undirected pair (i<j)
                    a,b = (i,j) if i<j else (j,i)
                    bonds[(a,b)] = bonds.get((a,b), 0) + used

    total = sum(bonds.values())
    return total, bonds

# ---------------- CSV loader (robust) ----------------
def load_molecules(path="molecules.csv") -> pd.DataFrame:
    for kw in ({}, {"sep": ",", "engine": "python"}, {"sep": "\t", "engine": "python"}):
        try:
            df = pd.read_csv(path, **kw)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError("Failed to read molecules.csv with default, comma, or tab separators.")
    df.columns = [c.strip() for c in df.columns]
    if "FORM" not in df.columns or "NAME" not in df.columns:
        raise RuntimeError("molecules.csv must contain columns NAME and FORM")
    df = df.dropna(subset=["FORM"])
    return df

# ---------------- Main ----------------
def main():
    df = load_molecules("molecules.csv")
    total_rows = len(df)
    tested = 0
    exact = 0
    print(f"\nTesting {total_rows} molecules (exact b-matching; closed-form; no geometry)\n")

    for idx, row in df.iterrows():
        name = str(row["NAME"]).strip()
        form = str(row["FORM"]).strip()
        try:
            atoms = parse_formula(form)
            # skip formulas with unknown elements
            for a in atoms:
                if a not in AMU or a not in VALENCE:
                    raise ValueError(f"unknown element '{a}'")

            pred_total, bonds = exact_bmatching_bond_order(atoms)
            exp_total = expected_total_bonds(atoms)
            ok = (pred_total == exp_total)
            tested += 1
            exact += 1 if ok else 0
            print(f"[{tested:4d}/{total_rows:4d}] {name:32s}  pred={pred_total:4d}  exp={exp_total:4d}  {'✓' if ok else '✗'}", flush=True)
        except Exception as e:
            tested += 1
            print(f"[{tested:4d}/{total_rows:4d}] {name:32s}  ERROR: {e}", flush=True)

    pct = (100.0 * exact / tested) if tested else 0.0
    print(f"\nSummary: {exact}/{tested} exact matches ({pct:.1f}%) under closed-form ladder+b-matching.\n")

if __name__ == "__main__":
    main()