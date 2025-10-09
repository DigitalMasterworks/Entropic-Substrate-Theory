#!/usr/bin/env python3
# ladder_to_bonds.py — Equations-only v10
# - Logistic ladder capacity s(ν): universal, no tables (tames H–H / O–O).
# - Triangle (3-cycle) penalty: discourages spurious light-light-light loops.
# - Dual-peak Δν kernel with slight heavy–light preference (W1 > W0).
# - Keeps: pre-bond Gaussian, Steiner 120°, global sparsity, per-pair (b-1)^2.
#
# ZERO chemistry priors. Only atomic masses -> ν, g, and universal constants.

import math, random
from typing import Dict, List, Tuple, Set
import numpy as np
from itertools import combinations

# ---------- Anchors & ladder ----------
MEV_PER_U   = 931.49410242
MASS_E_MEV  = 0.51099895000
MASS_MU_MEV = 105.6583755
MU_E        = MASS_MU_MEV / MASS_E_MEV
g           = MU_E ** (1.0 / 13.0)

AMU = {
    "H": 1.00784, "C": 12.0107, "N": 14.0067, "O": 15.999,
    "F": 18.998403163, "Cl": 35.45, "S": 32.065, "P": 30.973761998,
}

def rung_from_atomic_mass_u(amu: float) -> float:
    mZ_MeV = amu * MEV_PER_U
    return 3.0 + 13.0 * math.log(mZ_MeV / MASS_E_MEV) / math.log(MU_E)

# ---------- Universal constants ----------
# Dual-peak resonance (Δν≈0 and Δν≈6.5)
SIGMA0 = 0.9
SIGMA1 = 1.2
DELTA1 = 6.5
W0     = 0.8   # like-likes-like
W1     = 1.2   # heavy–light (slight preference)

NU0        = 10.0
D_STAR     = 1.40

# Energetics (encourage σ to click; doubles/triples still pay)
KAPPA_R    = 1.25
ETA_BIND   = 1.55
Q_MULT     = 0.30

KAPPA_ANG  = 0.28
KAPPA_REP  = 0.03
REP_POW    = 6.0
LAMBDA_S   = 0.020

KAPPA_ORD  = 0.75

# Logistic ladder capacity (equations-only; no tables)
S_MAX      = 6.0     # asymptotic capacity
NU_CAP     = 24.0    # logistic center
K_CAP      = 0.70    # logistic slope
KAPPA_CAP  = 0.20    # capacity penalty weight

# Small-ring and aromatic terms
KAPPA_TRI  = 0.30    # triangle (3-cycle) penalty (universal)
KAPPA_6      = 0.05  # aromatic smoothing on emergent 6-cycles
AROM_VAR_EPS = 0.06

# Pre-bond Gaussian (acts on all pairs, even when b_ij == 0)
ETA_PRE   = 1.20
S_LEN     = 0.75

# Seeding proximity gate (equations-only)
TAU_LEN   = 0.70

# Optimizer settings
SEED=7
GEOM_ITERS=250
GEOM_LR_INIT=0.03
GEOM_LR_DECAY=0.995
GEOM_GRAD_TOL=1e-4
BOND_SWEEPS=40
MAX_ORDER=3
RESTARTS=4

random.seed(SEED); np.random.seed(SEED)

# ---------- Molecules (compositions only) ----------
MOLECULES: Dict[str, List[str]] = {
    "water":            ["O","H","H"],
    "ammonia":          ["N","H","H","H"],
    "methane":          ["C","H","H","H","H"],
    "ethane":           ["C","C","H","H","H","H","H","H"],
    "ethene":           ["C","C","H","H","H","H"],
    "acetylene":        ["C","C","H","H"],
    "carbon_dioxide":   ["O","C","O"],
    "hydrogen_cyanide": ["H","C","N"],
    "benzene":          ["C","C","C","C","C","C","H","H","H","H","H","H"],
    "ethanol":          ["C","C","O","H","H","H","H","H","H"],
}

# ---------- Helpers ----------
def rung_resonance_dual(nu_i: float, nu_j: float) -> float:
    d = abs(nu_i - nu_j)
    p0 = math.exp(-(d / SIGMA0)**2)                           # Δν≈0
    p1 = math.exp(-((d - DELTA1)**2) / (2.0 * SIGMA1**2))     # Δν≈6.5
    return W0*p0 + W1*p1

def preferred_length(nu_i: float, nu_j: float) -> float:
    return D_STAR * (g ** ((nu_i + nu_j - 2.0 * NU0) / 13.0))

def phi(b: int) -> float:
    return math.sqrt(float(b))

def neighbors(bmat: np.ndarray, i: int) -> List[int]:
    return [j for j in range(bmat.shape[0]) if j != i and bmat[min(i,j), max(i,j)] > 0]

def angle_energy_at(i: int, pos: np.ndarray, nbrs: List[int], bmat: np.ndarray) -> float:
    e = 0.0
    if len(nbrs) < 2: return 0.0
    xi = pos[i]
    for j, k in combinations(nbrs, 2):
        if bmat[i,j] <= 0 or bmat[i,k] <= 0: continue
        v1 = pos[j] - xi; v2 = pos[k] - xi
        n1 = np.linalg.norm(v1) + 1e-12; n2 = np.linalg.norm(v2) + 1e-12
        cos_th = float(np.dot(v1, v2) / (n1 * n2))
        e += (cos_th + 0.5)**2 * phi(bmat[i,j]) * phi(bmat[i,k])
    return e

def six_cycles(edges: Set[Tuple[int,int]], n: int) -> List[List[int]]:
    adj = {i: set() for i in range(n)}
    for u,v in edges: adj[u].add(v); adj[v].add(u)
    cycles = set()
    def dfs(path, start):
        if len(path) > 6: return
        u = path[-1]
        for v in adj[u]:
            if v == start and len(path) == 6:
                cyc = path[:]
                m = min(cyc); mi = cyc.index(m)
                r1 = tuple(cyc[mi:] + cyc[:mi]); r2 = tuple(reversed(r1))
                cycles.add(min(r1, r2))
            elif v not in path and len(path) < 6:
                dfs(path + [v], start)
    for s in range(n): dfs([s], s)
    return [list(c) for c in cycles]

def triangle_count(edges: Set[Tuple[int,int]], n: int) -> int:
    adj = {i:set() for i in range(n)}
    for u,v in edges:
        adj[u].add(v); adj[v].add(u)
    count = 0
    for i in range(n):
        nbrs = sorted(adj[i])
        for a_idx in range(len(nbrs)):
            a = nbrs[a_idx]
            for b in nbrs[a_idx+1:]:
                if b in adj[a]:  # triangle i-a-b
                    count += 1
    return count
    
def unsup_score(pos: np.ndarray, bmat: np.ndarray, nu: np.ndarray,
                d_pref: np.ndarray, R: np.ndarray | None) -> float:
    """
    Higher is better. Uses only ν, g, R (or auto-build), d_pref, geometry.
    Rewards: resonant, length-matched edges; 120° junctions; equalized 6-cycles.
    Penalizes: triangles; light-light edges; capacity strain; excess total order.
    """
    n = bmat.shape[0]
    # --- make sure R is valid (build locally if None or wrong shape) ---
    if R is None or R.ndim != 2 or R.shape[0] != n or R.shape[1] != n:
        Rloc = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i+1, n):
                Rloc[i, j] = Rloc[j, i] = rung_resonance_dual(nu[i], nu[j])
        R = Rloc

    score = 0.0
    light_cut = 23.5  # ν-only “light” threshold

    # Edge rewards / penalties
    for i in range(n):
        for j in range(i+1, n):
            bij = int(bmat[i, j])
            if bij <= 0:
                continue
            lij  = np.linalg.norm(pos[i] - pos[j]) + 1e-12
            near = math.exp(-((lij - d_pref[i, j])**2) / (2.0 * (0.45**2)))
            score += (1.0 + 0.2*(bij-1)) * R[i, j] * near
            if nu[i] < light_cut and nu[j] < light_cut:
                score -= 0.6

    # Junction angle reward (Steiner 120°)
    def _phi(b): return math.sqrt(float(b))
    for i in range(n):
        nbrs = [k for k in range(n) if k != i and bmat[min(i,k), max(i,k)] > 0]
        if len(nbrs) < 2:
            continue
        xi = pos[i]; acc = 0.0; cnt = 0
        for a, b in combinations(nbrs, 2):
            if bmat[i, a] <= 0 or bmat[i, b] <= 0:
                continue
            v1 = pos[a] - xi; v2 = pos[b] - xi
            n1 = np.linalg.norm(v1) + 1e-12; n2 = np.linalg.norm(v2) + 1e-12
            cos_th = float(np.dot(v1, v2) / (n1 * n2))
            acc += 1.0 - ((cos_th + 0.5) ** 2) * _phi(bmat[i, a]) * _phi(bmat[i, b])
            cnt += 1
        if cnt:
            score += 0.1 * acc / cnt

    # Triangle penalty
    edges = {(i, j) for i in range(n) for j in range(i+1, n) if bmat[i, j] >= 1}
    def _tri_count(E, n_):
        adj = {x: set() for x in range(n_)}
        for u, v in E: adj[u].add(v); adj[v].add(u)
        c = 0
        for u in range(n_):
            nbr = sorted(adj[u])
            for a_i in range(len(nbr)):
                a = nbr[a_i]
                for b in nbr[a_i+1:]:
                    if b in adj[a]: c += 1
        return c
    score -= 0.6 * _tri_count(edges, n)

    # 6-cycle equalization reward
    def _six_cycles(E, n_):
        adj = {x: set() for x in range(n_)}
        for u, v in E: adj[u].add(v); adj[v].add(u)
        C = set()
        def dfs(path, s):
            if len(path) > 6: return
            u = path[-1]
            for v in adj[u]:
                if v == s and len(path) == 6:
                    cyc = path[:]
                    m = min(cyc); mi = cyc.index(m)
                    r1 = tuple(cyc[mi:]+cyc[:mi]); r2 = tuple(reversed(r1))
                    C.add(min(r1, r2))
                elif v not in path and len(path) < 6:
                    dfs(path+[v], s)
        for s in range(n_): dfs([s], s)
        return [list(c) for c in C]

    C6 = _six_cycles(edges, n)
    if C6:
        for cyc in C6:
            devs = []
            for t in range(6):
                a, b = cyc[t], cyc[(t+1) % 6]
                aa, bb = (a, b) if a < b else (b, a)
                if bmat[aa, bb] <= 0:
                    devs.append(1.0); continue
                l = np.linalg.norm(pos[a] - pos[b]) + 1e-12
                devs.append(abs(l - d_pref[aa, bb]))
            m = sum(devs)/6.0 + 1e-12
            var = sum((x - m)**2 for x in devs)/6.0
            score += max(0.0, 0.5 - var)

    # Capacity strain (via logistic s(ν)) + tiny sparsity
    def s_of_nu(v): return S_MAX / (1.0 + math.exp(-K_CAP * (v - NU_CAP)))
    for i in range(n):
        Si  = sum(int(bmat[min(i,j), max(i,j)]) for j in range(n) if j != i)
        cap = s_of_nu(nu[i]) + 1e-12
        score -= 0.05 * (Si / cap)**2

    tot_order = sum(int(bmat[i, j]) for i in range(n) for j in range(i+1, n))
    score -= 0.01 * tot_order

    return float(score)

def autotune_constants(molecules: Dict[str, List[str]],
                         trials: int = 160,
                         restarts: int = 3,
                         rng_seed: int = 123) -> dict:
      """
      Random-search autotuner over *universal* constants.
      Equations-only: evaluates unsup_score on your molecule set.
      Returns a dict of best constants you can paste back up top.
      """
      rng = np.random.RandomState(rng_seed)

      # search ranges (all universal; equations-only)
      ranges = {
          "W0":        (0.5, 1.2),
          "W1":        (0.9, 1.6),
          "ETA_BIND":  (0.9, 1.8),
          "KAPPA_R":   (0.9, 1.8),
          "LAMBDA_S":  (0.005, 0.06),
          "KAPPA_ORD": (0.4, 1.2),
          "KAPPA_ANG": (0.18, 0.40),
          "ETA_PRE":   (0.6, 1.6),
          "S_LEN":     (0.40, 0.90),
          "TAU_LEN":   (0.30, 0.90),
          "KAPPA_TRI": (0.10, 0.40),
          "S_MAX":     (3.0, 8.0),
          "NU_CAP":    (22.0, 27.0),
          "K_CAP":     (0.30, 1.00),
          "KAPPA_CAP": (0.12, 0.30),
      }

      # freeze baseline that aren’t the focus
      fixed = {
          "SIGMA0": SIGMA0, "SIGMA1": SIGMA1, "DELTA1": DELTA1,
          "NU0": NU0, "D_STAR": D_STAR, "Q_MULT": Q_MULT,
          "KAPPA_REP": KAPPA_REP, "REP_POW": REP_POW, "KAPPA_6": KAPPA_6,
      }

      def sample():
          p = {}
          for k, (a, b) in ranges.items():
              val = float(rng.uniform(a, b))
              # ints where appropriate
              if k in ("S_MAX",):
                  val = float(val)
              p[k] = val
          return p

      def evaluate(params: dict) -> float:
          globs = globals()
          saved = {}
          keys = list(ranges.keys())
          try:
              for k in keys:
                  saved[k] = globs[k]
                  globs[k] = params[k]

              total = 0.0
              for name, elems in molecules.items():
                  best = None
                  for r in range(restarts):
                      E, pos, bmat, nu, d_pref = solve_once(elems, seed_shift=r+1)
                      # build Rloc for the current ν with the current dual-peak kernel
                      n = len(elems)
                      Rloc = np.zeros((n, n), dtype=float)
                      for i in range(n):
                          for j in range(i+1, n):
                              Rloc[i, j] = Rloc[j, i] = rung_resonance_dual(nu[i], nu[j])
                      s = unsup_score(pos, bmat, nu, d_pref, Rloc)
                      best = s if best is None else max(best, s)
                  total += best
              return total
          finally:
              for k in keys:
                  globs[k] = saved[k]

      best_score = -1e18
      best_params = None

      for t in range(trials):
          cand = sample()
          sc = evaluate(cand)
          if sc > best_score:
              best_score, best_params = sc, cand

      # return a single dict merging fixed + best
      out = dict(fixed); out.update(best_params)
      return out
        
def cycle_variance(cyc: List[int], pos: np.ndarray, bmat: np.ndarray, d_pref: np.ndarray) -> float:
    m = len(cyc); devs = []
    for t in range(m):
        i = cyc[t]; j = cyc[(t+1) % m]
        a, b = (i, j) if i < j else (j, i)
        if bmat[a, b] <= 0: return 1.0
        lij = np.linalg.norm(pos[i] - pos[j]) + 1e-12
        devs.append(abs(lij - d_pref[a, b]))
    devs = np.array(devs, dtype=float)
    return float(np.var(devs) / (np.mean(devs) + 1e-12))

def apply_constants(params: dict):
    globs = globals()
    for k, v in params.items():
        if k in globs:
            globs[k] = v

def AUTOTUNE():
              best = autotune_constants(MOLECULES, trials=200, restarts=2, rng_seed=314159)
              print("\n=== AUTOTUNED CONSTANTS (paste into your constants block) ===")
              for k in sorted(best.keys()):
                  print(f"{k} = {best[k]:.6f}")
              print("=== END ===\n")
              apply_constants(best)
                        
# Logistic ladder capacity (used both for seeding & site penalty)
def capacity_scale_logistic(nu_i: float) -> float:
    # s(ν) = S_MAX / (1 + exp(-K_CAP * (ν - NU_CAP)))
    return S_MAX / (1.0 + math.exp(-K_CAP * (nu_i - NU_CAP)))

# ---------- Equations-only seeding ----------
def seed_bonds_equations_only(pos: np.ndarray, nu: np.ndarray,
                              d_pref: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Equations-only σ seeding:
      - capacity_i = round( S_MAX / (1 + exp(-K_CAP*(nu_i - NU_CAP))) )
      - skip ultra-light pairs: min(nu_i, nu_j) >= NU_LIGHT (ν threshold)
      - proximity gate: |r_ij - d_ij| <= TAU_LEN
      - score = (W0/W1 dual-peak) * exp(-|r_ij - d_ij| / w)
      - greedy under per-site capacity
    """
    n = pos.shape[0]
    b0 = np.zeros((n, n), dtype=int)

    # logistic ladder capacity (may be 0 on very light rungs; heavy gets more)
    cap = np.rint([
        (S_MAX / (1.0 + math.exp(-K_CAP * (v - NU_CAP))))
        for v in nu
    ]).astype(int)
    cap = np.maximum(0, cap)

    # universal ν-light cutoff (equations-only; no element tables)
    NU_LIGHT = 23.5  # below this, sites are "light" and shouldn't seed together

    pairs = []
    w = max(0.30, 0.5 * S_LEN)
    for i in range(n):
        for j in range(i+1, n):
            # forbid light-light seeding (e.g., H–H) using ν only
            if min(nu[i], nu[j]) < NU_LIGHT and max(nu[i], nu[j]) < NU_LIGHT:
                continue
            rij = np.linalg.norm(pos[i] - pos[j]) + 1e-12
            if abs(rij - d_pref[i, j]) > TAU_LEN:
                continue
            proximity = math.exp(-abs(rij - d_pref[i, j]) / w)
            score = float(R[i, j] * proximity)
            pairs.append((score, i, j))

    pairs.sort(reverse=True)

    rem = cap.copy()
    for score, i, j in pairs:
        if rem[i] <= 0 or rem[j] <= 0:
            continue
        b0[i, j] = 1     # seed exactly one σ lane
        rem[i] -= 1
        rem[j] -= 1

    return b0
    
# ---------- Energy ----------
def total_energy(pos: np.ndarray, bmat: np.ndarray, nu: np.ndarray,
                 d_pref: np.ndarray, R: np.ndarray) -> float:
    n = pos.shape[0]; E = 0.0
    # (A) bonded terms
    for i in range(n):
        for j in range(i+1, n):
            bij = bmat[i, j]
            if bij > 0:
                lij = np.linalg.norm(pos[i] - pos[j]) + 1e-12
                E += KAPPA_R * bij * (lij - d_pref[i, j])**2
                E -= ETA_BIND * R[i, j] * (1.0 + Q_MULT * (bij - 1.0))
                E += LAMBDA_S * bij
                E += KAPPA_ORD * (bij - 1.0)**2
    # (B) Steiner 120°
    for i in range(n):
        nbr = neighbors(bmat, i)
        if len(nbr) >= 2:
            E += KAPPA_ANG * angle_energy_at(i, pos, nbr, bmat)
    # (C) repulsion (all pairs)
    for i in range(n):
        for j in range(i+1, n):
            rvec = pos[i] - pos[j]; dist = np.linalg.norm(rvec) + 1e-9
            E += KAPPA_REP / (dist ** REP_POW)
    # (D) site capacity penalty, logistic s(ν)
    for i in range(n):
        Si = sum(int(bmat[min(i,j), max(i,j)]) for j in range(n) if j != i)
        cap = capacity_scale_logistic(nu[i]) + 1e-12
        E += KAPPA_CAP * (Si / cap)**2
    # (E) pre-bond Gaussian (all pairs)
    for i in range(n):
        for j in range(i+1, n):
            lij = np.linalg.norm(pos[i] - pos[j]) + 1e-12
            gauss = math.exp(- ((lij - d_pref[i, j])**2) / (2.0 * S_LEN**2))
            E -= ETA_PRE * R[i, j] * gauss
    # (F) triangle penalty (3-cycles)
    edges = {(i,j) for i in range(n) for j in range(i+1, n) if bmat[i, j] >= 1}
    E += KAPPA_TRI * triangle_count(edges, n)
    # (G) aromatic smoothing on emergent 6-cycles
    if KAPPA_6 > 0.0:
        for cyc in six_cycles(edges, n):
            E += KAPPA_6 * (cycle_variance(cyc, pos, bmat, d_pref) ** 2)
    return float(E)

# ---------- Optimization ----------
def geom_descent(pos: np.ndarray, bmat: np.ndarray, d_pref: np.ndarray, R: np.ndarray) -> np.ndarray:
    n = pos.shape[0]; lr = GEOM_LR_INIT
    for _ in range(GEOM_ITERS):
        grad = np.zeros_like(pos)
        # length fit (bonded)
        for i in range(n):
            for j in range(i+1, n):
                bij = bmat[i, j]
                if bij <= 0: continue
                rij = pos[i] - pos[j]; dist = np.linalg.norm(rij) + 1e-12
                coeff = 2.0 * KAPPA_R * bij * (dist - d_pref[i, j]) / dist
                gvec = coeff * rij
                grad[i] += gvec; grad[j] -= gvec
        # repulsion (all)
        for i in range(n):
            for j in range(i+1, n):
                rij = pos[i] - pos[j]; dist = np.linalg.norm(rij) + 1e-18
                coeff = -KAPPA_REP * REP_POW / (dist ** (REP_POW + 2))
                gvec = coeff * rij
                grad[i] += gvec; grad[j] -= gvec
        # pre-bond attraction gradient (all pairs)
        for i in range(n):
            for j in range(i+1, n):
                rij = pos[i] - pos[j]; dist = np.linalg.norm(rij) + 1e-12
                gauss = math.exp(- ((dist - d_pref[i, j])**2) / (2.0 * S_LEN**2))
                coeff = ETA_PRE * R[i, j] * gauss * ((dist - d_pref[i, j]) / (S_LEN**2 * dist))
                gvec = -coeff * rij
                grad[i] += gvec; grad[j] -= gvec
        pos -= lr * grad
        if np.linalg.norm(grad) / (n + 1e-12) < GEOM_GRAD_TOL:
            break
        lr *= GEOM_LR_DECAY
    return pos

def discrete_best_improvement(pos: np.ndarray, bmat: np.ndarray, nu: np.ndarray,
                              d_pref: np.ndarray, R: np.ndarray) -> bool:
    n = pos.shape[0]; E0 = total_energy(pos, bmat, nu, d_pref, R)
    best_delta = 0.0; best_pair = None; best_val = None
    for i in range(n):
        for j in range(i+1, n):
            old = int(bmat[i, j])
            for nij in (0,1,2,3):
                if nij == old: continue
                bmat[i, j] = nij
                E1 = total_energy(pos, bmat, nu, d_pref, R)
                dE = E0 - E1
                if dE > best_delta + 1e-12:
                    best_delta = dE; best_pair = (i, j); best_val = nij
            bmat[i, j] = old
    if best_pair is not None:
        i, j = best_pair
        bmat[i, j] = int(best_val)
        return True
    return False

def solve_once(elems: List[str], seed_shift: int = 0):
    n = len(elems)
    nu = np.array([rung_from_atomic_mass_u(AMU[z]) for z in elems], dtype=float)
    d_pref = np.zeros((n, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d_pref[i, j] = d_pref[j, i] = preferred_length(nu[i], nu[j])
            R[i, j] = R[j, i] = rung_resonance_dual(nu[i], nu[j])

    # init positions on circle with small noise (different angle per restart)
    theta = np.linspace(0, 2*np.pi, n, endpoint=False) + (seed_shift * 0.123)
    radius = D_STAR * 1.6
    rng = np.random.RandomState(SEED + 13*seed_shift)
    pos = np.stack([radius*np.cos(theta), radius*np.sin(theta)], axis=1).astype(float)
    pos += 0.05 * rng.randn(n, 2)

    # equations-only seeding of σ lanes
    bmat = seed_bonds_equations_only(pos, nu, d_pref, R)

    # alternate geometry + best-improvement discrete flips
    for _ in range(BOND_SWEEPS):
        pos = geom_descent(pos, bmat, d_pref, R)
        if not discrete_best_improvement(pos, bmat, nu, d_pref, R):
            pos = geom_descent(pos, bmat, d_pref, R)
            break

    E = total_energy(pos, bmat, nu, d_pref, R)
    return E, pos, bmat, nu, d_pref

def solve_molecule(name: str, elems: List[str]):
    best = None
    for r in range(RESTARTS):
        E, pos, bmat, nu, d_pref = solve_once(elems, seed_shift=r+1)
        if (best is None) or (E < best[0]):
            best = (E, pos, bmat, nu, d_pref)
    return best[1], best[2], best[3], best[4]

# ---------- Reporting ----------
def sigma_pi(bmat: np.ndarray) -> Tuple[int, int]:
    n = bmat.shape[0]; s = 0; p = 0
    for i in range(n):
        for j in range(i+1, n):
            bij = int(bmat[i, j])
            if bij >= 1:
                s += 1
                p += max(0, bij - 1)
    return s, p

def edges_list(bmat: np.ndarray) -> List[Tuple[int,int,int]]:
    n = bmat.shape[0]; out = []
    for i in range(n):
        for j in range(i+1, n):
            bij = int(bmat[i, j])
            if bij > 0:
                out.append((i, j, bij))
    return out

def aromatic_info(pos: np.ndarray, bmat: np.ndarray, d_pref: np.ndarray):
    n = bmat.shape[0]
    edges = {(i,j) for i in range(n) for j in range(i+1, n) if bmat[i, j] >= 1}
    cycles = six_cycles(edges, n); items = []
    for cyc in cycles:
        var = cycle_variance(cyc, pos, bmat, d_pref)
        items.append((cyc, var))
    return items

# ---------- Main ----------
def main():
    print(f"g = (mu/e)^(1/13) = {g:.9f}   (mu/e ≈ {MU_E:.6f})")
    print("Universal constants:",
          f"SIGMA0={SIGMA0}, SIGMA1={SIGMA1}, DELTA1={DELTA1}, NU0={NU0}, D*={D_STAR},",
          f"KAPPA_R={KAPPA_R}, ETA_BIND={ETA_BIND}, Q_MULT={Q_MULT}, KAPPA_ANG={KAPPA_ANG},",
          f"KAPPA_REP={KAPPA_REP}, REP_POW={REP_POW}, LAMBDA_S={LAMBDA_S},",
          f"KAPPA_ORD={KAPPA_ORD}, KAPPA_CAP={KAPPA_CAP}, S_MAX={S_MAX}, NU_CAP={NU_CAP}, K_CAP={K_CAP},",
          f"ETA_PRE={ETA_PRE}, S_LEN={S_LEN}, TAU_LEN={TAU_LEN}, KAPPA_TRI={KAPPA_TRI}, KAPPA_6={KAPPA_6}, RESTARTS={RESTARTS}")
    print()
    for name, elems in MOLECULES.items():
        pos, bmat, nu, d_pref = solve_molecule(name, elems)
        s, p = sigma_pi(bmat); edges = edges_list(bmat)
        print(f"== {name} ==")
        print("atoms:", elems)
        print("σ-bonds:", s, "   π-bonds:", p)
        if edges:
            print("edges (i,j,order):")
            for (i, j, o) in sorted(edges):
                print(f"  ({i:>2},{j:>2})  order={o}")
        else:
            print("  (no edges)")
        if KAPPA_6 > 0.0:
            info = aromatic_info(pos, bmat, d_pref)
            if info:
                print("emergent 6-cycles (nodes) with equalization variance:")
                for cyc, var in info:
                    tag = " (equalized)" if var <= AROM_VAR_EPS else ""
                    print(f"  {cyc}  var={var:.4f}{tag}")
        print()

if __name__ == "__main__":
    AUTOTUNE()
    main()