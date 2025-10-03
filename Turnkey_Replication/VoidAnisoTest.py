#!/usr/bin/env python3
# VoidAnisoTest.py
# Standalone anisotropy diagnostic using the same inputs as VoidDeepTests.py.
# - Reuses *_zobovoids.dat, *_zonevoids.dat, *_galzones.dat, and nsa_v1_0_1.fits
# - Computes cleaned anisotropy by fitting [1, cosθ, sinθ, cos2θ, sin2θ, cos3θ, sin3θ]
#   and reporting k=3 amplitude (sqrt(A3^2 + B3^2)) as Pk3^(1/2), and power as A3^2+B3^2.
# - Runs both on shell (0.8R<r<1.2R) and interior (r<=0.8R).
# - Includes randomized zone→void null control.

from __future__ import annotations
import os, sys, math, glob, csv, time, random, itertools
from typing import Dict, Tuple, Optional, List

# --- LOGGING SETUP ---
# File to log output to
LOG_FILENAME = "VoidAnisoTest.log" 
LOG_FILE = None
_original_print = print # Save a reference to the original print

def log_print(*args, **kwargs):
    """Prints to console and writes to the log file if open."""
    # Convert args to a single string
    text = ' '.join(map(str, args))

    # Determine end character (defaults to '\n')
    end_char = kwargs.get('end', '\n')

    # Print to console using the original print
    _original_print(text, **kwargs, file=sys.stdout)

    # Write to file
    if LOG_FILE:
        LOG_FILE.write(text + end_char)

print = log_print

# --- END LOGGING SETUP ---

# ---------- Config ----------
SHELL_MIN = 0.8
SHELL_MAX = 1.2
ANISO_MIN_SAMPLES_PER_VOID = 24     # per-void minimum to fit k=3
RAND_SEED = 42

# ---------- Numpy ----------
try:
    import numpy as np
except Exception:
    print("[fatal] numpy is required for this script.")
    sys.exit(1)

# ---------- NSA loader ----------
NSA_FITS_BASENAME = "nsa_v1_0_1.fits"
def load_nsa_positions_or_none(fits_path: str) -> Optional[Dict[int, Tuple[float,float,float]]]:
    if not os.path.exists(fits_path):
        print(f"[note] NSA FITS not found at {fits_path}")
        return None
    try:
        from fitsloader import load_nsa_positions
        return load_nsa_positions(fits_path)  # {NSAID: (RA,Dec,z)}
    except Exception as e:
        print(f"[note] Could not load NSA positions ({e}); continuing without NSA lookup.")
        return None

# ---------- Cosmo + geometry (aligned with VoidDeepTests.py) ----------
class Cosmo:
    def __init__(self, Om=0.315, Ol=0.685, h=0.674):
        self.Om = Om; self.Ol = Ol; self.Ok = 0.0; self.h = h
        self.H0 = 100.0 * h
        self.c  = 299792.458
    def E(self, z: float) -> float:
        return math.sqrt(self.Om*(1+z)**3 + self.Ok*(1+z)**2 + self.Ol)
    def Dc(self, z: float) -> float:
        if z <= 0: return 0.0
        n = 512; dz = z / n; s = 0.0
        for i in range(n+1):
            zi = i * dz
            w  = 4 if i % 2 == 1 else 2
            if i == 0 or i == n: w = 1
            s += w / self.E(zi)
        integ = s * dz / 3.0
        Dc_mpc = (self.c / self.H0) * integ
        return Dc_mpc / self.h  # h^-1 Mpc

def pick_cosmo_from_name(name: str) -> Cosmo:
    low = name.lower()
    if "wmap5" in low:   return Cosmo(Om=0.258, Ol=0.742, h=0.719)
    return Cosmo(Om=0.315, Ol=0.685, h=0.674)

def ra_dec_z_to_xyz(ra_deg: float, dec_deg: float, z: float, cosmo: Cosmo) -> Tuple[float,float,float]:
    ra  = math.radians(ra_deg) if abs(ra_deg) > 2*math.pi else ra_deg
    dec = math.radians(dec_deg) if abs(dec_deg) > 2*math.pi else dec_deg
    r = cosmo.Dc(z)
    return (r*math.cos(dec)*math.cos(ra), r*math.cos(dec)*math.sin(ra), r*math.sin(dec))

def dist3(a: Tuple[float,float,float], b: Tuple[float,float,float]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))

# ---------- IO helpers ----------
def tokenize(line: str) -> List[str]:
    return line.strip().split()

def iter_data_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            if line.lstrip().startswith(("#",";","//")): continue
            yield line

# ---------- Sniffers (aligned with VoidDeepTests.py) ----------
def sniff_voids_format(path: str) -> Dict[str,int]:
    examples = []
    for line in iter_data_lines(path):
        t = tokenize(line)
        if len(t) >= 5:
            examples.append(t)
            if len(examples) >= 40: break
    if not examples:
        raise RuntimeError(f"No parseable lines in {path}")

    num_cols = max(len(t) for t in examples)
    best = None
    for a,b,c,d in itertools.permutations(range(num_cols), 4):
        if not all(len(t) > max(a,b,c,d) for t in examples[:10]): continue
        try:
            ra  = [float(t[a]) for t in examples[:10]]
            dec = [float(t[b]) for t in examples[:10]]
            zz  = [float(t[c]) for t in examples[:10]]
            rr  = [float(t[d]) for t in examples[:10]]
        except: continue
        if all(0.0 <= x <= 360.0 for x in ra) and all(-90.0 <= x <= 90.0 for x in dec) and \
           all(-1e-3 <= x <= 3.0 for x in zz) and all(x > 0 for x in rr):
            best = ("ra","dec","z","r",(a,b,c,d))
            break
    if best is None:
        raise RuntimeError(f"Could not detect (ra,dec,z,R) in {path}")
    keys = best[:4]; idxs = best[4]
    return {keys[i]: idxs[i] for i in range(4)}

def read_voids(path: str, cosmo: Cosmo):
    m = sniff_voids_format(path)
    voids = {}         # key -> (center_xyz, R, z_void)
    centers = {}
    row_idx = -1
    for line in iter_data_lines(path):
        t = tokenize(line); row_idx += 1
        ra  = float(t[m["ra"]]); dec = float(t[m["dec"]]); zz = float(t[m["z"]]); R = float(t[m["r"]])
        ctr = ra_dec_z_to_xyz(ra, dec, zz, cosmo)
        voids[row_idx] = (ctr, R, zz)
        centers[row_idx] = ctr
    return voids, centers

def build_zone_to_void(path: str) -> Dict[int,int]:
    z2v = {}
    for line in iter_data_lines(path):
        t = tokenize(line)
        if len(t) < 2: continue
        try:
            z = int(t[0]); v = int(t[1])
        except: continue
        z2v[z] = v
    return z2v

def sniff_galzones(path: str) -> Dict[str,int]:
    examples = []
    for line in iter_data_lines(path):
        t = tokenize(line)
        if len(t) >= 2:
            examples.append(t)
            if len(examples) >= 50: break
    if not examples:
        raise RuntimeError(f"No parseable lines in {path}")
    num_cols = max(len(t) for t in examples)

    best_zone, zone_idx = -1e9, 0
    for ci in range(num_cols):
        vals = [t[ci] for t in examples]
        ints = sum(1 for v in vals if v.isdigit())
        uniq = len({v for v in vals if v.isdigit()})
        score = ints - 0.25*uniq
        if score > best_zone:
            best_zone, zone_idx = score, ci

    best_gal, gal_idx = -1e9, None
    for ci in range(num_cols):
        if ci == zone_idx: continue
        vals = [t[ci] for t in examples]
        ints = sum(1 for v in vals if v.isdigit())
        uniq = len({v for v in vals if v.isdigit()})
        score = ints + 0.5*uniq
        if score > best_gal:
            best_gal, gal_idx = score, ci
    return {"gal_id": gal_idx, "zone": zone_idx}

# ---------- Local angle around a void (tangent-plane) ----------
def void_plane_angle(ctr_xyz: Tuple[float,float,float], g_xyz: Tuple[float,float,float]) -> float:
    n = np.asarray(ctr_xyz, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        n = np.array([0.0,0.0,1.0], dtype=float)
    else:
        n = n / n_norm
    # build stable basis in plane ⟂ n
    zhat = np.array([0.0,0.0,1.0], dtype=float)
    e1 = np.cross(n, zhat)
    if np.linalg.norm(e1) < 1e-12:
        xhat = np.array([1.0,0.0,0.0], dtype=float)
        e1 = np.cross(n, xhat)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    p = np.asarray(g_xyz, dtype=float) - np.asarray(ctr_xyz, dtype=float)
    p_perp = p - np.dot(p, n) * n
    x = float(np.dot(p_perp, e1))
    y = float(np.dot(p_perp, e2))
    ang = math.atan2(y, x)
    if ang < 0: ang += 2*math.pi
    return ang

# ---------- Harmonic fit with k=3 amplitude ----------
def fit_k3_amplitude(theta_arr: np.ndarray, y_arr: np.ndarray, w_arr: Optional[np.ndarray]=None):
    """Fit [1, cosθ, sinθ, cos2θ, sin2θ, cos3θ, sin3θ] to y(θ) and return k=3 amplitude."""
    t = np.asarray(theta_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    if w_arr is None:
        w = np.ones_like(t)
    else:
        w = np.asarray(w_arr, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(w)
    t, y, w = t[mask], y[mask], w[mask]
    if t.size < 7:  # not enough dof
        return {"ok": False}

    X = np.column_stack([
        np.ones_like(t),
        np.cos(t), np.sin(t),
        np.cos(2*t), np.sin(2*t),
        np.cos(3*t), np.sin(3*t),
    ])
    sqrtw = np.sqrt(w)[:,None]
    Xw = X * sqrtw
    yw = y * np.sqrt(w)
    try:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except Exception:
        return {"ok": False}
    A3, B3 = float(beta[-2]), float(beta[-1])
    amp3 = math.hypot(A3, B3)
    power3 = A3*A3 + B3*B3
    return {"ok": True, "A3": A3, "B3": B3, "amp3": amp3, "power3": power3, "beta": beta}

# ---------- Catalog discovery ----------
def discover_catalogs(cwd: str):
    pairs = []
    for vpath in sorted(glob.glob(os.path.join(cwd, "*_zobovoids.dat"))):
        base = os.path.basename(vpath).replace("_zobovoids.dat","")
        gz = glob.glob(os.path.join(cwd, f"{base}_galzones.dat"))
        zv = glob.glob(os.path.join(cwd, f"{base}_zonevoids.dat"))
        if not gz or not zv:
            continue
        pairs.append((vpath, gz[0], zv[0], base))
    return pairs

# ---------- Main processing ----------
def process_catalog_anisotropy(voids_path: str, galzones_path: str, zonevoids_path: str,
                               nsa_pos: Dict[int, Tuple[float,float,float]]):
    label = os.path.basename(voids_path).replace("_zobovoids.dat","")
    cosmo = pick_cosmo_from_name(voids_path)
    t0 = time.time()

    voids, centers = read_voids(voids_path, cosmo)
    if not voids:
        print(f"[warn] No voids found in {voids_path}")
        return
    z2v = build_zone_to_void(zonevoids_path)
    gmap = sniff_galzones(galzones_path)
    if gmap["gal_id"] is None:
        print(f"[warn] Could not detect galaxy ID column in {galzones_path}")
        return

    # global accumulators (galaxy-level)
    th_shell, y_shell, w_shell, vid_shell = [], [], [], []
    th_int,   y_int,   w_int,   vid_int   = [], [], [], []

    # per-void accumulators
    per_void_shell = {vid: {"t": [], "y": []} for vid in voids}
    per_void_int   = {vid: {"t": [], "y": []} for vid in voids}

    # cache for null control (zone_id -> gxyz)
    gal_cache = []

    n_gal = 0
    used_shell = used_int = 0
    missing_nomap = 0
    missing_pos = 0

    for line in iter_data_lines(galzones_path):
        t = tokenize(line)
        try:
            zone_id = int(t[gmap["zone"]])
        except:
            n_gal += 1
            continue
        v_id = z2v.get(zone_id, None)
        if v_id is None or v_id < 0 or v_id not in voids:
            missing_nomap += 1; n_gal += 1
            continue
        try:
            gid = int(t[gmap["gal_id"]])
        except:
            missing_pos += 1; n_gal += 1
            continue

        pos = nsa_pos.get(gid)
        if pos is None:
            missing_pos += 1; n_gal += 1
            continue
        ra, dec, zz = pos
        gxyz = ra_dec_z_to_xyz(ra, dec, zz, cosmo)
        gal_cache.append((zone_id, gxyz))

        ctr, R, _ = voids[v_id]
        r = dist3(gxyz, ctr)
        u = r / R
        theta = void_plane_angle(ctr, gxyz)

        # interior & shell collections
        if r <= SHELL_MIN * R:
            th_int.append(theta); y_int.append(u); w_int.append(1.0); vid_int.append(v_id)
            per_void_int[v_id]["t"].append(theta); per_void_int[v_id]["y"].append(u)
            used_int += 1
        elif SHELL_MIN * R < r < SHELL_MAX * R:
            th_shell.append(theta); y_shell.append(u); w_shell.append(1.0); vid_shell.append(v_id)
            per_void_shell[v_id]["t"].append(theta); per_void_shell[v_id]["y"].append(u)
            used_shell += 1

        n_gal += 1

    # ---- Global fits (galaxy-level) ----
    shell_fit = fit_k3_amplitude(np.array(th_shell), np.array(y_shell), np.array(w_shell))
    int_fit   = fit_k3_amplitude(np.array(th_int),   np.array(y_int),   np.array(w_int))

    # ---- Per-void fits ----
    by_void_rows = []
    for vid in voids:
        row = [vid]
        # shell
        Ts = np.array(per_void_shell[vid]["t"], dtype=float)
        Ys = np.array(per_void_shell[vid]["y"], dtype=float)
        if Ts.size >= ANISO_MIN_SAMPLES_PER_VOID:
            sfit = fit_k3_amplitude(Ts, Ys)
            if sfit["ok"]:
                row += [Ts.size, sfit["amp3"], sfit["power3"]]
            else:
                row += [Ts.size, "", ""]
        else:
            row += [Ts.size, "", ""]
        # interior
        Ti = np.array(per_void_int[vid]["t"], dtype=float)
        Yi = np.array(per_void_int[vid]["y"], dtype=float)
        if Ti.size >= ANISO_MIN_SAMPLES_PER_VOID:
            ifit = fit_k3_amplitude(Ti, Yi)
            if ifit["ok"]:
                row += [Ti.size, ifit["amp3"], ifit["power3"]]
            else:
                row += [Ti.size, "", ""]
        else:
            row += [Ti.size, "", ""]
        by_void_rows.append(row)

    # ---- Null control: randomize zone→void map, recompute global k=3 ----
    random.seed(RAND_SEED)
    void_keys = list(voids.keys())
    z2v_keys = list(z2v.keys())
    rand_assign = { z: random.choice(void_keys) for z in z2v_keys }

    th_shell_null, y_shell_null = [], []
    th_int_null,   y_int_null   = [], []
    for zone_id, gxyz in gal_cache:
        v = rand_assign.get(zone_id, None)
        if v is None: continue
        ctr,R,_ = voids[v]
        r = dist3(gxyz, ctr)
        u = r / R
        theta = void_plane_angle(ctr, gxyz)
        if r <= SHELL_MIN * R:
            th_int_null.append(theta);   y_int_null.append(u)
        elif SHELL_MIN * R < r < SHELL_MAX * R:
            th_shell_null.append(theta); y_shell_null.append(u)

    shell_null_fit = fit_k3_amplitude(np.array(th_shell_null), np.array(y_shell_null))
    int_null_fit   = fit_k3_amplitude(np.array(th_int_null),   np.array(y_int_null))

    # ---- Outputs ----
    # Per-void CSV
    csv_path = f"aniso_by_void_{label}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["void_id",
                    "N_shell","k3_amp_shell","k3_power_shell",
                    "N_interior","k3_amp_interior","k3_power_interior"])
        for row in by_void_rows:
            w.writerow(row)

    # Summary
    sum_path = f"aniso_summary_{label}.txt"
    with open(sum_path, "w") as f:
        f.write(f"Catalog: {label}\n")
        f.write(f"Voids: {len(voids)}\n")
        f.write(f"Processed galaxy lines: {n_gal}\n")
        f.write(f"Used (shell): {used_shell}\n")
        f.write(f"Used (interior): {used_int}\n")
        f.write(f"Missing (nomap): {missing_nomap}\n")
        f.write(f"Missing (no NSA pos): {missing_pos}\n\n")

        f.write("Global anisotropy (cleaned; k=3 harmonic):\n")
        if shell_fit["ok"]:
            f.write(f"  shell:   k3_amp = {shell_fit['amp3']:.6f}  k3_power = {shell_fit['power3']:.6f}\n")
        else:
            f.write("  shell:   insufficient samples\n")
        if int_fit["ok"]:
            f.write(f"  interior: k3_amp = {int_fit['amp3']:.6f}  k3_power = {int_fit['power3']:.6f}\n")
        else:
            f.write("  interior: insufficient samples\n")

        f.write("\nNull / Control (random zone→void remap):\n")
        if shell_null_fit["ok"]:
            f.write(f"  shell(null):   k3_amp = {shell_null_fit['amp3']:.6f}  k3_power = {shell_null_fit['power3']:.6f}\n")
        else:
            f.write("  shell(null):   insufficient samples\n")
        if int_null_fit["ok"]:
            f.write(f"  interior(null): k3_amp = {int_null_fit['amp3']:.6f}  k3_power = {int_null_fit['power3']:.6f}\n")
        else:
            f.write("  interior(null): insufficient samples\n")

    dt = time.time()-t0
    print(f"[done] {label}:")
    print(f"  per-void → {csv_path}")
    print(f"  summary  → {sum_path}")
    print(f"  elapsed: {dt:.1f}s\n")

def main():
    global LOG_FILE # Access the global file handle
    cwd = os.getcwd()

    # Open the log file
    try:
        with open(LOG_FILENAME, 'w') as f:
            LOG_FILE = f # Set the global file handle

            catalogs = discover_catalogs(cwd)
            if not catalogs:
                print("No *_zobovoids.dat + companions found.")
                sys.exit(1)
            nsa = load_nsa_positions_or_none(os.path.join(cwd, NSA_FITS_BASENAME))
            if not nsa:
                print("[fatal] NSA positions missing; cannot proceed.")
                sys.exit(1)
            print(f"[NSA] positions loaded: {len(nsa):,}")

            for vpath, gpath, zpath, base in catalogs:
                print("="*80)
                print(f"Catalog: {os.path.basename(vpath)}")
                print(f"  galzones:  {os.path.basename(gpath)}")
                print(f"  zonevoids: {os.path.basename(zpath)}")
                try:
                    process_catalog_anisotropy(vpath, gpath, zpath, nsa)
                except Exception as e:
                    print(f"[error] {base}: {e}")

    except Exception as e:
        # If logging itself fails, print to original console print
        _original_print(f"Fatal error during main execution (or file open): {e}", file=sys.stderr)
    finally:
        LOG_FILE = None # Clear the global file handle when done

if __name__ == "__main__":
    main()