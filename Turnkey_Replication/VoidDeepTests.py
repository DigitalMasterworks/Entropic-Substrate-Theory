#!/usr/bin/env python3
# VoidDeepTests.py
# Deep tests on SDSS/NSA void catalogs:
# - Stacked radial profiles (wall sharpness) with volume correction
# - Robustness splits by void size and redshift
# - Anisotropy dipole in shell (and interior)
# - Null control via randomized zone→void mapping
#
# Expects:
#   - *_zobovoids.dat (RA,DEC,z,R)  [REVOLVER works; VIDE may map -1; will skip]
#   - *_zonevoids.dat (zone_id, void_id)
#   - *_galzones.dat  (gal_id, zone_id)
#   - nsa.v1_0_1.fits  (NSA catalog) + fitsloader.load_nsa_positions
#
# Outputs (per catalog label):
#   - deep_stats_by_void_<label>.csv
#   - deep_profiles_stacked_<label>.csv
#   - deep_profiles_by_sizequartile_<label>.csv
#   - deep_summary_<label>.txt

from __future__ import annotations
import os, sys, math, glob, csv, time, random, itertools
from typing import Dict, Tuple, Optional, List

# --- LOGGING SETUP ---
# File to log output to
LOG_FILENAME = "VoidDeepTests.log"
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

# ---------- Optional numpy for speed ----------
try:
    import numpy as np
except Exception:
    np = None

# ---------- NSA loader ----------
NSA_FITS_BASENAME = "nsa.v1_0_1.fits"
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

# ---------- Cosmo + geometry ----------
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
    if np is not None:
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    dx=a[0]-b[0]; dy=a[1]-b[1]; dz=a[2]-b[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

# ---------- IO helpers ----------
def tokenize(line: str) -> List[str]:
    return line.strip().split()

def iter_data_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            if line.lstrip().startswith(("#",";","//")): continue
            yield line

# ---------- Sniffers ----------
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
    # choose columns that look like ra,dec,z,R
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
    void_radii = []    # for quantiles
    void_redshifts = []
    centers = {}
    row_idx = -1
    for line in iter_data_lines(path):
        t = tokenize(line); row_idx += 1
        ra  = float(t[m["ra"]]); dec = float(t[m["dec"]]); zz = float(t[m["z"]]); R = float(t[m["r"]])
        ctr = ra_dec_z_to_xyz(ra, dec, zz, cosmo)
        voids[row_idx] = (ctr, R, zz)
        centers[row_idx] = ctr
        void_radii.append(R)
        void_redshifts.append(zz)
    return voids, centers, void_radii, void_redshifts

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
    # detect columns: galaxy id and zone id
    examples = []
    for line in iter_data_lines(path):
        t = tokenize(line)
        if len(t) >= 2:
            examples.append(t)
            if len(examples) >= 50: break
    if not examples:
        raise RuntimeError(f"No parseable lines in {path}")
    num_cols = max(len(t) for t in examples)
    # zone column: many ints, low cardinality
    best_zone, zone_idx = -1e9, 0
    for ci in range(num_cols):
        vals = [t[ci] for t in examples]
        ints = sum(1 for v in vals if v.isdigit())
        uniq = len({v for v in vals if v.isdigit()})
        score = ints - 0.25*uniq
        if score > best_zone:
            best_zone, zone_idx = score, ci
    # galaxy id column: many ints, high cardinality
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

# ---------- Analysis ----------
SHELL_MIN = 0.8
SHELL_MAX = 1.2
RMAX_NORM = 1.5        # max r/R for profiles
NBINS = 60             # radial profile bins
RAND_SEED = 42

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

def summarize_bins(vals: List[float], nbins: int=4):
    # return (edges, bin_ids) for quantiles
    if not vals: return [], {}
    arr = sorted(vals)
    edges = [arr[int(len(arr)*k/nbins)] for k in range(nbins)] + [arr[-1]+1e-9]
    # build map: value -> bin index
    return edges

def which_bin(x: float, edges: List[float]) -> int:
    # edges like [q0, q1, q2, q3, max]
    for i in range(len(edges)-1):
        if edges[i] <= x < edges[i+1]:
            return i
    return len(edges)-2

def process_catalog(voids_path: str, galzones_path: str, zonevoids_path: str,
                    nsa_pos: Dict[int, Tuple[float,float,float]]):
    label = os.path.basename(voids_path).replace("_zobovoids.dat","")
    cosmo = pick_cosmo_from_name(voids_path)
    t0 = time.time()

    # Load voids & mappings
    voids, centers, void_radii, void_redshifts = read_voids(voids_path, cosmo)
    z2v = build_zone_to_void(zonevoids_path)
    gmap = sniff_galzones(galzones_path)
    if gmap["gal_id"] is None:
        raise RuntimeError("Could not detect galaxy ID column in galzones.")

    # Pre-allocations
    n_voids = len(voids)
    bin_edges = np.linspace(0.0, RMAX_NORM, NBINS+1) if np is not None else [i*RMAX_NORM/NBINS for i in range(NBINS+1)]
    per_void_hist = {vid: [0]*NBINS for vid in voids}
    per_void_hist_vol = [ ( (bin_edges[i+1]-bin_edges[i]) * ((bin_edges[i+1]+bin_edges[i]+1e-12)/2.0)**2 ) for i in range(NBINS) ] if np is None \
                        else (np.diff(bin_edges) * ((bin_edges[:-1]+bin_edges[1:]+1e-12)/2.0)**2)

    counts_interior = {vid:0 for vid in voids}
    counts_shell    = {vid:0 for vid in voids}
    sum_inward      = {vid:0.0 for vid in voids}
    n_inward        = {vid:0 for vid in voids}
    # anisotropy dipole vector (shell & interior)
    dip_shell_vec   = {vid:[0.0,0.0,0.0] for vid in voids}
    dip_shell_N     = {vid:0 for vid in voids}
    dip_int_vec     = {vid:[0.0,0.0,0.0] for vid in voids}
    dip_int_N       = {vid:0 for vid in voids}

    # cache of computed galaxy positions for null-test reuse
    gal_cache = []  # list of (zone_id, gxyz)

    used = 0
    missing_nomap = 0
    missing_pos   = 0
    n_gal = 0

    for line in iter_data_lines(galzones_path):
        t = tokenize(line)
        try: zone_id = int(t[gmap["zone"]])
        except: continue
        void_idx = z2v.get(zone_id, None)
        if void_idx is None or void_idx < 0 or void_idx not in voids:
            missing_nomap += 1; n_gal += 1; continue
        try:
            gid = int(t[gmap["gal_id"]])
        except:
            missing_pos += 1; n_gal += 1; continue
        pos = nsa_pos.get(gid)
        if pos is None:
            missing_pos += 1; n_gal += 1; continue
        ra,dec,zz = pos
        gxyz = ra_dec_z_to_xyz(ra, dec, zz, cosmo)
        gal_cache.append((zone_id, gxyz))
        ctr, R, zvoid = voids[void_idx]
        r = dist3(gxyz, ctr)
        rnorm = r / R

        # update per-void radial hist within RMAX_NORM
        if rnorm < RMAX_NORM:
            # find bin
            if np is not None:
                bi = int(np.searchsorted(bin_edges, rnorm, side="right")-1)
            else:
                # linear search
                bi = 0
                for i in range(NBINS):
                    if bin_edges[i] <= rnorm < bin_edges[i+1]: bi = i; break
            if 0 <= bi < NBINS:
                per_void_hist[void_idx][bi] += 1

        # occupancy & inward-bias (interior only)
        if r <= SHELL_MIN * R:
            counts_interior[void_idx] += 1
            used += 1
            ib = (R - r) / R
            sum_inward[void_idx] += ib
            n_inward[void_idx]   += 1
            # anisotropy dipole (interior)
            ux,uy,uz = (gxyz[0]-ctr[0])/r, (gxyz[1]-ctr[1])/r, (gxyz[2]-ctr[2])/r
            v = dip_int_vec[void_idx]
            dip_int_vec[void_idx] = [v[0]+ux, v[1]+uy, v[2]+uz]
            dip_int_N[void_idx]  += 1
        elif SHELL_MIN * R < r < SHELL_MAX * R:
            counts_shell[void_idx] += 1
            used += 1
            # anisotropy dipole (shell)
            ux,uy,uz = (gxyz[0]-ctr[0])/r, (gxyz[1]-ctr[1])/r, (gxyz[2]-ctr[2])/r
            v = dip_shell_vec[void_idx]
            dip_shell_vec[void_idx] = [v[0]+ux, v[1]+uy, v[2]+uz]
            dip_shell_N[void_idx]  += 1

        n_gal += 1

    # size & redshift quartiles
    if np is not None:
        r_edges = np.quantile(np.array(void_radii), [0,0.25,0.5,0.75,1.0])
        z_edges = np.quantile(np.array(void_redshifts), [0,0.25,0.5,0.75,1.0])
    else:
        r_edges = summarize_bins(void_radii, 4); z_edges = summarize_bins(void_redshifts, 4)

    # per-void stats CSV
    void_stats_csv = f"deep_stats_by_void_{label}.csv"
    with open(void_stats_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["void_id","R_void","z_void",
                    "n_interior","n_shell","f_boundary",
                    "mean_inward_proxy",
                    "dipole_shell","dipole_interior",
                    "size_q","z_q"])
        for vid,(ctrRZ) in enumerate(voids.values()):
            ctr,R,zv = ctrRZ
            nint = counts_interior.get(vid,0)
            nsh  = counts_shell.get(vid,0)
            denom = nint + nsh
            f_b = (nsh/denom) if denom>0 else ""
            m_ib = (sum_inward[vid]/n_inward[vid]) if n_inward[vid] > 0 else ""
            # dipoles
            dsN = dip_shell_N[vid]
            dip_shell = ""
            if dsN > 0:
                vx,vy,vz = dip_shell_vec[vid]
                mag = math.sqrt(vx*vx+vy*vy+vz*vz) / dsN
                dip_shell = f"{mag:.6f}"
            diN = dip_int_N[vid]
            dip_int = ""
            if diN > 0:
                vx,vy,vz = dip_int_vec[vid]
                mag = math.sqrt(vx*vx+vy*vy+vz*vz) / diN
                dip_int = f"{mag:.6f}"
            # bins
            rq = which_bin(R, r_edges)
            zq = which_bin(zv, z_edges)
            w.writerow([vid, f"{R:.6f}", f"{zv:.6f}",
                        nint, nsh,
                        ("" if f_b=="" else f"{f_b:.6f}"),
                        ("" if m_ib=="" else f"{m_ib:.6f}"),
                        dip_shell, dip_int, rq, zq])

    # stacked profiles (sum over voids), also by size quartile
    stacked_counts = np.zeros(NBINS, dtype=float) if np is not None else [0.0]*NBINS
    size_counts = [ (np.zeros(NBINS, dtype=float) if np is not None else [0.0]*NBINS) for _ in range(4) ]
    for vid,h in per_void_hist.items():
        if np is not None:
            stacked_counts += np.array(h, dtype=float)
            size_counts[ which_bin(voids[vid][1], r_edges) ] += np.array(h, dtype=float)
        else:
            for i in range(NBINS):
                stacked_counts[i] += h[i]
                size_counts[ which_bin(voids[vid][1], r_edges) ][i] += h[i]

    # volume-corrected density: divide by u^2 Δu (u ≈ bin center)
    if np is not None:
        u_cent = 0.5*(bin_edges[:-1]+bin_edges[1:])
        volfac = (u_cent*u_cent*np.diff(bin_edges) + 1e-12)
        dens = stacked_counts / volfac
    else:
        u_cent = [0.5*(bin_edges[i]+bin_edges[i+1]) for i in range(NBINS)]
        volfac = [ (u_cent[i]*u_cent[i]*(bin_edges[i+1]-bin_edges[i]) + 1e-12) for i in range(NBINS) ]
        dens = [ stacked_counts[i]/volfac[i] for i in range(NBINS) ]

    prof_csv = f"deep_profiles_stacked_{label}.csv"
    with open(prof_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u_r_over_R","count","dens_volcorr"])
        for i in range(NBINS):
            w.writerow([f"{u_cent[i]:.5f}", f"{float(stacked_counts[i]):.6f}", f"{float(dens[i]):.6f}"])

    byq_csv = f"deep_profiles_by_sizequartile_{label}.csv"
    with open(byq_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quartile","u_r_over_R","count","dens_volcorr"])
        for q in range(4):
            if np is not None:
                dens_q = size_counts[q] / volfac
            else:
                dens_q = [ size_counts[q][i]/volfac[i] for i in range(NBINS) ]
            for i in range(NBINS):
                w.writerow([q, f"{u_cent[i]:.5f}", f"{float(size_counts[q][i]):.6f}", f"{float(dens_q[i]):.6f}"])

    # Null / control: randomize zone→void map and recompute occupancy & inward bias
    random.seed(RAND_SEED)
    void_keys = list(voids.keys())
    z2v_keys = list(z2v.keys())
    # random void assignment per zone:
    rand_assign = { z: random.choice(void_keys) for z in z2v_keys }

    null_interior = 0
    null_shell    = 0
    null_ib_sum   = 0.0
    null_ib_cnt   = 0

    for zone_id, gxyz in gal_cache:
        v = rand_assign.get(zone_id, None)
        if v is None: continue
        ctr,R,zv = voids[v]
        r = dist3(gxyz, ctr)
        if r <= SHELL_MIN * R:
            null_interior += 1
            null_ib_sum += (R - r)/R
            null_ib_cnt += 1
        elif SHELL_MIN * R < r < SHELL_MAX * R:
            null_shell += 1

    # summary file
    summary_txt = f"deep_summary_{label}.txt"
    with open(summary_txt, "w") as f:
        f.write(f"Catalog: {label}\n")
        f.write(f"Voids: {n_voids}\n")
        f.write(f"Processed galaxy lines: {n_gal}\n")
        f.write(f"Used (interior+shell): {used}\n")
        f.write(f"Missing (nomap): {missing_nomap}\n")
        f.write(f"Missing (no NSA pos): {missing_pos}\n")
        f.write("\nOccupancy & Inward Bias (empirical):\n")
        total_int = sum(counts_interior.values())
        total_sh  = sum(counts_shell.values())
        f_b = (total_sh / (total_sh+total_int)) if (total_sh+total_int)>0 else float('nan')
        global_ib = (sum(sum_inward.values()) / sum(n_inward.values())) if sum(n_inward.values())>0 else float('nan')
        f.write(f"  global f_boundary = {f_b:.6f}\n")
        f.write(f"  global inward_bias_proxy (interior only) = {global_ib:.6f}\n")
        f.write("\nNull / Control (random zone→void remap):\n")
        null_f = (null_shell / (null_shell + null_interior)) if (null_shell + null_interior)>0 else float('nan')
        null_ib = (null_ib_sum / null_ib_cnt) if null_ib_cnt>0 else float('nan')
        f.write(f"  null f_boundary = {null_f:.6f}\n")
        f.write(f"  null inward_bias_proxy = {null_ib:.6f}\n")

    dt = time.time()-t0
    print(f"[done] {label}:")
    print(f"  stats → {void_stats_csv}")
    print(f"  stacked profiles → {prof_csv}")
    print(f"  size-quartile profiles → {byq_csv}")
    print(f"  summary → {summary_txt}")
    print(f"  elapsed: {dt:.1f}s\n")

def main():
    global LOG_FILE # Access the global file handle
    cwd = os.getcwd()

    # Open the log file
    try:
        with open(LOG_FILENAME, 'w') as f:
            LOG_FILE = f # Set the global file handle

            # --- Original main logic starts here ---
            catalogs = discover_catalogs(cwd)
            if not catalogs:
                print("No *_zobovoids.dat + companions found."); sys.exit(1)
            nsa = load_nsa_positions_or_none(os.path.join(cwd, NSA_FITS_BASENAME))
            if not nsa:
                print("[fatal] NSA positions missing; cannot proceed."); sys.exit(1)
            print(f"[NSA] positions loaded: {len(nsa):,}")

            for vpath, gpath, zpath, base in catalogs:
                print("="*80)
                print(f"Catalog: {os.path.basename(vpath)}")
                print(f"  galzones:  {os.path.basename(gpath)}")
                print(f"  zonevoids: {os.path.basename(zpath)}")
                try:
                    process_catalog(vpath, gpath, zpath, nsa)
                except Exception as e:
                    print(f"[error] {base}: {e}")
            # --- Original main logic ends here ---

    except Exception as e:
        # If logging itself fails, print to original console print
        _original_print(f"Fatal error during main execution (or file open): {e}", file=sys.stderr)
    finally:
        LOG_FILE = None # Clear the global file handle when done

if __name__ == "__main__":
    main()