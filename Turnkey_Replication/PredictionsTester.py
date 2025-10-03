#!/usr/bin/env python3
# Streaming SDSS/VAST Void Boundary-Occupancy Analyzer
#
# - Discovers VIDE/REVOLVER catalogs in the current folder
# - Loads void centers/radii into a small dict
# - Streams gal→zone lines once, mapping zone→void
# - Resolves galaxy positions from galzones (if real RA/Dec/z) or NSA FITS (via fitsloader)
# - Computes boundary occupancy per void; writes CSV for each catalog

from __future__ import annotations
import os, re, sys, math, glob, csv, itertools, time
from typing import Dict, Tuple, Optional, List

# --- LOGGING SETUP ---
LOG_FILENAME = "your_log_file_name.log" # Make sure to change this file name for each script!
LOG_FILE = None
_original_print = print

def log_print(*args, **kwargs):
    kwargs_for_stdout = dict(kwargs)
    kwargs_for_stdout.pop('file', None) 

    text = kwargs.get('sep', ' ').join(map(str, args))
    end_char = kwargs.get('end', '\n')

    # Print to console (always sys.stdout)
    _original_print(text, **kwargs_for_stdout, file=sys.stdout) 

    # Log to file
    if LOG_FILE:
        LOG_FILE.write(text + end_char)

print = log_print
# --- END LOGGING SETUP ---

try:
    import numpy as np
except Exception:
    np = None  # fallback to math

# --- NSA loader (expected in same dir; see fitsloader.py you already ran) ---
NSA_FITS_BASENAME = "nsa_v1_0_1.fits"
def load_nsa_positions_or_none(fits_path: str) -> Optional[Dict[int, Tuple[float,float,float]]]:
    if not os.path.exists(fits_path):
        return None
    try:
        from fitsloader import load_nsa_positions
    except Exception as e:
        print(f"[note] NSA FITS present but fitsloader not importable ({e}); skipping NSA lookup.")
        return None
    try:
        return load_nsa_positions(fits_path)  # dict: {NSAID: (RA, Dec, z)}
    except Exception as e:
        print(f"[note] Could not load NSA positions ({e}); continuing without NSA lookup.")
        return None

# ---------- USER CONFIG ----------
SHELL_MIN = 0.8
SHELL_MAX = 1.2
PRINT_SNIFF_EXAMPLES = 3
VERBOSE = True
# ---------------------------------

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
        return Dc_mpc / self.h  # h^-1 Mpc (correct)

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

# ---------- utils ----------
def is_int_token(tok: str) -> bool:
    try: int(tok); return True
    except: return False
def is_float_token(tok: str) -> bool:
    try: float(tok); return True
    except: return False
def tokenize(line: str) -> List[str]:
    return line.strip().split()
def iter_data_lines(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip(): continue
            if line.lstrip().startswith(("#",";","//")): continue
            yield line

# ---------- parsers ----------
class VoidRecord:
    __slots__ = ("void_id","center","radius")
    def __init__(self, void_id: int, center: Tuple[float,float,float], radius: float):
        self.void_id = void_id; self.center = center; self.radius = radius

def sniff_voids_format(path: str) -> Dict[str,int]:
    examples = []
    for line in iter_data_lines(path):
        t = tokenize(line)
        if len(t) >= 5:
            examples.append(t)
            if len(examples) >= 50: break
    if not examples:
        raise RuntimeError(f"No parseable lines in {path}")

    # integer id column? (optional)
    id_idx = None
    num_cols = max(len(t) for t in examples)
    for j in range(num_cols):
        if all(j < len(r) and is_int_token(r[j]) for r in examples[:15]):
            id_idx = j; break

    best = None
    for a,b,c,d in itertools.permutations(range(num_cols), 4):
        if any(max(a,b,c,d) >= len(r) for r in examples[:15]): continue
        if not all(is_float_token(r[a]) and is_float_token(r[b]) and is_float_token(r[c]) and is_float_token(r[d])
                   for r in examples[:15]): continue
        def ok_ra(x):  v=float(x); return 0.0 <= v <= 360.0
        def ok_dec(x): v=float(x); return -90.0 <= v <= 90.0
        def ok_z(x):   v=float(x); return -0.01 <= v <= 3.0
        def ok_r(x):   v=float(x); return v > 0
        is_radeczr = all(ok_ra(r[a]) and ok_dec(r[b]) and ok_z(r[c]) and ok_r(r[d]) for r in examples[:15])
        def not_small(x): return abs(float(x)) > 3.0
        is_xyzr    = all(not_small(r[a]) and not_small(r[b]) and not_small(r[c]) and ok_r(r[d]) for r in examples[:15])
        if is_radeczr:
            best = ("ra","dec","z","r",(a,b,c,d)); break
        if is_xyzr and best is None:
            best = ("x","y","z","r",(a,b,c,d))
    if best is None:
        colmap = {"id": id_idx, "ra":1,"dec":2,"z":3,"r":4}
    else:
        nm = {"id": id_idx}
        keys = best[:4]; idxs = best[4]
        for k,i in zip(keys, idxs): nm[k]=i
        colmap = nm

    if VERBOSE:
        base = os.path.basename(path)
        print(f"[sniff] {base} → column map: {colmap}")
        shown = 0
        for line in iter_data_lines(path):
            toks = tokenize(line)
            vid = (int(toks[colmap["id"]]) if colmap.get("id") is not None else f"(auto:{shown+1})")
            rr  = float(toks[colmap["r"]])
            if "ra" in colmap:
                ra = float(toks[colmap["ra"]]); dec = float(toks[colmap["dec"]]); zz = float(toks[colmap["z"]])
                print(f"  ex void: id={vid} ra={ra:.3f} dec={dec:.3f} z={zz:.4f} R={rr:.3f}")
            else:
                x = float(toks[colmap["x"]]); y = float(toks[colmap["y"]]); zc = float(toks[colmap["z"]])
                print(f"  ex void: id={vid} x={x:.2f} y={y:.2f} z={zc:.2f} R={rr:.2f}")
            shown += 1
            if shown >= PRINT_SNIFF_EXAMPLES: break
    return colmap

def read_voids(path: str, cosmo: Cosmo) -> Dict[int, VoidRecord]:
    cmap = sniff_voids_format(path)
    voids: Dict[int, VoidRecord] = {}
    row_idx = -1
    for line in iter_data_lines(path):
        t = tokenize(line); row_idx += 1
        R = float(t[cmap["r"]])
        if "ra" in cmap:
            ra = float(t[cmap["ra"]]); dec = float(t[cmap["dec"]]); zz = float(t[cmap["z"]])
            ctr = ra_dec_z_to_xyz(ra, dec, zz, cosmo)
        else:
            x = float(t[cmap["x"]]); y = float(t[cmap["y"]]); zc = float(t[cmap["z"]])
            ctr = (x,y,zc)
        voids[row_idx] = VoidRecord(row_idx, ctr, R)
    if VERBOSE:
        print(f"[voids] loaded {len(voids)} voids from {os.path.basename(path)} (keys=0..{len(voids)-1})")
    return voids

def sniff_zonevoids_format(path: str) -> Tuple[int,int]:
    examples = []
    for line in iter_data_lines(path):
        t = tokenize(line)
        if len(t) >= 2 and is_int_token(t[0]) and is_int_token(t[1]):
            examples.append(t)
            if len(examples) >= 5: break
    if not examples:
        raise RuntimeError(f"Could not sniff {path}")
    if VERBOSE:
        print(f"[sniff] {os.path.basename(path)} → assume cols: zone_id=0, void_id=1")
        for t in examples[:PRINT_SNIFF_EXAMPLES]:
            print(f"  ex map: zone={t[0]} → void={t[1]}")
    return (0,1)

def build_zone_to_void(path: str) -> Dict[int,int]:
    zcol, vcol = sniff_zonevoids_format(path)
    z2v: Dict[int,int] = {}
    for line in iter_data_lines(path):
        t = tokenize(line)
        try:
            z = int(t[zcol]); v = int(t[vcol])
            z2v[z] = v
        except: continue
    if VERBOSE:
        print(f"[zonevoids] loaded {len(z2v)} zone→void mappings")
    return z2v

def sniff_galzones_format(path: str) -> Dict[str,int]:
    """
    Detect zone column and an integer galaxy ID column.
    We DO NOT treat any columns as (x,y,z) coords here to avoid misusing indices as positions.
    We only accept RA/DEC/Z if they clearly look like sky coords with low z.
    """
    ex = []
    for line in iter_data_lines(path):
        t = tokenize(line)
        if len(t) >= 3:
            ex.append(t)
            if len(ex) >= 100: break
    if not ex:
        raise RuntimeError(f"No parseable lines in {path}")

    num_cols = max(len(t) for t in ex)

    # zone column: lots of ints, low cardinality
    best_score, zone_idx = -1e9, 0
    for ci in range(num_cols):
        vals = [t[ci] for t in ex]
        ints = sum(1 for v in vals if is_int_token(v))
        uniq = len(set(v for v in vals if is_int_token(v)))
        score = ints - 0.25*uniq
        if score > best_score:
            best_score, zone_idx = score, ci

    # galaxy id column: many ints, high cardinality
    gal_idx = None; best_gal = -1e9
    for ci in range(num_cols):
        if ci == zone_idx: continue
        vals = [t[ci] for t in ex]
        ints = sum(1 for v in vals if is_int_token(v))
        uniq = len(set(v for v in vals if is_int_token(v)))
        # reward high cardinality
        score = ints + 0.5*uniq
        if score > best_gal:
            best_gal = score; gal_idx = ci

    # Try to detect RA/DEC/Z triple (optional)
    radecz = None
    for a,b,c in itertools.permutations(range(num_cols), 3):
        if a==zone_idx or b==zone_idx or c==zone_idx: continue
        try:
            RA  = [float(t[a]) for t in ex[:80]]
            DEC = [float(t[b]) for t in ex[:80]]
            ZZ  = [float(t[c]) for t in ex[:80]]
        except: continue
        ra_ok  = sum(1 for v in RA  if 0.0 <= v <= 360.0) >= 0.9*len(RA)
        dec_ok = sum(1 for v in DEC if -90.0 <= v <= 90.0) >= 0.9*len(DEC)
        z_ok   = sum(1 for v in ZZ  if -1e-3 <= v <= 0.25) >= 0.7*len(ZZ)
        if ra_ok and dec_ok and z_ok:
            radecz = (a,b,c); break

    colmap = {"zone": zone_idx, "gal_id": gal_idx}
    if radecz is not None:
        colmap["ra"], colmap["dec"], colmap["z"] = radecz

    if VERBOSE:
        print(f"[sniff] {os.path.basename(path)} → {colmap}")
        shown=0
        for t in iter_data_lines(path):
            toks = tokenize(t)
            z = int(toks[colmap["zone"]])
            gid = int(toks[colmap["gal_id"]]) if (colmap.get("gal_id") is not None and len(toks) > colmap["gal_id"] and is_int_token(toks[colmap["gal_id"]])) else -1
            if "ra" in colmap:
                ra=float(toks[colmap["ra"]]); dec=float(toks[colmap["dec"]]); zz=float(toks[colmap["z"]])
                print(f"  ex gal: gid={gid} zone={z} ra={ra:.2f} dec={dec:.2f} z={zz:.4f}")
            else:
                print(f"  ex gal: gid={gid} zone={z} (no coords in this file)")
            shown+=1
            if shown>=PRINT_SNIFF_EXAMPLES: break
    return colmap

# ---------- main processing ----------
def process_catalog(voids_path: str, galzones_path: str, zonevoids_path: Optional[str], nsa_pos: Optional[Dict[int, Tuple[float,float,float]]]) -> None:
    label = os.path.basename(voids_path).replace("_zobovoids.dat","")
    cosmo = pick_cosmo_from_name(voids_path)
    t0 = time.time()

    voids = read_voids(voids_path, cosmo)  # keys 0..N-1
    counts_interior = {k:0 for k in voids}
    counts_shell    = {k:0 for k in voids}

    sum_inward = {k: 0.0 for k in voids}   # sum of (R - r)/R over interior gals
    n_inward   = {k: 0   for k in voids}   # count of interior gals

    global_inward_sum = 0.0
    global_inward_cnt = 0

    sum_launch = {k: 0.0 for k in voids}   # Σ (R+Δ - r)/R for interior+shell
    n_launch   = {k: 0   for k in voids}
    global_launch_sum = 0.0
    global_launch_cnt = 0

    # zone→void (required)
    if not (zonevoids_path and os.path.exists(zonevoids_path)):
        print(f"[fatal] {label}: zone→void mapping missing; cannot proceed.")
        return
    z2v = build_zone_to_void(zonevoids_path)

    gmap = sniff_galzones_format(galzones_path)
    has_coords = ("ra" in gmap)  # we only trust RA/Dec/z if detected as such

    n_gal = 0
    used = 0
    missing_nomap = 0   # zone not mapped to a valid void
    missing_pos  = 0    # could not resolve galaxy position (no RA/Dec/z & no NSA hit)
    found_nsa = 0
    checked_nsa = 0

    for line in iter_data_lines(galzones_path):
        t = tokenize(line)
        try:
            zone_id = int(t[gmap["zone"]])
        except:
            continue
        void_idx = z2v.get(zone_id, None)
        if void_idx is None or void_idx < 0 or void_idx not in voids:
            missing_nomap += 1; n_gal += 1; continue

        # Resolve galaxy position
        if has_coords:
            try:
                ra=float(t[gmap["ra"]]); dec=float(t[gmap["dec"]]); zz=float(t[gmap["z"]])
                gxyz = ra_dec_z_to_xyz(ra,dec,zz,cosmo)
            except:
                missing_pos += 1; n_gal += 1; continue
        else:
            gid = int(t[gmap["gal_id"]]) if (gmap.get("gal_id") is not None and is_int_token(t[gmap["gal_id"]])) else None
            if gid is None or nsa_pos is None:
                missing_pos += 1; n_gal += 1; continue
            checked_nsa += 1
            pos = nsa_pos.get(gid)
            if pos is None:
                missing_pos += 1; n_gal += 1; continue
            found_nsa += 1
            ra,dec,zz = pos
            gxyz = ra_dec_z_to_xyz(ra,dec,zz,cosmo)

        vrec = voids[void_idx]
        r = dist3(gxyz, vrec.center)

        if r <= SHELL_MIN * vrec.radius:
            counts_interior[void_idx] += 1
            used += 1
            # inward-bias proxy: interior only
            ib = (vrec.radius - r) / vrec.radius   # in [0,1]
            sum_inward[void_idx] += ib
            n_inward[void_idx]   += 1
            global_inward_sum    += ib
            global_inward_cnt    += 1

            # (optional) launch-proxy also counts interior
            dR = (SHELL_MAX - 1.0) * vrec.radius   # Δ = 0.2R for [0.8,1.2]
            ib_launch = max(0.0, (vrec.radius + dR - r) / vrec.radius)
            sum_launch[void_idx] += ib_launch
            n_launch[void_idx]   += 1
            global_launch_sum    += ib_launch
            global_launch_cnt    += 1

        elif SHELL_MIN * vrec.radius < r < SHELL_MAX * vrec.radius:
            counts_shell[void_idx] += 1
            used += 1

            # (optional) launch-proxy also counts shell
            dR = (SHELL_MAX - 1.0) * vrec.radius
            ib_launch = max(0.0, (vrec.radius + dR - r) / vrec.radius)
            sum_launch[void_idx] += ib_launch
            n_launch[void_idx]   += 1
            global_launch_sum    += ib_launch
            global_launch_cnt    += 1

        n_gal += 1

    out_csv = f"boundary_occupancy_{label}.csv"
    with open(out_csv,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["void_key","R_void",
                    "n_interior","n_shell","f_boundary",
                    "n_inward","mean_inward_proxy",
                    "n_launch","mean_launch_proxy"])
        fvals=[]
        for vid,vrec in voids.items():
            nint = counts_interior.get(vid,0)
            nsh  = counts_shell.get(vid,0)
            denom = nint + nsh
            f_b = (nsh/denom) if denom>0 else ""

            m_ib = (sum_inward[vid]/n_inward[vid]) if n_inward[vid] > 0 else ""
            m_lp = (sum_launch[vid]/n_launch[vid]) if n_launch[vid] > 0 else ""

            # Check if f_b is not an empty string before attempting comparison
            if f_b != "":
                try:
                    fvals.append(float(f_b))
                except ValueError:
                    pass # Ignore if conversion fails (shouldn't happen with the logic above)

            w.writerow([
                vid, f"{vrec.radius:.6f}",
                nint, nsh,
                (f_b if f_b=="" else f"{f_b:.6f}"),
                n_inward[vid], (m_ib if m_ib=="" else f"{m_ib:.6f}"),
                n_launch[vid], (m_lp if m_lp=="" else f"{m_lp:.6f}")
            ])

    fv = [v for v in fvals if isinstance(v,float)]
    mean = (sum(fv)/len(fv)) if fv else float("nan")
    # Calculate median safely
    med = float("nan")
    if fv:
        sorted_fv = sorted(fv)
        n = len(sorted_fv)
        if n % 2 == 1:
            med = sorted_fv[n//2]
        else:
            med = (sorted_fv[n//2 - 1] + sorted_fv[n//2]) / 2

    dt=time.time()-t0

    global_ib = (global_inward_sum / global_inward_cnt) if global_inward_cnt > 0 else float("nan")
    print(f"       inward_bias_proxy (global, interior only) = {global_ib:.4f}  [count={global_inward_cnt:,}]")

    if global_launch_cnt > 0:
        global_lp = global_launch_sum / global_launch_cnt
        print(f"       launch_proxy (global, interior+shell, Δ={(SHELL_MAX-1.0):.2f}R) = {global_lp:.4f}  [count={global_launch_cnt:,}]")

    print(f"[done] {label}: wrote {out_csv} ({len(voids)} voids).")
    print(f"       galaxy lines: {n_gal:,}  used: {used:,}  missing_nomap: {missing_nomap:,}  missing_pos: {missing_pos:,}")
    if not has_coords and nsa_pos is not None:
        rate = (found_nsa/checked_nsa) if checked_nsa else 0.0
        print(f"       NSA lookup hit rate: {found_nsa:,}/{checked_nsa:,} = {rate:.2%}")
    print(f"       f_boundary: mean={mean:.4f}  median={med:.4f}  (SHELL=[{SHELL_MIN},{SHELL_MAX}])")
    print(f"       elapsed: {dt:.1f}s")

def discover_catalogs(cwd: str) -> List[Tuple[str,str,Optional[str]]]:
    pairs = []
    for vpath in sorted(glob.glob(os.path.join(cwd, "*_zobovoids.dat"))):
        base = os.path.basename(vpath).replace("_zobovoids.dat", "")
        cand_gz = glob.glob(os.path.join(cwd, f"{base}_galzones.dat"))
        if not cand_gz:
            pattern = base.split("_zobovoids")[0] + "*_galzones.dat"
            cand_gz = glob.glob(os.path.join(cwd, pattern))
        if not cand_gz:
            print(f"[warn] No galzones found for {base}, skipping."); continue
        gpath = cand_gz[0]
        cand_zv = glob.glob(os.path.join(cwd, f"{base}_zonevoids.dat"))
        if not cand_zv:
            pattern = base.split("_zobovoids")[0] + "*_zonevoids.dat"
            cand_zv = glob.glob(os.path.join(cwd, pattern))
        zpath = cand_zv[0] if cand_zv else None
        pairs.append((vpath, gpath, zpath))
    return pairs

def main():
    global LOG_FILE # Access the global file handle
    cwd = os.getcwd()

    # Open the log file
    try:
        with open(LOG_FILENAME, 'w') as f:
            LOG_FILE = f # Set the global file handle

            catalogs = discover_catalogs(cwd)
            if not catalogs:
                print("No *_zobovoids.dat catalogs found in this folder."); sys.exit(1)

            # NSA positions (if FITS present)
            nsa_pos = load_nsa_positions_or_none(os.path.join(cwd, NSA_FITS_BASENAME))
            if nsa_pos:
                print(f"[NSA] positions loaded: {len(nsa_pos):,} entries")

            print(f"Discovered {len(catalogs)} catalog(s). Running all...\n")

            for (vpath, gpath, zpath) in catalogs:
                print("="*80)
                print(f"Catalog: {os.path.basename(vpath)}")
                print(f"  galzones:  {os.path.basename(gpath)}")
                print(f"  zonevoids: {os.path.basename(zpath) if zpath else '(none)'}")
                try:
                    process_catalog(vpath, gpath, zpath, nsa_pos)
                except Exception as e:
                    print(f"[error] while processing {os.path.basename(vpath)}: {e}")
                    continue

    except Exception as e:
        # If logging itself fails, print to original console print
        _original_print(f"Fatal error during main execution (or file open): {e}", file=sys.stderr)
    finally:
        LOG_FILE = None # Clear the global file handle when done

if __name__ == "__main__":
    main()