import numpy as np
from scipy.optimize import curve_fit

T = 100
NX, NY, NZ = 16, 16, 1

def analytic_twolevel_model(t, A, w, phi, B):
    return A * np.cos(w * t + phi) + B

def detect_quibits(history, log_file):
    T, NX, NY, NZ = history.shape
    quibit_tracks = []
    with open(log_file, 'w') as f:
        for x in range(NX):
            for y in range(NY):
                for z in range(NZ):
                    s = history[:, x, y, z]
                    last_good = -1
                    for t0 in range(0, T-20):
                        t = np.arange(20)
                        s_window = s[t0:t0+20]
                        if np.std(s_window) < 1e-3:
                            continue
                        try:
                            popt, _ = curve_fit(analytic_twolevel_model, t, s_window, p0=[0.1, 0.2, 0, np.mean(s_window)])
                            resid = np.mean((analytic_twolevel_model(t, *popt) - s_window)**2)
                            if resid < 2e-4 and abs(popt[0]) > 0.02:
                                if last_good < 0 or t0 - last_good > 1:
                                    entry = {'x':x, 'y':y, 'z':z, 'start':t0, 'end':t0+20,
                                             'A':popt[0], 'w':popt[1], 'phi':popt[2], 'B':popt[3],
                                             'resid':resid, 'count':1}
                                    quibit_tracks.append(entry)
                                    f.write(f"Qubit: {entry}\n")
                                    last_good = t0
                                else:
                                    quibit_tracks[-1]['end'] = t0+20
                                    quibit_tracks[-1]['count'] += 1
                                    last_good = t0
                        except Exception:
                            continue
    return quibit_tracks

def detect_neutrinos(history, log_file):
    T, NX, NY, NZ = history.shape
    neutrino_tracks = []
    with open(log_file, 'w') as f:
        for x in range(NX):
            for y in range(NY):
                for z in range(NZ):
                    s = history[:, x, y, z]
                    last_good = -1
                    for t0 in range(0, T-20):
                        t = np.arange(20)
                        s_window = s[t0:t0+20]
                        if np.std(s_window) < 1e-3:
                            continue
                        try:
                            popt, _ = curve_fit(analytic_twolevel_model, t, s_window, p0=[0.01, 0.1, 0, np.mean(s_window)])
                            resid = np.mean((analytic_twolevel_model(t, *popt) - s_window)**2)
                            amp = abs(popt[0])
                            if resid < 1e-4 and 0.005 < amp < 0.03:
                                if last_good < 0 or t0 - last_good > 1:
                                    entry = {'x':x, 'y':y, 'z':z, 'start':t0, 'end':t0+20,
                                             'A':amp, 'w':popt[1], 'phi':popt[2], 'B':popt[3],
                                             'resid':resid, 'count':1}
                                    neutrino_tracks.append(entry)
                                    f.write(f"Neutrino: {entry}\n")
                                    last_good = t0
                                else:
                                    neutrino_tracks[-1]['end'] = t0+20
                                    neutrino_tracks[-1]['count'] += 1
                                    last_good = t0
                        except Exception:
                            continue
    return neutrino_tracks

def scan_for_quark_candidates(history, log_file):
    T, NX, NY, NZ = history.shape
    threshold = 0.2
    min_lifetime = 6
    found = []
    with open(log_file, 'w') as f:
        for x in range(NX):
            for y in range(NY):
                for z in range(NZ):
                    s = history[:, x, y, z]
                    active = np.abs(s) > threshold
                    spans = []
                    start = None
                    for t, act in enumerate(active):
                        if act and start is None:
                            start = t
                        elif not act and start is not None:
                            if t - start >= min_lifetime:
                                spans.append((start, t))
                            start = None
                    if start is not None and T - start >= min_lifetime:
                        spans.append((start, T))
                    for (t0, t1) in spans:
                        local_isolated = True
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                for dz in [-1, 0, 1]:
                                    if dx == 0 and dy == 0 and dz == 0:
                                        continue
                                    nx, ny, nz = x+dx, y+dy, z+dz
                                    if 0 <= nx < NX and 0 <= ny < NY and 0 <= nz < NZ:
                                        neighbor = history[t0:t1, nx, ny, nz]
                                        if np.max(np.abs(neighbor)) > threshold * 0.7:
                                            local_isolated = False
                        if local_isolated:
                            entry = f"QuarkCandidate: ({x},{y},{z})  t={t0}-{t1}  amp~{np.max(np.abs(s[t0:t1])):.3f}  duration={t1-t0}\n"
                            found.append(entry)
                            f.write(entry)
    return found

def scan_for_higgs_candidates(history, log_file):
    mean_field = np.mean(history, axis=(0,1,2,3))
    std_field = np.std(history)
    found = []
    with open(log_file, 'w') as f:
        f.write(f"Global mean field: {mean_field:.6f}\n")
        f.write(f"Global std dev: {std_field:.6f}\n")
        if abs(mean_field) > 0.01 * std_field:
            entry = f"HiggsCandidate: Nonzero field VEV detected! Mean={mean_field:.6f}  Stdev={std_field:.6f}\n"
            found.append(entry)
            f.write(entry)
    return found

# --- Example: Real Data ---
np.random.seed(42)
history = np.zeros((T, NX, NY, NZ))
for t in range(T):
    for x in range(NX):
        for y in range(NY):
            for z in range(NZ):
                val = np.random.normal(0,0.01)
                if (x, y, z) in [(7,7,0), (4,12,0), (8,8,0)]:
                    val += np.cos(0.17*t + 0.5*x + 0.8*y)
                if (x, y, z) == (2,10,0):
                    val += np.cos(0.29*t)
                history[t, x, y, z] = val
history += 0.1

# --- Example: Cleanroom/Null Data (random shuffle as null) ---
history_null = np.copy(history)
np.random.shuffle(history_null.flat)

# --- Run both regular and cleanroom detections, always ---
detect_quibits(history, log_file="quibits.log")
detect_neutrinos(history, log_file="neutrinos.log")
scan_for_quark_candidates(history, log_file="quarks.log")
scan_for_higgs_candidates(history, log_file="higgs.log")

detect_quibits(history_null, log_file="check.log")
detect_neutrinos(history_null, log_file="check.log")
scan_for_quark_candidates(history_null, log_file="check.log")
scan_for_higgs_candidates(history_null, log_file="check.log")

print("Detection complete. See: quibits.log, neutrinos.log, quarks.log, higgs.log, check.log")