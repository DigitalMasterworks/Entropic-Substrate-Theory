#!/usr/bin/env python3
import json, glob, os
paths = sorted(glob.glob("outputs/precession_run_*_summary.json"))
out = ["# Substrate g−2: Precession Results\n"]
for p in paths:
    with open(p) as f: J=json.load(f)
    cfg = J["config"]; S = J["S_stats"]
    out += [
      f"## {os.path.basename(p)}",
      f"- mode: **{cfg['mode']}**   κμ={cfg['kappa_mu']}, κp={cfg['kappa_p']}",
      f"- S_mean={S['S_mean']:.12f} (min={S['S_min']:.12f}, max={S['S_max']:.12f})",
      f"- R_std={J['ratio_std']:.12e}  R_meas={J['ratio_meas']:.12e}",
      f"- Δln R={J['delta_ln_ratio']:.12e}",
      f"- δaμ={J['delta_a_linear']:.12e}  (×1e11={J['delta_a_linear_x1e11']:.3f})",
      f"- within 1σ room? {J['within_current_room_1sigma']}",
      ""
    ]
text = "\n".join(out)
with open("outputs/g2_report.md","w") as f: f.write(text)
print("wrote outputs/g2_report.md")
