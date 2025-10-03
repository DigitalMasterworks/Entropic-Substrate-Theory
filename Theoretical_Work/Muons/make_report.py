#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, glob, os, math, datetime

OUT = "outputs"
ROOM_1SIG, ROOM_2SIG = 63.0, 126.0   # ×1e-11

def bounds_for(J):
    km, kp = J["config"]["kappa_mu"], J["config"]["kappa_p"]
    kd = km - kp
    da = J["delta_a_linear_x1e11"]  # already ×1e-11
    if abs(kd) < 1e-15:
        return kd, da, None, None
    sens = da / kd
    if not math.isfinite(sens) or abs(sens) < 1e-15:
        return kd, da, None, None
    return kd, da, ROOM_1SIG/abs(sens), ROOM_2SIG/abs(sens)

def main():
    outs = []
    outs.append("# Substrate g−2: Fermilab-style Precession Test (Auto Report)\n")
    outs.append(f"_Generated: {datetime.datetime.utcnow().isoformat()}Z_\n")
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "precession_run_*_summary.json"))):
        J = json.load(open(p))
        cfg, S = J["config"], J["S_stats"]
        kd, da, b1, b2 = bounds_for(J)
        block = [
          f"## {os.path.basename(p)}",
          f"- mode: **{cfg['mode']}**   κμ={cfg['kappa_mu']}  κp={cfg['kappa_p']}",
          f"- S_mean={S['S_mean']:.12f} (min={S['S_min']:.12f}, max={S['S_max']:.12f})",
          f"- ΔlnR={J['delta_ln_ratio']:.12e}   δaμ={J['delta_a_linear']:.12e}  (×1e11={J['delta_a_linear_x1e11']:.3f})",
          f"- within 1σ ‘room’ (±63×1e−11)?  **{J['within_current_room_1sigma']}**",
        ]
        if b1 is None:
            block.append(f"- κμ−κp = {kd:+.3f} ⇒ **no bound** (universal/cancel or zero sensitivity)")
        else:
            block.append(f"- Sensitivity ⇒ |κμ−κp| ≤ **{b1:.3f}** (1σ), ≤ **{b2:.3f}** (2σ)")
        block.append("")
        outs.extend(block)
        rows.append((p, cfg, S, J))
    open(os.path.join(OUT, "final_report.md"), "w").write("\n".join(outs))
    print("wrote outputs/final_report.md")

if __name__ == "__main__":
    main()