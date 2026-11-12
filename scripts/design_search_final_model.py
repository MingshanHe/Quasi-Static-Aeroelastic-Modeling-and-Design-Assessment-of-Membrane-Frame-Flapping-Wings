"""Design-family search under an extended model produced by the study (a `fit_predict` program).

    python scripts/design_search_final_model.py final_model/program_full.py results/analyses/design_final_full

The program is identified on the six reference wings (as when the designs were chosen) and asked to
predict lift and electrical input for a grid of the design family (span 80 mm, chord 26 mm envelope)
plus the two fabricated designs, at 10 and 15 Hz. Needs measurements/conditions.csv.
"""
import csv
import importlib.util
import itertools
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wing_lib"))
from wingmodel import design, wings  # noqa: E402

KEYS = ["wing", "planform", "membrane", "freq_Hz", "n", "freq_actual_Hz", "lift_mean_mN", "lift_sd_mN", "power_mean_W",
        "power_sd_W", "ratio_mean_mN_per_W", "ratio_sd"]


def main(program_path, out_dir, freqs=(10, 15)):
    out = Path(out_dir)
    (out / "work").mkdir(parents=True, exist_ok=True)
    grid = {}
    for lam1, eta1, te in itertools.product([0.0, 0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8, 0.9], [30.0, 60.0, 90.0]):
        if design.feasible_mesh(lam1, eta1, te) is not None:
            grid[f"d_{lam1:.1f}_{eta1:.1f}_{te:.0f}"] = dict(R=0.080, C=0.026, lam1=lam1, eta1=eta1, theta_e_deg=te)
    wings.DESIGN_PARAMETERS.update(grid)          # inherited by forked worker processes
    train, frame = [], []
    with open(ROOT / "measurements" / "conditions.csv") as fh:
        for r in csv.DictReader(fh):
            row = {k: (r[k] if k in ("wing", "planform", "membrane") else float(r[k])) for k in KEYS}
            row["freq_Hz"] = int(row["freq_Hz"])
            if r["wing"] in wings.REFERENCE_WINGS:
                train.append(row)
            elif r["wing"] == "Bone":
                frame.append(row)
    names = list(grid) + ["opt", "s-opt"]
    test = [dict(wing=n, planform=n, membrane="kapton", freq_Hz=f) for n in names for f in freqs]
    sys.path.insert(0, str(Path(program_path).resolve().parent))
    spec = importlib.util.spec_from_file_location(Path(program_path).stem, program_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    ctx = types.SimpleNamespace(train=train, test=test, frame_only=frame, fold="design_search", work_dir=str(out / "work"),
                                num_workers=len(os.sched_getaffinity(0)), time_budget_s=6 * 3600)
    preds = mod.fit_predict(ctx)
    rows = []
    for p in preds:
        par = wings.DESIGN_PARAMETERS[p["wing"]]
        mesh = wings.planform_mesh(p["wing"])
        rows.append(dict(design=p["wing"], lam1=par["lam1"], eta1=par["eta1"], theta_e_deg=par["theta_e_deg"], freq_Hz=p["freq_Hz"],
                         area_mm2=round(1e6 * mesh.area, 1), lift_mN=round(p["lift_mN"], 4), power_W=round(p["power_W"], 5),
                         ratio_mN_per_W=round(p["lift_mN"] / p["power_W"], 4)))
    with open(out / "design_search.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for f in freqs:
        sub = [r for r in rows if r["freq_Hz"] == f]
        for key in ("lift_mN", "ratio_mN_per_W"):
            order = sorted(sub, key=lambda r: -r[key])
            print(f"--- {f} Hz, ranked by {key} ({len(sub)} designs)")
            for r in order[:5]:
                print("   %-16s area %6.0f  lift %6.2f  P %.3f  ratio %.2f" % (r["design"], r["area_mm2"], r["lift_mN"], r["power_W"], r["ratio_mN_per_W"]))
            for n in ("opt", "s-opt"):
                r = next(x for x in sub if x["design"] == n)
                print("   fabricated %-6s rank %d/%d  lift %.2f  P %.3f  ratio %.2f" % (n, order.index(r) + 1, len(sub), r["lift_mN"], r["power_W"], r["ratio_mN_per_W"]))
        v = np.array([r["ratio_mN_per_W"] for r in sub])
        print(f"   ratio range {v.min():.2f}..{v.max():.2f}; within 2% of best: {(v >= 0.98 * v.max()).sum()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
