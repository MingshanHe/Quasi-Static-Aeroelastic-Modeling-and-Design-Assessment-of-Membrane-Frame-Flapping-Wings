"""Computations requested by reviewers, for the model under prescribed analytical motion.

    python scripts/reviewer_analyses.py rigid_control   # deformation-off control, all wings and frequencies
    python scripts/reviewer_analyses.py convergence     # mesh and phase-resolution convergence at 15 Hz
    python scripts/reviewer_analyses.py sensitivity     # motion parameters and moduli, one at a time, 15 Hz
    python scripts/reviewer_analyses.py design_grid     # full grid over the design family at 15 Hz

Each writes results/analyses/<name>.csv. Use --workers N (default: CPUs in the affinity mask).
`design_grid` needs measurements/conditions.csv to fit the electrical-input regression.
"""
import argparse
import csv
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wing_lib"))
from wingmodel import design, features, model as M, wings  # noqa: E402

OUT = ROOT / "results" / "analyses"
BASE = dict(phi_amp_deg=80.0, theta_m_deg=25.0, c_theta=1 / 0.22, phase_c=np.pi / 2)


def _write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / (name + '.csv')} ({len(rows)} rows)", flush=True)


def _rigid_control(job):
    label, freq = job
    mesh, mat, _ = wings.build_wing(label)
    mo = M.Motion(freq=freq, **BASE)
    c = M.run_cycle(mesh, mat, mo, n_phase=48, n_cycles=3)
    r = M.run_cycle(mesh, mat, mo, n_phase=48, rigid=True)
    return dict(wing=label, freq_Hz=freq, lift_coupled_mN=round(1e3 * c["mean_lift"], 5), lift_rigid_mN=round(1e3 * r["mean_lift"], 5),
                relative_change_pct=round(100 * (c["mean_lift"] - r["mean_lift"]) / r["mean_lift"], 4),
                max_displacement_mm=round(1e3 * c["max_disp"], 4), max_chordwise_slope_deg=round(M.chordwise_slope_deg(mesh, c["q_at_peak"]), 4),
                aero_power_coupled_W=round(c["aero_power"], 6), aero_power_rigid_W=round(r["aero_power"], 6), converged=c["all_converged"])


def _convergence(job):
    label, nx, ny, n_phase = job
    t0 = time.time()
    mesh, mat, _ = wings.build_wing(label, nx=nx, ny=ny)
    c = M.run_cycle(mesh, mat, M.Motion(freq=15, **BASE), n_phase=n_phase, n_cycles=2)
    r = M.run_cycle(mesh, mat, M.Motion(freq=15, **BASE), n_phase=n_phase, rigid=True)
    return dict(wing=label, nx=nx, ny=ny, nodes=len(mesh.points0), triangles=len(mesh.faces), n_phase=n_phase,
                lift_coupled_mN=round(1e3 * c["mean_lift"], 5), lift_rigid_mN=round(1e3 * r["mean_lift"], 5),
                max_displacement_mm=round(1e3 * c["max_disp"], 4), converged=c["all_converged"], seconds=round(time.time() - t0, 1))


def _sensitivity(job):
    label, name, value = job
    planform, membrane = wings.WINGS[label]
    mat = wings.MEMBRANES[membrane]
    motion = dict(BASE)
    if name == "membrane_E_factor":
        mat = M.replace(mat, membrane_E=mat.membrane_E * value)
    elif name == "frame_E_factor":
        mat = M.replace(mat, frame_E=mat.frame_E * value)
    elif name == "phase_c_deg":
        motion["phase_c"] = np.deg2rad(value)
    elif name != "nominal":
        motion[name] = value
    mesh = wings.planform_mesh(planform, mat)
    c = M.run_cycle(mesh, mat, M.Motion(freq=15, **motion), n_phase=48, n_cycles=2)
    return dict(wing=label, parameter=name, value=value, lift_mN=round(1e3 * c["mean_lift"], 5),
                max_displacement_mm=round(1e3 * c["max_disp"], 4), converged=c["all_converged"])


def _power_model():
    """Non-negative regression of electrical input on the manuscript's three features, six reference wings."""
    from scipy.optimize import nnls
    rows, y = [], []
    with open(ROOT / "measurements" / "conditions.csv") as fh:
        for r in csv.DictReader(fh):
            if r["wing"] not in wings.REFERENCE_WINGS:
                continue
            mesh, mat, mem = wings.build_wing(r["wing"])
            mo = M.Motion(freq=int(r["freq_Hz"]), **BASE)
            pi_i, q_i = features.inertial_features(mesh, mat, mem, mo)
            rows.append([1.0, features.rigid_aero_power(mesh, mat, mo), pi_i, q_i])
            y.append(float(r["power_mean_W"]))
    coef, _ = nnls(np.array(rows), np.array(y))
    return coef


def _design(job):
    lam1, eta1, theta_e, coef = job
    mesh = design.feasible_mesh(lam1, eta1, theta_e) if lam1 < 1 else M.design_mesh(0.080, 0.026, 1.0, 1.0, 90.0)
    if mesh is None:
        return None
    mo = M.Motion(freq=15, **BASE)
    c = M.run_cycle(mesh, M.KAPTON, mo, n_phase=24, n_cycles=2)
    pi_i, q_i = features.inertial_features(mesh, M.KAPTON, "kapton", mo)
    pa = features.rigid_aero_power(mesh, M.KAPTON, mo)
    power = float(np.dot(coef, [1.0, pa, pi_i, q_i]))
    return dict(lam1=lam1, eta1=eta1, theta_e_deg=theta_e, area_mm2=round(1e6 * mesh.area, 2), lift_mN=round(1e3 * c["mean_lift"], 4),
                power_W=round(power, 5), ratio_mN_per_W=round(1e3 * c["mean_lift"] / power, 4),
                max_displacement_mm=round(1e3 * c["max_disp"], 4), converged=c["all_converged"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", choices=["rigid_control", "convergence", "sensitivity", "design_grid"])
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    args = ap.parse_args()
    labels = list(wings.WINGS)
    if args.analysis == "rigid_control":
        fn, jobs = _rigid_control, [(w, f) for w in labels for f in wings.FREQUENCIES]
    elif args.analysis == "convergence":
        meshes = [(7, 5), (9, 7), (13, 9), (17, 11), (21, 13)]
        fn = _convergence
        jobs = [(w, nx, ny, 24) for w in labels for nx, ny in meshes] + [(w, 7, 5, n) for w in labels for n in (12, 48, 96)]
        jobs.sort(key=lambda j: -j[1] * j[2])   # longest first
    elif args.analysis == "sensitivity":
        grid = dict(theta_m_deg=[15, 20, 30, 35, 40, 45, 50], c_theta=[2.0, 3.0, 7.0, 10.0], phase_c_deg=[50, 70, 110, 130],
                    phi_amp_deg=[70, 75, 85, 90], membrane_E_factor=[0.25, 0.5, 2.0, 4.0], frame_E_factor=[0.25, 0.5, 2.0, 4.0])
        fn = _sensitivity
        jobs = [(w, "nominal", 0.0) for w in labels] + [(w, k, float(v)) for w in labels for k, vs in grid.items() for v in vs]
    else:
        coef = _power_model()
        print("power regression coefficients [c0, c_a, c_I, c_Q]:", np.round(coef, 5).tolist(), flush=True)
        fn = _design
        jobs = [(round(a, 3), round(b, 3), float(t), coef) for a, b, t in itertools.product(
            np.linspace(0.0, 0.9, 10), np.linspace(0.1, 0.9, 9), [30, 45, 60, 75, 90])] + [(1.0, 1.0, 90.0, coef)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for r in pool.map(fn, jobs) if r is not None]
    _write(args.analysis, rows)
    print(f"{len(jobs)} jobs in {time.time() - t0:.0f} s on {args.workers} workers", flush=True)


if __name__ == "__main__":
    main()
