"""Control: feathering at the 50 deg limiter, with ONE flapping amplitude Phi fitted on the training lift.

The video shows flapping amplitudes of 83-103 deg at 20 Hz against the nominal 80 deg. This control asks
whether a single corrected amplitude, shared by all wings and frequencies, closes the lift gaps.
Electrical input uses the original regression (it is not the subject of this control).
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from wingmodel import features, model as M, wings

PHI_GRID = [70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0]
BASE = dict(theta_m_deg=50.0, c_theta=1.0 / 0.22, phase_c=np.pi / 2)


def _physics(job):
    planform, membrane, freq, phi = job
    material = wings.MEMBRANES[membrane]
    mesh = wings.planform_mesh(planform, material)
    motion = M.Motion(freq=freq, phi_amp_deg=phi, **BASE)
    cycle = M.run_cycle(mesh, material, motion, n_phase=24, n_cycles=2)
    pi_i, q_i = features.inertial_features(mesh, material, membrane, motion)
    return dict(lift_mN=1e3 * cycle["mean_lift"], pa_rigid=features.rigid_aero_power(mesh, material, motion), pi_i=pi_i, q_i=q_i)


def fit_predict(ctx):
    conds = sorted({(r["planform"], r["membrane"], r["freq_Hz"]) for r in ctx.train + ctx.test})
    jobs = [c + (phi,) for c in conds for phi in PHI_GRID]
    cache_path = Path(ctx.work_dir) / "phi_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    key = lambda j: "|".join(map(str, j))  # noqa: E731
    todo = [j for j in jobs if key(j) not in cache]
    if todo:
        with ProcessPoolExecutor(max_workers=ctx.num_workers) as pool:
            for j, res in zip(todo, pool.map(_physics, todo)):
                cache[key(j)] = res
        cache_path.write_text(json.dumps(cache))
    cid = lambda r: (r["planform"], r["membrane"], r["freq_Hz"])  # noqa: E731
    y = np.array([r["lift_mean_mN"] for r in ctx.train])
    w = 1.0 / (np.array([r["lift_sd_mN"] for r in ctx.train]) ** 2 + 1.0)
    loss = [float(np.sum(w * (np.array([cache[key(cid(r) + (phi,))]["lift_mN"] for r in ctx.train]) - y) ** 2)) for phi in PHI_GRID]
    phi = PHI_GRID[int(np.argmin(loss))]
    print(f"[phi] fold={ctx.fold} fitted Phi={phi} deg; loss by Phi: {dict(zip(PHI_GRID, np.round(loss, 1)))}", flush=True)
    row = lambda r: [1.0] + [cache[key(cid(r) + (phi,))][k] for k in ("pa_rigid", "pi_i", "q_i")]  # noqa: E731
    coef, _ = nnls(np.array([row(r) for r in ctx.train]), np.array([r["power_mean_W"] for r in ctx.train]))
    return [dict(wing=r["wing"], freq_Hz=r["freq_Hz"], lift_mN=cache[key(cid(r) + (phi,))]["lift_mN"],
                 power_W=float(max(np.dot(coef, row(r)), 1e-3))) for r in ctx.test]
