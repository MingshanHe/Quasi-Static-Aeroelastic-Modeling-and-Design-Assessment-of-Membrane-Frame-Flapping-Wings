"""Baseline: the manuscript's model, used as published.

Lift: quasi-static membrane-frame aeroelastic model under the prescribed analytical motion
(flapping 80 deg sine; feathering 50 deg (the mechanical stop) tanh waveform, C_theta = 1/0.22, phase pi/2), with
aerodynamic coefficients fixed in advance. Nothing is fitted to the lift measurements.

Electrical input: non-negative linear regression on a rigid-surface aerodynamic power feature and
two inertial features (manuscript Eqs. 12-13), fitted on the training conditions only.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from wingmodel import features, model as M, wings

MOTION = dict(phi_amp_deg=80.0, theta_m_deg=50.0, c_theta=1.0 / 0.22, phase_c=np.pi / 2)
N_PHASE, N_CYCLES = 24, 2


def _physics(job):
    """Model quantities of one (planform, membrane, frequency) condition."""
    planform, membrane, freq = job
    material = wings.MEMBRANES[membrane]
    mesh = wings.planform_mesh(planform, material)
    motion = M.Motion(freq=freq, **MOTION)
    cycle = M.run_cycle(mesh, material, motion, n_phase=N_PHASE, n_cycles=N_CYCLES)
    pi_i, q_i = features.inertial_features(mesh, material, membrane, motion)
    return dict(lift_mN=1e3 * cycle["mean_lift"], pa_rigid=features.rigid_aero_power(mesh, material, motion),
                pi_i=pi_i, q_i=q_i, max_disp_mm=1e3 * cycle["max_disp"], converged=bool(cycle["all_converged"]))


def fit_predict(ctx):
    """ctx.train: measured conditions; ctx.test: conditions to predict. Returns one dict per test row."""
    jobs = sorted({(r["planform"], r["membrane"], r["freq_Hz"]) for r in ctx.train + ctx.test})
    # Physics solves do not depend on any measurement, so they are cached across the evaluator's folds.
    cache_path = Path(ctx.work_dir) / "physics_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    todo = [j for j in jobs if "|".join(map(str, j)) not in cache]
    if todo:
        with ProcessPoolExecutor(max_workers=ctx.num_workers) as pool:
            for job, result in zip(todo, pool.map(_physics, todo)):
                cache["|".join(map(str, job))] = result
        cache_path.write_text(json.dumps(cache))
    phys = {j: cache["|".join(map(str, j))] for j in jobs}

    def row(r):
        p = phys[(r["planform"], r["membrane"], r["freq_Hz"])]
        return [1.0, p["pa_rigid"], p["pi_i"], p["q_i"]]

    coef, _ = nnls(np.array([row(r) for r in ctx.train]), np.array([r["power_mean_W"] for r in ctx.train]))
    out = []
    for r in ctx.test:
        p = phys[(r["planform"], r["membrane"], r["freq_Hz"])]
        out.append(dict(wing=r["wing"], freq_Hz=r["freq_Hz"], lift_mN=p["lift_mN"], power_W=float(np.dot(coef, row(r)))))
    return out
