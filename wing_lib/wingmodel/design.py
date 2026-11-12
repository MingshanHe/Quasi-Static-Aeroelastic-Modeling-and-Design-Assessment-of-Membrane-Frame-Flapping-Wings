"""Model-based search over the manuscript's wing-design family (straight segment + elliptical arc)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from . import model as M

# Envelope and bounds of the manuscript: span R <= 80 mm, root chord C <= 26 mm.
BOUNDS = dict(lam1=(0.0, 0.95), eta1=(0.05, 0.95), theta_e_deg=(20.0, 90.0))


def feasible_mesh(lam1, eta1, theta_e_deg, R=0.080, C=0.026, material=M.KAPTON, nx=7, ny=5,
                  section=M.FrameSection()):
    """Mesh of a family member, or None when the outline is invalid (crossed edges, zero chord)."""
    try:
        mesh = M.design_mesh(R, C, lam1, eta1, theta_e_deg, nx, ny, material, section)
    except (ValueError, FloatingPointError):
        return None
    if np.any(mesh.chord <= 1e-4) or np.any(mesh.area0 <= 1e-9) or not np.all(np.isfinite(mesh.points0)):
        return None
    return mesh


def search(objective, R=0.080, C=0.026, include_rectangle=True, seed=0, popsize=12, maxiter=25, workers=1):
    """Maximise objective(mesh, params) over (lam1, eta1, theta_e); the R x C rectangle is scored
    separately because it is a distinct branch of the family. Returns the ranked evaluated designs."""
    log = []

    def neg(x):
        params = dict(lam1=float(x[0]), eta1=float(x[1]), theta_e_deg=float(x[2]), R=R, C=C)
        mesh = feasible_mesh(**params)
        value = -np.inf if mesh is None else float(objective(mesh, params))
        log.append((value, params))
        return -value if np.isfinite(value) else 1e9

    res = differential_evolution(neg, [BOUNDS["lam1"], BOUNDS["eta1"], BOUNDS["theta_e_deg"]], seed=seed,
                                 popsize=popsize, maxiter=maxiter, tol=1e-4, polish=False, workers=workers,
                                 updating="deferred" if workers != 1 else "immediate")
    best = dict(lam1=float(res.x[0]), eta1=float(res.x[1]), theta_e_deg=float(res.x[2]), R=R, C=C, value=float(-res.fun))
    rect = None
    if include_rectangle:
        params = dict(lam1=1.0, eta1=1.0, theta_e_deg=90.0, R=R, C=C)
        rect = dict(params, value=float(objective(M.design_mesh(**params), params)))
    return dict(best_curved=best, rectangle=rect, n_evaluations=int(res.nfev), log=log)
