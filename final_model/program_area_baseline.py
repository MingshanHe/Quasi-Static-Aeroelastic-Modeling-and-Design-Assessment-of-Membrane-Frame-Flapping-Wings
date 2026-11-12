"""Geometry-only baseline: no aeroelastic model, no hinge closure.

Lift = k * S2 * f^2 with S2 the second moment of planform area about the flapping axis (the
quasi-steady scaling of a rigid plate at fixed kinematics); electrical input = non-negative
regression on [1, f, S3 * f^3] with S3 the third moment (aerodynamic power scaling). k and the
regression are fitted on the training conditions. This is the trivial baseline reviewers asked for:
it ranks wings by size alone.
"""
import numpy as np
from scipy.optimize import nnls

from wingmodel import wings

ROOT_OFFSET = 6e-3


def _moments(planform):
    mesh = wings.planform_mesh(planform)
    r = mesh.points0[mesh.faces].mean(axis=1)[:, 0] + ROOT_OFFSET
    return float(np.sum(mesh.area0 * r**2)), float(np.sum(mesh.area0 * r**3))


def fit_predict(ctx):
    mom = {p: _moments(p) for p in {r["planform"] for r in ctx.train + ctx.test}}
    xl = np.array([mom[r["planform"]][0] * r["freq_Hz"] ** 2 for r in ctx.train])
    yl = np.array([r["lift_mean_mN"] for r in ctx.train])
    k = float(xl @ yl / (xl @ xl))
    row = lambda r: [1.0, r["freq_Hz"], mom[r["planform"]][1] * r["freq_Hz"] ** 3]  # noqa: E731
    X = np.array([row(r) for r in ctx.train])
    sc = np.linalg.norm(X, axis=0)
    coef = nnls(X / sc, np.array([r["power_mean_W"] for r in ctx.train]))[0] / sc
    return [dict(wing=r["wing"], freq_Hz=r["freq_Hz"], lift_mN=k * mom[r["planform"]][0] * r["freq_Hz"] ** 2,
                 power_W=float(max(np.dot(coef, row(r)), 1e-3))) for r in ctx.test]
