"""Power-side features of the manuscript's electrical-input regression (Eqs. 12-13)."""
from __future__ import annotations

import numpy as np

from . import model as M
from .wings import node_masses

TAU_REF = 1e-3   # N m
N_INERTIA = 2048


def kinetic_energy_history(mesh: M.Mesh, masses: np.ndarray, motion: M.Motion, n: int = N_INERTIA) -> np.ndarray:
    """Kinetic energy of the undeformed wing under the prescribed motion at n phases of one cycle."""
    flap_point = np.array([-motion.root_offset, 0.0, 0.0])
    ex, ez = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    k = np.empty(n)
    for j in range(n):
        st = motion.state(j / n)
        r_feather, r_flap = M._rot(ex, st["theta"]), M._rot(ez, st["phi"])
        pts = (mesh.points0 @ r_feather.T - flap_point) @ r_flap.T + flap_point
        feather_point = (-flap_point) @ r_flap.T + flap_point
        feather_dir = r_flap @ ex
        v = np.cross(st["phi_d"] * ez, pts - flap_point) + np.cross(st["theta_d"] * feather_dir, pts - feather_point)
        k[j] = 0.5 * np.sum(masses * np.einsum("ij,ij->i", v, v))
    return k


def inertial_features(mesh: M.Mesh, material: M.Material, membrane: str, motion: M.Motion) -> tuple[float, float]:
    """(Pi_I, Q_I): mean positive kinetic-energy rate [W] and normalised squared rate [-]."""
    k = kinetic_energy_history(mesh, node_masses(mesh, material, membrane), motion)
    p = (np.roll(k, -1) - k) * N_INERTIA * motion.freq
    return float(np.mean(np.maximum(p, 0))), float(np.mean((p / (2 * np.pi * motion.freq * TAU_REF)) ** 2))


def rigid_aero_power(mesh: M.Mesh, material: M.Material, motion: M.Motion, aero: M.Aero = M.Aero(), n_phase: int = 96) -> float:
    """Signed cycle mean of -M_phi * phi_dot on the undeformed surface [W]."""
    return M.run_cycle(mesh, material, motion, aero, n_phase=n_phase, rigid=True)["aero_power"]
