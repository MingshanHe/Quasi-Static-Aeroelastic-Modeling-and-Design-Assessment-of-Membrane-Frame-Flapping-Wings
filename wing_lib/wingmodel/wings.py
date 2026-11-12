"""The eight fabricated wings and their meshes (geometry only; no measurements live in this package)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import model as M

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE / "geometry"

MEMBRANES = {"kapton": M.KAPTON, "tpu": M.TPU}
MEMBRANE_DENSITY = {"kapton": 1420.0, "tpu": 1200.0}   # kg/m^3
FRAME_DENSITY = 1350.0                                  # kg/m^3

# label: (planform, membrane). Opt-K and S-Opt-K use the design family of the manuscript's Table I
# (R, C in m; lam1, eta1; theta_e in degrees). These parameters are author-supplied.
WINGS = {
    "Rect-K": ("rectangular", "kapton"), "Rect-T": ("rectangular", "tpu"),
    "Tri-K": ("triangular", "kapton"), "Tri-T": ("triangular", "tpu"),
    "Curve-K": ("curve", "kapton"), "Curve-T": ("curve", "tpu"),
    "Opt-K": ("opt", "kapton"), "S-Opt-K": ("s-opt", "kapton"),
}
REFERENCE_WINGS = ["Rect-K", "Rect-T", "Tri-K", "Tri-T", "Curve-K", "Curve-T"]
DESIGNED_WINGS = ["Opt-K", "S-Opt-K"]
DESIGN_PARAMETERS = {
    "opt": dict(R=0.080, C=0.026, lam1=1.0, eta1=1.0, theta_e_deg=90.0),
    "s-opt": dict(R=0.080, C=0.026, lam1=0.4, eta1=0.8, theta_e_deg=60.0),
}
FREQUENCIES = [10, 11, 12, 13, 14, 15]


def planform_mesh(planform: str, material: M.Material = M.KAPTON, nx: int = 7, ny: int = 5,
                  section: M.FrameSection = M.FrameSection()) -> M.Mesh:
    if planform == "rectangular":
        return M.rectangular_mesh(0.080, 0.028, nx, ny, material, section)
    if planform in ("triangular", "curve"):
        lead = np.loadtxt(GEOMETRY / f"{planform}_leading_edge_m.csv", delimiter=",", skiprows=1)
        chord = json.loads((GEOMETRY / "planform_manifest.json").read_text())[planform]["root_chord_mm"] * 1e-3
        return M.single_tip_mesh(lead, chord, nx, ny, material, section, name=planform)
    if planform in DESIGN_PARAMETERS:
        return M.design_mesh(**DESIGN_PARAMETERS[planform], nx=nx, ny=ny, material=material, section=section)
    raise KeyError(planform)


def build_wing(label: str, nx: int = 7, ny: int = 5, section: M.FrameSection = M.FrameSection()):
    """Return (mesh, material, membrane_name) of a fabricated wing."""
    planform, membrane = WINGS[label]
    material = MEMBRANES[membrane]
    return planform_mesh(planform, material, nx, ny, section), material, membrane


def node_masses(mesh: M.Mesh, material: M.Material, membrane: str) -> np.ndarray:
    """Lumped nodal masses of membrane (one third of each triangle) and frame (half of each segment)."""
    m = np.zeros(len(mesh.points0))
    tri_mass = MEMBRANE_DENSITY[membrane] * material.membrane_t * mesh.area0
    for k in range(3):
        np.add.at(m, mesh.faces[:, k], tri_mass / 3)
    seg_mass = FRAME_DENSITY * (mesh.seg_EA / material.frame_E) * mesh.seg_len0
    for k in range(2):
        np.add.at(m, mesh.seg[:, k], seg_mass / 2)
    return m
