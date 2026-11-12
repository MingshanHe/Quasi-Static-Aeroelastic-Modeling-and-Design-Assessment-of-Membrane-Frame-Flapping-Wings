"""Quasi-static aeroelastic model of a membrane-frame flapping wing (Python port).

This is a line-by-line port of the MATLAB "V_final round-15" model used for the manuscript
(revision_round15/recalculation/model + run_proposed.m):

* structure: edge-spring and hinge membrane energies plus a discrete rod for the printed frame,
  with one normal (z) displacement per free node;
* aerodynamics: one quasi-steady panel per structural triangle (translation, rotational
  circulation, added mass), evaluated on the deformed surface;
* coupling: under-relaxed fixed-point iteration at each prescribed phase, continued in phase.

Coordinates of the wing-fixed frame: x root-to-tip (span), y leading-to-trailing (chord), z normal.
All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# ----------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """Membrane and frame elastic constants."""

    membrane_E: float = 2.5e9
    membrane_nu: float = 0.34
    membrane_t: float = 10e-6
    frame_E: float = 4.0e9


KAPTON = Material()
TPU = Material(membrane_E=20e6, membrane_nu=0.45)


@dataclass(frozen=True)
class FrameSection:
    """Rectangular frame sections (width, thickness) at the ends of each member, as in the MATLAB
    rectangular_validation_config: the leading member is constant, the root and tip members taper
    toward the free trailing edge."""

    lead_w: tuple[float, float] = (1.3e-3, 1.3e-3)
    lead_t: tuple[float, float] = (1.0e-3, 1.0e-3)
    side_w: tuple[float, float] = (1.3e-3, 0.2e-3)  # (at leading edge, at trailing edge)
    side_t: tuple[float, float] = (1.0e-3, 0.2e-3)


@dataclass(frozen=True)
class Aero:
    rho: float = 1.225
    cl_amp: float = 1.8
    cd_mean: float = 1.95
    cd_cos: float = 1.5
    coeff_limit: float = 8.0
    rotational: bool = True
    added_mass: bool = True
    translational: bool = True


@dataclass(frozen=True)
class Motion:
    """Prescribed flapping phi = Phi sin(2 pi f t) and feathering
    theta = theta0 + theta_m tanh(C sin(2 pi f t + c)) / tanh(C)."""

    freq: float = 15.0
    phi_amp_deg: float = 80.0
    theta_m_deg: float = 25.0
    theta0_deg: float = 0.0
    c_theta: float = 1.0 / 0.22
    phase_c: float = np.pi / 2
    root_offset: float = 6e-3  # radial offset of the wing root from the flapping axis

    def state(self, phase: float) -> dict:
        """Angles, rates and accelerations at cycle phase in [0, 1)."""
        w = 2 * np.pi * self.freq
        u = 2 * np.pi * phase
        phi_a = np.deg2rad(self.phi_amp_deg)
        a = np.deg2rad(self.theta_m_deg) / np.tanh(self.c_theta)
        x = self.c_theta * np.sin(u + self.phase_c)
        xd = self.c_theta * w * np.cos(u + self.phase_c)
        xdd = -self.c_theta * w * w * np.sin(u + self.phase_c)
        h = np.tanh(x)
        s = 1 - h * h
        return dict(
            phi=phi_a * np.sin(u), phi_d=phi_a * w * np.cos(u), phi_dd=-phi_a * w * w * np.sin(u),
            theta=np.deg2rad(self.theta0_deg) + a * h, theta_d=a * s * xd,
            theta_dd=a * s * (xdd - 2 * h * xd * xd),
        )


@dataclass(frozen=True)
class Solver:
    relaxation: float = 0.65
    disp_tol: float = 2e-6
    force_tol: float = 5e-3
    force_ref: float = 1e-3
    min_iter: int = 2
    max_iter: int = 60
    gtol: float = 1e-10          # N; structural residual tolerance
    max_bfgs_iter: int = 60      # Newton iterations


# ----------------------------------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------------------------------


@dataclass
class Mesh:
    points0: np.ndarray           # (n, 3)
    faces: np.ndarray             # (m, 3) zero-based
    fixed: np.ndarray
    free: np.ndarray
    edges: np.ndarray
    edge_len0: np.ndarray
    hinges: np.ndarray            # (h, 2) face indices
    hinge_angle0: np.ndarray
    seg: np.ndarray               # frame segments (s, 2)
    seg_len0: np.ndarray
    seg_EA: np.ndarray
    triples: np.ndarray           # (s-1, 3)
    triple_angle0: np.ndarray
    triple_len0: np.ndarray
    triple_EI: np.ndarray
    chord: np.ndarray             # local chord per panel
    pitch_frac: np.ndarray        # pitching-axis position from the leading edge / chord
    area0: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_free(self) -> int:
        return len(self.free)

    @property
    def area(self) -> float:
        return float(self.area0.sum())


def _angle_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    out = np.zeros(len(a))
    ok = (la > 1e-15) & (lb > 1e-15)
    out[ok] = np.arccos(np.clip(np.einsum("ij,ij->i", a[ok], b[ok]) / (la[ok] * lb[ok]), -1, 1))
    return out


def _lerp(a, b, u):
    return a + (b - a) * u


def _finish_mesh(points, faces, pet_nodes, seg_w, seg_t, fixed, material: Material,
                 lower_fn, upper_fn, meta) -> Mesh:
    n = len(points)
    all_edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, group = np.unique(all_edges, axis=0, return_inverse=True)
    group = np.asarray(group).ravel()
    face_index = np.tile(np.arange(len(faces)), 3)
    hinges = []
    for e in range(len(edges)):
        adj = face_index[group == e]
        if len(adj) == 2:
            hinges.append(adj)
    hinges = np.array(hinges, dtype=int).reshape(-1, 2)
    normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    seg = np.column_stack([pet_nodes[:-1], pet_nodes[1:]])
    seg_len0 = np.linalg.norm(points[seg[:, 1]] - points[seg[:, 0]], axis=1)
    seg_I = seg_w * seg_t**3 / 12
    triples = np.column_stack([pet_nodes[:-2], pet_nodes[1:-1], pet_nodes[2:]])
    cen = points[faces].mean(axis=1)
    lower = lower_fn(cen[:, 0])
    chord = upper_fn(cen[:, 0]) - lower
    fixed = np.array(sorted(fixed), dtype=int)
    return Mesh(
        points0=points, faces=faces, fixed=fixed, free=np.setdiff1d(np.arange(n), fixed),
        edges=edges, edge_len0=np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1),
        hinges=hinges, hinge_angle0=_angle_rows(normals[hinges[:, 0]], normals[hinges[:, 1]]),
        seg=seg, seg_len0=seg_len0, seg_EA=material.frame_E * seg_w * seg_t,
        triples=triples,
        triple_angle0=_angle_rows(points[triples[:, 1]] - points[triples[:, 0]],
                                  points[triples[:, 2]] - points[triples[:, 1]]),
        triple_len0=0.5 * (seg_len0[:-1] + seg_len0[1:]),
        triple_EI=material.frame_E * 0.5 * (seg_I[:-1] + seg_I[1:]),
        chord=chord, pitch_frac=-lower / chord, area0=0.5 * np.linalg.norm(normals, axis=1), meta=meta,
    )


def rectangular_mesh(width=80e-3, height=28e-3, nx=7, ny=5, material: Material = KAPTON,
                     section: FrameSection = FrameSection()) -> Mesh:
    """Structured rectangle; frame on root, leading and tip edges, trailing edge free."""
    xs, ys = np.linspace(0, width, nx), np.linspace(0, height, ny)
    points = np.array([[x, y, 0.0] for x in xs for y in ys])  # index = iy + ix*ny
    nid = lambda iy, ix: iy + ix * ny  # noqa: E731
    faces = []
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            v1, v2, v3, v4 = nid(iy, ix), nid(iy, ix + 1), nid(iy + 1, ix), nid(iy + 1, ix + 1)
            faces += [[v1, v2, v4], [v1, v4, v3]]
    left = [nid(iy, 0) for iy in range(ny - 1, -1, -1)]
    bottom = [nid(0, ix) for ix in range(1, nx)]
    right = [nid(iy, nx - 1) for iy in range(1, ny)]
    pet = np.array(left + bottom + right)
    n_side, n_lead = ny - 1, nx - 1
    u_left = 1 - (np.arange(1, n_side + 1) - 0.5) / n_side     # 1 at trailing end, 0 at leading
    u_lead = (np.arange(1, n_lead + 1) - 0.5) / n_lead
    u_right = (np.arange(1, n_side + 1) - 0.5) / n_side
    seg_w = np.concatenate([_lerp(*section.side_w, u_left), _lerp(*section.lead_w, u_lead),
                            _lerp(*section.side_w, u_right)])
    seg_t = np.concatenate([_lerp(*section.side_t, u_left), _lerp(*section.lead_t, u_lead),
                            _lerp(*section.side_t, u_right)])
    fixed = [nid(0, 0), nid(0, 1), nid(1, 0), nid(1, 1)]
    return _finish_mesh(points, np.array(faces), pet, seg_w, seg_t, fixed, material,
                        lambda x: np.zeros_like(x), lambda x: np.full_like(x, height),
                        dict(kind="rectangular", width=width, height=height, nx=nx, ny=ny))


def single_tip_mesh(leading_xy: np.ndarray, root_chord: float, nx=7, ny=5, material: Material = KAPTON,
                    section: FrameSection = FrameSection(), name="single-tip") -> Mesh:
    """Planform bounded by a frame-supported leading curve from (0,0) to the tip and a straight
    free membrane edge from (0, root_chord) to the tip. Spanwise columns contract to one tip node."""
    leading_xy = np.asarray(leading_xy, float)
    tip = leading_xy[-1]
    s_vals, t_vals = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    lower_fn = lambda x: np.interp(x, leading_xy[:, 0], leading_xy[:, 1])  # noqa: E731
    upper_fn = lambda x: (1 - x / tip[0]) * root_chord + (x / tip[0]) * tip[1]  # noqa: E731
    n = (nx - 1) * ny + 1
    tip_id = n - 1
    points = np.zeros((n, 3))
    for ix in range(nx - 1):
        x = s_vals[ix] * tip[0]
        lo, up = float(lower_fn(x)), float(upper_fn(x))
        if up <= lo:
            raise ValueError("free edge must stay above the frame edge")
        points[ix * ny:(ix + 1) * ny, 0] = x
        points[ix * ny:(ix + 1) * ny, 1] = lo + (up - lo) * t_vals
    points[tip_id, :2] = tip
    faces = []
    for ix in range(nx - 2):
        for iy in range(ny - 1):
            a, b = ix * ny + iy, (ix + 1) * ny + iy
            faces += [[a, b, b + 1], [a, b + 1, a + 1]]
    for iy in range(ny - 1):
        a = (nx - 2) * ny + iy
        faces.append([a, tip_id, a + 1])
    faces = np.array(faces)
    pet = np.array(list(range(ny - 1, -1, -1)) + [ix * ny for ix in range(1, nx - 1)] + [tip_id])
    seg_len0 = np.linalg.norm(points[pet[1:]] - points[pet[:-1]], axis=1)
    n_left = ny - 1
    u_left = 1 - (np.arange(1, n_left + 1) - 0.5) / n_left
    lead_len = seg_len0[n_left:]
    u_lead = (np.cumsum(lead_len) - 0.5 * lead_len) / lead_len.sum()
    seg_w = np.concatenate([_lerp(*section.side_w, u_left), _lerp(*section.lead_w, u_lead)])
    seg_t = np.concatenate([_lerp(*section.side_t, u_left), _lerp(*section.lead_t, u_lead)])
    fixed = [0, 1, ny, ny + 1]
    # Local chord and pitching-axis fraction follow the meshed outline (piecewise linear between
    # spanwise stations), as in the MATLAB inputs, not the dense CAD curve.
    col_x = np.r_[s_vals[:-1] * tip[0], tip[0]]
    col_lo = np.r_[[float(lower_fn(x)) for x in s_vals[:-1] * tip[0]], tip[1]]
    mesh_lower = lambda x: np.interp(x, col_x, col_lo)  # noqa: E731
    return _finish_mesh(points, faces, pet, seg_w, seg_t, fixed, material, mesh_lower, upper_fn,
                        dict(kind=name, root_chord=root_chord, tip=tip.tolist(), nx=nx, ny=ny))


def ellipse_family_leading(R: float, C: float, lam1: float, eta1: float, theta_e_deg: float,
                           n_arc: int = 200) -> np.ndarray:
    """Frame-supported edge of the manuscript's design family (Eq. wing_parameterization):
    a straight segment of length a = lam1 R, then an elliptical arc ending at (R, h), h = eta1 C."""
    a, h, te = lam1 * R, eta1 * C, np.deg2rad(theta_e_deg)
    A, B = (R - a) / np.sin(te), h / (1 - np.cos(te))
    v = np.linspace(0, te, n_arc)
    arc = np.column_stack([a + A * np.sin(v), B * (1 - np.cos(v))])
    return np.vstack([[0.0, 0.0], arc]) if a > 0 else arc


def design_mesh(R, C, lam1, eta1, theta_e_deg, nx=7, ny=5, material: Material = KAPTON,
                section: FrameSection = FrameSection()) -> Mesh:
    """Mesh of one member of the design family; lam1 = eta1 = 1 is the R x C rectangle."""
    if lam1 >= 1 - 1e-9 and eta1 >= 1 - 1e-9:
        return rectangular_mesh(R, C, nx, ny, material, section)
    if eta1 >= 1 - 1e-9:
        raise ValueError("eta1 = 1 with lam1 < 1 has a zero-length tip chord and no free edge above it")
    return single_tip_mesh(ellipse_family_leading(R, C, lam1, eta1, theta_e_deg), C, nx, ny, material,
                           section, name="ellipse-family")


# ----------------------------------------------------------------------------------------------
# Kinematics and aerodynamics
# ----------------------------------------------------------------------------------------------


def _rot(axis, angle):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    a = np.array([x, y, z])
    return np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(a, a) + np.sin(angle) * k


_EX, _EZ = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])


def aerodynamics(mesh: Mesh, points_body: np.ndarray, st: dict, motion: Motion, aero: Aero) -> dict:
    """Panel loads on the given body-frame surface at the kinematic state `st`."""
    flap_point = np.array([-motion.root_offset, 0.0, 0.0])
    r_feather, r_flap = _rot(_EX, st["theta"]), _rot(_EZ, st["phi"])
    rot = r_flap @ r_feather
    pts = (points_body @ r_feather.T - flap_point) @ r_flap.T + flap_point
    feather_point = (np.zeros(3) - flap_point) @ r_flap.T + flap_point
    feather_dir = r_flap @ _EX
    omega_flap, omega_feather = st["phi_d"] * _EZ, st["theta_d"] * feather_dir
    normal_ref, chord_ref = rot @ _EZ, rot @ np.array([0, 1.0, 0])

    f = mesh.faces
    tri = pts[f]                                            # (m, 3, 3)
    area_vec = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice = np.linalg.norm(area_vec, axis=1)
    nrm = area_vec / twice[:, None]
    nrm[nrm @ normal_ref < 0] *= -1
    cen = tri.mean(axis=1)
    rel_v = -np.cross(omega_flap, cen - flap_point)          # zero freestream; flapping velocity only

    p0 = mesh.points0
    jac = np.stack([p0[f[:, 1], :2] - p0[f[:, 0], :2], p0[f[:, 2], :2] - p0[f[:, 0], :2]], axis=1)
    dr = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]], axis=1)
    span = np.linalg.solve(jac, dr)[:, 0, :]                 # d r / d X
    span /= np.linalg.norm(span, axis=1)[:, None]
    span[span @ feather_dir < 0] *= -1
    chd = np.cross(nrm, span)
    chd /= np.linalg.norm(chd, axis=1)[:, None]
    flip = chd @ chord_ref < 0
    chd[flip] *= -1
    nrm[flip] *= -1

    cross = rel_v - np.einsum("ij,ij->i", rel_v, span)[:, None] * span
    speed = np.linalg.norm(cross, axis=1)
    ok = speed > 1e-12
    flow = np.zeros_like(cross)
    flow[ok] = cross[ok] / speed[ok, None]
    lift_hat = np.cross(span, flow)
    ln = np.linalg.norm(lift_hat, axis=1)
    lift_hat[ok] /= ln[ok, None]
    alpha = np.where(ok, np.arctan2(np.einsum("ij,ij->i", cross, nrm), np.einsum("ij,ij->i", cross, chd)), 0.0)
    cl = np.clip(aero.cl_amp * np.sin(2 * alpha), -aero.coeff_limit, aero.coeff_limit)
    cd = np.clip(aero.cd_mean - aero.cd_cos * np.cos(2 * alpha), 0, aero.coeff_limit)
    area = 0.5 * twice
    force = np.zeros_like(cross)
    if aero.translational:
        force += (0.5 * aero.rho * speed**2 * area)[:, None] * (cd[:, None] * flow + cl[:, None] * lift_hat)
    if aero.rotational:
        pitch_rate = st["theta_d"] * (span @ feather_dir)
        force += (aero.rho * np.pi * (0.75 - mesh.pitch_frac) * speed * pitch_rate * mesh.chord * area)[:, None] * nrm
    if aero.added_mass:
        omega = omega_flap + omega_feather
        acc_ang = st["phi_dd"] * _EZ + st["theta_dd"] * feather_dir + np.cross(omega_flap, omega_feather)
        rel = cen - feather_point
        root_rel = feather_point - flap_point
        root_acc = np.cross(st["phi_dd"] * _EZ, root_rel) + np.cross(omega_flap, np.cross(omega_flap, root_rel))
        acc = root_acc + np.cross(acc_ang, rel) + np.cross(omega, np.cross(omega, rel))
        a_n = -np.einsum("ij,ij->i", acc, nrm)
        force += (np.pi / 4 * aero.rho * mesh.chord * area * a_n)[:, None] * nrm

    nodal = np.zeros_like(pts)
    for k in range(3):
        np.add.at(nodal, f[:, k], force / 3)
    total = nodal.sum(axis=0)
    moment_flap = np.cross(pts - flap_point, nodal).sum(axis=0) @ _EZ
    return dict(nodal_body=nodal @ rot, total_inertial=total, total_body=(nodal @ rot).sum(axis=0),
                lift=float(total[2]), moment_flap=float(moment_flap), alpha=alpha, speed=speed, area=area)


# ----------------------------------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------------------------------


def _angle_energy_grad(a, b, k, theta0):
    """Energy 0.5 k (theta - theta0)^2 of the angle between row vectors a, b and its gradients."""
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    cosv = np.clip(np.einsum("ij,ij->i", a, b) / (la * lb), -1, 1)
    th = np.arccos(cosv)
    sinv = np.sqrt(np.maximum(1 - cosv**2, 0))
    # d theta / d a = -(b/(|a||b|) - cos a/|a|^2) / sin ; (theta - theta0)/sin is finite when theta0 = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sinv > 1e-7, (th - theta0) / sinv, np.where(theta0 == 0, 1.0 + th**2 / 6, 0.0))
    coef = -(k * ratio)[:, None]
    ga = coef * (b / (la * lb)[:, None] - cosv[:, None] * a / (la**2)[:, None])
    gb = coef * (a / (la * lb)[:, None] - cosv[:, None] * b / (lb**2)[:, None])
    return 0.5 * np.sum(k * (th - theta0) ** 2), ga, gb


def structural_energy(q: np.ndarray, mesh: Mesh, material: Material, nodal_force_body: np.ndarray):
    """Total potential and its gradient with respect to the free normal displacements q."""
    p = mesh.points0.copy()
    p[mesh.free, 2] += q
    g = np.zeros_like(p)

    ev = p[mesh.edges[:, 1]] - p[mesh.edges[:, 0]]
    el = np.linalg.norm(ev, axis=1)
    ks = material.membrane_E * material.membrane_t
    e_total = 0.5 * ks * np.sum((el - mesh.edge_len0) ** 2)
    ge = (ks * (el - mesh.edge_len0) / el)[:, None] * ev
    np.add.at(g, mesh.edges[:, 1], ge)
    np.add.at(g, mesh.edges[:, 0], -ge)

    f = mesh.faces
    u, v = p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]
    nrm = np.cross(u, v)
    d_m = material.membrane_E * material.membrane_t**3 / (12 * (1 - material.membrane_nu**2))
    h = mesh.hinges
    e_h, g1, g2 = _angle_energy_grad(nrm[h[:, 0]], nrm[h[:, 1]], np.full(len(h), d_m), mesh.hinge_angle0)
    e_total += e_h
    gn = np.zeros_like(nrm)
    np.add.at(gn, h[:, 0], g1)
    np.add.at(gn, h[:, 1], g2)
    gu, gv = np.cross(v, gn), np.cross(gn, u)               # n = u x v
    np.add.at(g, f[:, 1], gu)
    np.add.at(g, f[:, 2], gv)
    np.add.at(g, f[:, 0], -gu - gv)

    sv = p[mesh.seg[:, 1]] - p[mesh.seg[:, 0]]
    sl = np.linalg.norm(sv, axis=1)
    ka = mesh.seg_EA / mesh.seg_len0
    e_total += 0.5 * np.sum(ka * (sl - mesh.seg_len0) ** 2)
    gs = (ka * (sl - mesh.seg_len0) / sl)[:, None] * sv
    np.add.at(g, mesh.seg[:, 1], gs)
    np.add.at(g, mesh.seg[:, 0], -gs)

    t = mesh.triples
    a, b = p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 1]]
    e_b, ga, gb = _angle_energy_grad(a, b, mesh.triple_EI / mesh.triple_len0, mesh.triple_angle0)
    e_total += e_b
    np.add.at(g, t[:, 0], -ga)
    np.add.at(g, t[:, 1], ga - gb)
    np.add.at(g, t[:, 2], gb)

    e_total -= np.sum(nodal_force_body[mesh.free, 2] * q)
    grad = g[mesh.free, 2] - nodal_force_body[mesh.free, 2]
    return e_total, grad


def solve_structure(mesh, material, nodal_force_body, q0, solver: Solver):
    """Minimise the total potential by a damped Newton iteration.

    The Hessian is a central difference of the analytic gradient (the problem has a few tens of
    unknowns). The MATLAB original used fminunc quasi-Newton with numerical gradients; the membrane
    stretching terms make the problem stiff, and Newton reaches gradients near round-off where the
    quasi-Newton line search stalls.
    """
    q = np.array(q0, float)
    n = len(q)
    if n == 0:
        return q, True
    e, g = structural_energy(q, mesh, material, nodal_force_body)
    h_step = 1e-7
    for _ in range(solver.max_bfgs_iter):
        gmax = float(np.max(np.abs(g)))
        if gmax < solver.gtol:
            return q, True
        hess = np.empty((n, n))
        for i in range(n):
            d = np.zeros(n)
            d[i] = h_step
            hess[:, i] = (structural_energy(q + d, mesh, material, nodal_force_body)[1]
                          - structural_energy(q - d, mesh, material, nodal_force_body)[1]) / (2 * h_step)
        hess = 0.5 * (hess + hess.T)
        lam, accepted = 0.0, False
        scale = float(np.mean(np.abs(np.diag(hess)))) + 1e-12
        for _ in range(30):
            try:
                step = -np.linalg.solve(hess + lam * scale * np.eye(n), g)
            except np.linalg.LinAlgError:
                step = None
            if step is not None and step @ g < 0:
                t = 1.0
                for _ in range(12):
                    e_new, g_new = structural_energy(q + t * step, mesh, material, nodal_force_body)
                    if e_new <= e + 1e-4 * t * (step @ g) or float(np.max(np.abs(g_new))) < gmax:
                        accepted = True
                        break
                    t *= 0.5
                if accepted:
                    break
            lam = max(10 * lam, 1e-6)
        if not accepted:
            return q, gmax < 1e3 * solver.gtol
        q, e, g = q + t * step, e_new, g_new
    return q, float(np.max(np.abs(g))) < 1e3 * solver.gtol


# ----------------------------------------------------------------------------------------------
# Coupled phase and cycle solutions
# ----------------------------------------------------------------------------------------------


def solve_phase(mesh, material, st, motion, aero, solver: Solver, q0=None, rigid=False) -> dict:
    """Coupled equilibrium at one prescribed phase. `rigid=True` holds q = 0 (deformation off)."""
    q = np.zeros(mesh.n_free) if q0 is None else q0.copy()
    pts = mesh.points0.copy()
    pts[mesh.free, 2] += q
    if rigid:
        a = aerodynamics(mesh, mesh.points0, st, motion, aero)
        return dict(q=np.zeros(mesh.n_free), aero=a, converged=True, iterations=0, max_disp=0.0, points=mesh.points0)
    prev, converged, it = None, False, 0
    for it in range(1, solver.max_iter + 1):
        a = aerodynamics(mesh, pts, st, motion, aero)
        q_star, ok = solve_structure(mesh, material, a["nodal_body"], q, solver)
        q_new = (1 - solver.relaxation) * q + solver.relaxation * q_star
        change = float(np.max(np.abs(q_new - q))) if len(q) else 0.0
        cur = a["total_body"]
        resid = np.inf if prev is None else np.linalg.norm(cur - prev) / max(np.linalg.norm(cur), np.linalg.norm(prev), solver.force_ref)
        q, prev = q_new, cur
        pts = mesh.points0.copy()
        pts[mesh.free, 2] += q
        if it >= solver.min_iter and change <= solver.disp_tol and resid <= solver.force_tol and ok:
            converged = True
            break
    a = aerodynamics(mesh, pts, st, motion, aero)
    return dict(q=q, aero=a, converged=converged, iterations=it, max_disp=float(np.max(np.abs(q))) if len(q) else 0.0, points=pts)


def run_cycle(mesh: Mesh, material: Material = KAPTON, motion: Motion = Motion(), aero: Aero = Aero(),
              solver: Solver = Solver(), n_phase: int = 48, n_cycles: int = 3, rigid: bool = False) -> dict:
    """Quasi-static continuation over `n_cycles`; the first cycle is discarded in the means.

    Returns per-phase histories (lift, flapping-axis aerodynamic moment, maximum displacement) and
    cycle means over the retained cycles. `aero_power` is the cycle mean of -M_phi * phi_dot.
    """
    if rigid:
        n_cycles = 1
    q, rows, q_peak, peak = None, [], None, -1.0
    for cyc in range(n_cycles):
        for j in range(n_phase):
            ph = j / n_phase
            st = motion.state(ph)
            r = solve_phase(mesh, material, st, motion, aero, solver, q, rigid)
            if not r["converged"] and abs(st["phi_d"]) < 1e-10 and abs(st["theta"]) < 1e-10:
                r = solve_phase(mesh, material, st, motion, aero, solver, np.zeros(mesh.n_free), rigid)
            q = r["q"]
            if r["max_disp"] > peak:
                peak, q_peak = r["max_disp"], q.copy()
            rows.append((cyc, ph, r["aero"]["lift"], r["aero"]["moment_flap"], st["phi_d"], r["max_disp"],
                         r["converged"], r["iterations"], st["phi"], st["theta"]))
    arr = np.array(rows, float)
    keep = arr[:, 0] >= (1 if n_cycles > 1 else 0)
    return dict(
        history=arr, columns=["cycle", "phase", "lift_N", "moment_flap_Nm", "phi_dot", "max_disp_m", "converged",
                              "iterations", "phi_rad", "theta_rad"],
        mean_lift=float(arr[keep, 2].mean()), aero_power=float(np.mean(-arr[keep, 3] * arr[keep, 4])),
        max_disp=float(arr[keep, 5].max()), all_converged=bool(arr[:, 6].all()), final_q=q,
        q_at_peak=q_peak,
    )


def chordwise_slope_deg(mesh: Mesh, q: np.ndarray) -> float:
    """Largest chordwise slope of the deformed surface over the spanwise stations [deg].

    At each spanwise station (nodes sharing x) a line z = a + b y is fitted to the normal
    displacements; atan(b) is a local twist/camber-line angle induced by deformation.
    """
    z = np.zeros(len(mesh.points0))
    z[mesh.free] = q
    xs = np.round(mesh.points0[:, 0], 9)
    worst = 0.0
    for x in np.unique(xs):
        sel = xs == x
        if sel.sum() < 3 or np.ptp(mesh.points0[sel, 1]) < 1e-6:
            continue
        slope = np.polyfit(mesh.points0[sel, 1], z[sel], 1)[0]
        worst = max(worst, abs(np.degrees(np.arctan(slope))))
    return float(worst)


def with_material(mesh_builder, material: Material, **kw) -> Mesh:
    return mesh_builder(material=material, **kw)


__all__ = ["Aero", "FrameSection", "KAPTON", "Material", "Mesh", "Motion", "Solver", "TPU", "aerodynamics",
           "design_mesh", "ellipse_family_leading", "rectangular_mesh", "replace", "run_cycle", "single_tip_mesh",
           "solve_phase", "structural_energy"]
