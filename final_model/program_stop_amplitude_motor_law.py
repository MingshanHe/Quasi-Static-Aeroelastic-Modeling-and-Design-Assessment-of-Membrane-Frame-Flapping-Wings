"""Passive pitch dynamics closure with stop contact and hinge friction -- ablation (a): amplitude only, extended damping range
(AMPLITUDE_ONLY = True: theta_m from the ODE, c_theta and phase_c at the published values).

The feathering hinge is passive: a 1 mm shaft along the leading edge (body x axis, y = 0) with
mechanical stops. The published model prescribes the same tanh feathering waveform for every wing and
frequency. Here the feathering history of each (planform, membrane, frequency) condition is the steady
periodic solution of a 1-DOF pitch ODE about the shaft, written for the rigid surface of that wing's
own mesh and lumped masses:

    (I_s + I_a) psi'' = -phi'' cos(psi) (J_s + J_a) + phi'^2 sin(psi) cos(psi) (I_s + I_a)
                        - 1/2 rho phi'|phi'| S3 C_N(psi) - R2 |phi'| psi'
                        - T_f tanh(psi'/eps) - c_v psi' - M_stop(psi; psi_s)

psi is the rotation of the wing from the vertical (membrane trailing the shaft), so the physical angle of
attack is 90 deg - |psi|. The inertial terms are the exact shaft moment of the node masses (I_s = sum m y^2,
J_s = sum m y r, r = x + root offset) and of the wingmodel added-mass law (m_a = pi/4 rho c A per panel);
the translational term is the panel normal force C_N = cl cos(a) + cd sin(a) of the wingmodel coefficient
laws acting at the panel centroids (S3 = sum y r^2 A); R2 = rho pi sum y (0.75 - x0) r c A is the
rotational-circulation term of the same model, acting as aerodynamic pitch damping. Without the hinge
terms every moment scales with f^2, so the frequency dependence of the feathering comes from the hinge
friction T_f and damping c_v. The stops are a stiff one-sided spring beyond +/-psi_s.

The symmetric periodic orbit is found by damped iteration of the half-stroke map; the last cycle,
theta(t) = -psi(t) in the wingmodel convention, is mapped onto the tanh Motion: theta_m = half peak-to-peak,
phase_c = phase of the first harmonic, c_theta from the fraction of the cycle spent above theta_m/2.

Only (T_f, c_v, psi_s) are identified, on training lift. During identification the lift is a cached
rigid run_cycle grid over (theta_m, c_theta, phase_c) times the coupled/rigid deformation factor of the
condition; predictions use a rigid run_cycle at the exact ODE motion times the same factor.

Electrical input (c04): DC-motor law on the shaft-power history the wing demands. The Scotch yoke turns at
w_m = 2 pi f, so the motor torque is p_w(t)/w_m and

    P_el = P_0 + k_f f + a <p_w> + c <(p_w + J_m phi' phi'')^2> / (w_m TAU_REF)^2

(friction, back-EMF work, copper loss with a mechanism inertia J_m shared by all wings). p_w = p_aero +
p_inert + p_hinge: p_aero = -M_flap phi' of the coupled run_cycle at the identified motion (the 24-phase
coupled history is added as a deformation correction to the 96-phase rigid history), p_inert the kinetic-energy
rate of the node masses under that motion, p_hinge = c_v theta'^2 with c_v from the lift fit. (P_0, k_f, a, c)
>= 0 by weighted NNLS inside a 1-D log grid over J_m (0 to 3x the largest wing flap inertia), with
1/eta = a >= 1 so that the electrical input is never below the shaft work it supplies.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import nnls

from wingmodel import features, model as M, wings

# ablation switches
AMPLITUDE_ONLY = True       # True: only theta_m from the ODE; c_theta and phase_c at the published values
INERTIAL_FORCING = True     # False: drop the flapping-inertia (phi'' and centrifugal) shaft moments
COUPLED_AERO_POWER = True   # False: p_aero from the rigid surface only (ablation b)
HINGE_POWER = False         # False: drop c_v theta'^2 from p_w (ablation a, with MECH_INERTIA = False)
MECH_INERTIA = False        # False: J_m fixed at 0

PHI_AMP_DEG = 80.0
BASE_C, BASE_PC = 1.0 / 0.22, np.pi / 2
AERO = M.Aero()
ROOT_OFFSET = M.Motion().root_offset

# lift surrogate grid
GRID_TH = np.arange(0.0, 70.01, 5.0)                              # theta_m [deg]
GRID_C = np.array([0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 9.0, 13.0])     # c_theta
GRID_D = np.arange(-2.8, 0.81, 0.4)                               # phase_c - pi/2 [rad]
GRID_NPHASE = 64
EXACT_NPHASE = 96
COUPLED = dict(theta=50.0, n_phase=24, n_cycles=2)

# pitch ODE (nondimensional time tau = 2 pi f t)
N_HALF = 400                 # RK4 steps per half stroke (c_v up to 0.8/dt = 102 without clipping)
N_MAP_ITER = 8               # damped half-stroke map iterations
K_STOP, D_STOP = 2500.0, 50.0
LIFT_SD_FLOOR = 1.0          # mN, added in quadrature to the acquisition SD

# identification grid: T_f [N m], c_v [N m s], psi_s [deg]
TF_GRID = np.r_[0.0, np.logspace(-6.0, -2.5, 29)]
CV_GRID = np.r_[0.0, np.logspace(-9.0, -4.5, 10)]   # extended past the 1e-5 bound hit by program_2
PS_GRID = np.array([50.0])          # stop angle fixed at the mechanical limiter
PS_BOUNDS = (50.0, 50.0)

# motor law
HIST = dict(n_phase=24, n_cycles=2)   # coupled torque history at the identified motion
KE_SUB = 8                            # kinetic-energy history at KE_SUB x EXACT_NPHASE phases
POWER_SD_FLOOR = 0.1                  # W, added in quadrature to the acquisition SD
JM_REL = np.r_[0.0, np.logspace(-2, np.log10(3.0), 25)]   # J_m / max wing flap inertia (up to a few times)
A_MIN = 1.0                           # 1/eta >= 1: electrical input is at least the shaft work (energy conservation)
TIME_SAFETY = 0.8                     # fraction of the fold budget the coupled histories may reach


def _key(job):
    return "_".join(str(x) for x in job).replace("/", "-")


def _mesh(planform, membrane):
    material = wings.MEMBRANES[membrane]
    return wings.planform_mesh(planform, material), material


def _motion(freq, th, c, pc):
    return M.Motion(freq=freq, phi_amp_deg=PHI_AMP_DEG, theta_m_deg=float(th), c_theta=float(c), phase_c=float(pc))


# ------------------------------------------------------------------------------------------------
# measurement-independent physics (cached in work_dir)
# ------------------------------------------------------------------------------------------------

def _physics(job):
    kind, planform, membrane, freq = job[:4]
    mesh, material = _mesh(planform, membrane)
    t0 = time.time()
    if kind == "coupled":
        motion = _motion(freq, COUPLED["theta"], BASE_C, BASE_PC)
        cyc = M.run_cycle(mesh, material, motion, n_phase=COUPLED["n_phase"], n_cycles=COUPLED["n_cycles"])
        rig = M.run_cycle(mesh, material, motion, n_phase=COUPLED["n_phase"], rigid=True)
        return dict(lift_mN=1e3 * cyc["mean_lift"], lift_rigid_mN=1e3 * rig["mean_lift"],
                    converged=bool(cyc["all_converged"]), wall_s=time.time() - t0)
    if kind == "grid":
        vals = np.zeros((len(GRID_TH), len(GRID_C), len(GRID_D)))
        for i, th in enumerate(GRID_TH):
            for j, c in enumerate(GRID_C):
                for k, d in enumerate(GRID_D):
                    rig = M.run_cycle(mesh, material, _motion(freq, th, c, BASE_PC + d), n_phase=GRID_NPHASE, rigid=True)
                    vals[i, j, k] = 1e3 * rig["mean_lift"]
        return dict(lift_rigid_mN=vals.tolist(), wall_s=time.time() - t0)
    if kind == "exact":
        th, c, pc, use_coupled = job[4:8]
        motion = _motion(freq, th, c, pc)
        rig = M.run_cycle(mesh, material, motion, n_phase=EXACT_NPHASE, rigid=True)
        pi_i, q_i = features.inertial_features(mesh, material, membrane, motion)
        h = rig["history"]
        pa_rig = -h[:, 3] * h[:, 4]                                   # rigid shaft aero power at EXACT_NPHASE phases
        out = dict(lift_rigid_mN=1e3 * rig["mean_lift"], pa=rig["aero_power"], pi_i=pi_i, q_i=q_i,
                   pa_rigid=pa_rig.tolist())
        # kinetic-energy rate of the node masses, periodic central difference on a finer grid
        n_ke = KE_SUB * EXACT_NPHASE
        k = features.kinetic_energy_history(mesh, wings.node_masses(mesh, material, membrane), motion, n=n_ke)
        out["p_inert"] = ((np.roll(k, -1) - np.roll(k, 1)) * n_ke * freq / 2)[::KE_SUB].tolist()
        st = [motion.state(j / EXACT_NPHASE) for j in range(EXACT_NPHASE)]
        out["phi_d"] = [s["phi_d"] for s in st]
        out["phi_dd"] = [s["phi_dd"] for s in st]
        out["theta_d"] = [s["theta_d"] for s in st]
        if use_coupled:
            cyc = M.run_cycle(mesh, material, motion, n_phase=HIST["n_phase"], n_cycles=HIST["n_cycles"])
            hc = cyc["history"]
            keep = hc[:, 0] == HIST["n_cycles"] - 1                   # last (retained) cycle
            pc_c = -hc[keep, 3] * hc[keep, 4]
            stride = EXACT_NPHASE // HIST["n_phase"]
            d = pc_c - pa_rig[::stride]                               # deformation correction at the coarse phases
            ph_f = np.arange(EXACT_NPHASE) / EXACT_NPHASE
            ph_c = np.arange(HIST["n_phase"]) / HIST["n_phase"]
            out["pa_coupled"] = (pa_rig + np.interp(ph_f, ph_c, d, period=1.0)).tolist()
            out["pa_coupled_mean_coarse"] = float(pc_c.mean())
            out["hist_converged"] = bool(cyc["all_converged"])
        out["wall_s"] = time.time() - t0
        return out
    raise KeyError(kind)


def _run_jobs(jobs, num_workers, cache_dir=None):
    res, todo = {}, []
    for j in jobs:
        f = cache_dir / f"{_key(j)}.json" if cache_dir is not None else None
        if f is not None and f.is_file():
            res[j] = json.loads(f.read_text())
        else:
            todo.append(j)
    # slowest first: coupled TPU, coupled Kapton, grids, exact
    order = {"coupled": 0, "grid": 1, "exact": 2}
    todo.sort(key=lambda j: (order[j[0]], j[2] != "tpu"))
    if todo:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futs = {pool.submit(_physics, j): j for j in todo}
            for fut in as_completed(futs):
                j = futs[fut]
                r = fut.result()
                res[j] = r
                if cache_dir is not None:
                    tmp = cache_dir / f"{_key(j)}.{os.getpid()}.tmp"
                    tmp.write_text(json.dumps(r))
                    os.replace(tmp, cache_dir / f"{_key(j)}.json")
    return res, len(todo)


# ------------------------------------------------------------------------------------------------
# pitch ODE
# ------------------------------------------------------------------------------------------------

def wing_constants(planform, membrane):
    """Shaft-moment constants of the rigid surface of one wing (SI)."""
    mesh, material = _mesh(planform, membrane)
    m = wings.node_masses(mesh, material, membrane)
    p = mesh.points0
    y_n, r_n = p[:, 1], p[:, 0] + ROOT_OFFSET
    cen = p[mesh.faces].mean(axis=1)
    y_p, r_p = cen[:, 1], cen[:, 0] + ROOT_OFFSET
    A, c, x0 = mesh.area0, mesh.chord, mesh.pitch_frac
    m_a = np.pi / 4 * AERO.rho * c * A
    return dict(
        I_s=float(np.sum(m * y_n**2)), I_a=float(np.sum(m_a * y_p**2)),
        J_s=float(np.sum(m * y_n * r_n)), J_a=float(np.sum(m_a * y_p * r_p)),
        S3=float(np.sum(y_p * r_p**2 * A)), R2=float(AERO.rho * np.pi * np.sum(y_p * (0.75 - x0) * r_p * c * A)),
        I_flap=float(np.sum(m * (r_n**2 + y_n**2))), mass=float(m.sum()))


def _cn(psi):
    """Panel normal-force coefficient at physical angle of attack 90 deg - |psi| (wingmodel laws)."""
    sp, cp = np.sin(psi), np.cos(psi)
    return cp * (2 * AERO.cl_amp * sp * sp + AERO.cd_mean + AERO.cd_cos * np.cos(2 * psi))


def simulate(Jr, Ar, Rr, T, cv, ps, n_record=200):
    """Steady symmetric periodic pitch orbit for a batch of nondimensional problems (arrays of shape (B,)).

    Jr = J/I, Ar = rho S3 / (2 I), Rr = R2 / I, T = T_f / (I w^2), cv = c_v / (I w), ps = stop angle [rad].
    Time tau = w t, phi = Phi sin(tau). Returns the recorded last cycle psi (n_record, B) starting at the
    stroke reversal tau = -pi/2, and the change of the reversal state over the last map iteration.
    """
    Phi = np.deg2rad(PHI_AMP_DEG)
    dt = np.pi / N_HALF
    eps = np.maximum(0.05, T * dt)                   # friction regularisation inside the RK4 stability range
    cvc = np.minimum(cv, 0.8 / dt)
    gi = 1.0 if INERTIAL_FORCING else 0.0

    def f(tau, psi, w):
        s, c = np.sin(tau), np.cos(tau)
        cp = np.cos(psi)
        acc = (gi * (Phi * s * cp * Jr + Phi * Phi * c * c * np.sin(psi) * cp)
               - Ar * Phi * Phi * c * abs(c) * _cn(psi) - Rr * Phi * abs(c) * w
               - T * np.tanh(w / eps) - cvc * w)
        pen = np.abs(psi) - ps
        contact = pen > 0
        return acc - np.where(contact, np.sign(psi) * K_STOP * pen + D_STOP * w, 0.0)

    def half(psi, w, record=None):
        tau = -np.pi / 2
        for n in range(N_HALF):
            k1p, k1w = w, f(tau, psi, w)
            k2p, k2w = w + 0.5 * dt * k1w, f(tau + 0.5 * dt, psi + 0.5 * dt * k1p, w + 0.5 * dt * k1w)
            k3p, k3w = w + 0.5 * dt * k2w, f(tau + 0.5 * dt, psi + 0.5 * dt * k2p, w + 0.5 * dt * k2w)
            k4p, k4w = w + dt * k3w, f(tau + dt, psi + dt * k3p, w + dt * k3w)
            if record is not None:
                record.append(psi.copy())
            psi = psi + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
            w = w + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
            tau += dt
        return psi, w

    B = len(Jr)
    psi, w = np.zeros(B), np.zeros(B)                # wing hanging vertical at the stroke reversal
    change = np.zeros(B)
    for _ in range(N_MAP_ITER):                      # symmetric orbit: s = -Half(s), damped iteration
        pe, we = half(psi, w)
        new_p, new_w = 0.5 * (psi - pe), 0.5 * (w - we)
        change = np.abs(new_p - psi)
        psi, w = new_p, new_w
    rec = []
    pe, we = half(psi, w, rec)
    rec2 = []
    half(-pe, -we, rec2)                             # second half by symmetry of the forcing
    full = np.array(rec + [-x for x in rec2])        # (2 N_HALF, B)
    stride = max(1, full.shape[0] // n_record)
    return full[::stride], change


_C_TAB = np.linspace(0.05, 20.0, 800)
_FRAC_TAB = 1 - (2 / np.pi) * np.arcsin(np.clip(np.arctanh(np.tanh(_C_TAB) / 2) / _C_TAB, 0, 1))


def tanh_params(psi_rec):
    """Map the recorded cycle theta = -psi onto (theta_m [deg], c_theta, phase_c) of the tanh Motion."""
    n = psi_rec.shape[0]
    tau = -np.pi / 2 + 2 * np.pi * np.arange(n) / n
    th = -psi_rec
    mean = th.mean(axis=0)
    amp = 0.5 * (th.max(axis=0) - th.min(axis=0))
    a = (th * np.cos(tau)[:, None]).sum(axis=0)
    b = (th * np.sin(tau)[:, None]).sum(axis=0)
    pc = np.arctan2(a, b)
    frac = (np.abs(th - mean) > 0.5 * amp).mean(axis=0)
    c = np.interp(frac, _FRAC_TAB, _C_TAB)
    small = np.rad2deg(amp) < 0.5
    pc = np.where(small, BASE_PC, pc)
    c = np.where(small, BASE_C, c)
    if AMPLITUDE_ONLY:
        pc, c = np.full_like(pc, BASE_PC), np.full_like(c, BASE_C)
    # keep the motion inside the lift-surrogate grid so that identification and prediction agree
    pc = BASE_PC + np.clip(_wrap(pc - BASE_PC), GRID_D[0], GRID_D[-1])
    c = np.clip(c, GRID_C[0], GRID_C[-1])
    th_deg = np.clip(np.rad2deg(amp), GRID_TH[0], GRID_TH[-1])
    return th_deg, c, pc


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


class Condition:
    def __init__(self, cond, wc, grid, coupled):
        self.cond = cond
        freq = cond[2]
        w = 2 * np.pi * freq
        I = wc["I_s"] + wc["I_a"]
        self.I, self.Iw2, self.Iw = I, I * w * w, I * w
        self.Jr = (wc["J_s"] + wc["J_a"]) / I
        self.Ar = 0.5 * AERO.rho * wc["S3"] / I
        self.Rr = wc["R2"] / I
        self.interp = RegularGridInterpolator((GRID_TH, GRID_C, GRID_D), np.asarray(grid["lift_rigid_mN"]))
        lr = coupled["lift_rigid_mN"]
        self.ratio = coupled["lift_mN"] / lr if abs(lr) > 1e-6 else 1.0

    def lift_surrogate(self, th, c, pc):
        d = np.clip(_wrap(np.asarray(pc) - BASE_PC), GRID_D[0], GRID_D[-1])
        pts = np.column_stack([np.clip(th, GRID_TH[0], GRID_TH[-1]), np.clip(c, GRID_C[0], GRID_C[-1]), d])
        return self.ratio * self.interp(pts)


def _ode_chunk(args):
    rec, change = simulate(*args)
    return tanh_params(rec) + (change,)


def run_ode(conds, tf, cv, ps_deg, chunk=3000, workers=1):
    """Integrate the pitch ODE for every condition (N) and every parameter set (P): arrays (P, N)."""
    tf, cv, ps = (np.atleast_1d(np.asarray(x, float)) for x in (tf, cv, ps_deg))
    P, N = len(tf), len(conds)
    Jr = np.tile([c.Jr for c in conds], P)
    Ar = np.tile([c.Ar for c in conds], P)
    Rr = np.tile([c.Rr for c in conds], P)
    T = np.repeat(tf, N) / np.tile([c.Iw2 for c in conds], P)
    CV = np.repeat(cv, N) / np.tile([c.Iw for c in conds], P)
    PS = np.deg2rad(np.repeat(ps, N))
    th, cc, pc, ch = (np.zeros(P * N) for _ in range(4))
    slices = [slice(s, s + chunk) for s in range(0, P * N, chunk)]
    args = [(Jr[sl], Ar[sl], Rr[sl], T[sl], CV[sl], PS[sl]) for sl in slices]
    if workers > 1 and len(slices) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(slices))) as pool:
            outs = list(pool.map(_ode_chunk, args))
    else:
        outs = [_ode_chunk(a) for a in args]
    for sl, o in zip(slices, outs):
        th[sl], cc[sl], pc[sl], ch[sl] = o
    return th.reshape(P, N), cc.reshape(P, N), pc.reshape(P, N), ch.reshape(P, N)


# ------------------------------------------------------------------------------------------------

def fit_predict(ctx):
    t_start = time.time()
    cid = lambda r: (r["planform"], r["membrane"], r["freq_Hz"])  # noqa: E731
    conds = sorted({cid(r) for r in ctx.train + ctx.test})
    cache_dir = Path(ctx.work_dir) / "c03_physics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(k,) + c for c in conds for k in ("coupled", "grid")]
    res, n_new = _run_jobs(jobs, ctx.num_workers, cache_dir)
    print(f"[c03] physics: {n_new} new jobs, {time.time() - t_start:.1f} s", flush=True)

    wcs = {w: wing_constants(*w) for w in sorted({c[:2] for c in conds})}
    for w, wc in wcs.items():
        print(f"[c03] wing {w[0]:12s} {w[1]:6s} I_s={wc['I_s']:.3e} I_a={wc['I_a']:.3e} J_s={wc['J_s']:.3e} "
              f"J_a={wc['J_a']:.3e} S3={wc['S3']:.3e} R2={wc['R2']:.3e} mass={1e6 * wc['mass']:.1f} mg", flush=True)
    C = {c: Condition(c, wcs[c[:2]], res[("grid",) + c], res[("coupled",) + c]) for c in conds}

    train_c = [C[cid(r)] for r in ctx.train]
    y = np.array([r["lift_mean_mN"] for r in ctx.train])
    wt = 1.0 / (np.array([r["lift_sd_mN"] for r in ctx.train]) ** 2 + LIFT_SD_FLOOR ** 2)

    def losses(tf, cv, ps):
        th, cc, pc, _ = run_ode(train_c, tf, cv, ps, workers=ctx.num_workers)
        pred = np.stack([train_c[i].lift_surrogate(th[:, i], cc[:, i], pc[:, i]) for i in range(len(train_c))], axis=1)
        return np.sum(wt * (pred - y) ** 2, axis=1)

    # coarse grid over (T_f, c_v, psi_s), then two local refinements in log space
    G = np.array([(a, b, s) for a in TF_GRID for b in CV_GRID for s in PS_GRID])
    L = losses(G[:, 0], G[:, 1], G[:, 2])
    print(f"[c03] coarse grid {len(G)} sets: best loss {L.min():.2f} at T_f={G[L.argmin(), 0]:.3e} "
          f"c_v={G[L.argmin(), 1]:.3e} psi_s={G[L.argmin(), 2]:.1f} ({time.time() - t_start:.1f} s)", flush=True)
    allG, allL = [G], [L]
    for step_tf, step_cv, step_ps in ((0.0625, 0.25, 2.5), (0.0208, 0.0833, 0.8)):
        cand = []
        for i in np.argsort(L)[:4]:
            a, b, s = G[i]
            la = np.log10(max(a, 1e-6)) + step_tf * np.arange(-2, 3)
            lb = np.minimum(np.log10(max(b, 1e-9)) + step_cv * np.arange(-2, 3), np.log10(CV_GRID[-1]))
            ss = np.clip(s + step_ps * np.arange(-2, 3), *PS_BOUNDS)
            cand += [(10 ** u, 10 ** v, q) for u in la for v in lb for q in ss]
        G = np.unique(np.array(cand), axis=0)
        L = losses(G[:, 0], G[:, 1], G[:, 2])
        allG.append(G)
        allL.append(L)
        print(f"[c03] refine {len(G)} sets: best loss {L.min():.2f} ({time.time() - t_start:.1f} s)", flush=True)
    G, L = np.vstack(allG), np.concatenate(allL)
    tf, cv, ps = G[L.argmin()]

    # motion of every condition under the identified hinge
    th, cc, pc, ch = (x[0] for x in run_ode([C[c] for c in conds], tf, cv, ps))
    mot = {c: (float(th[i]), float(cc[i]), float(pc[i]), float(ch[i])) for i, c in enumerate(conds)}
    # Video-consistent kinematics: every wing feathers to the +/-50 deg limiter (no hinge model); motor law unchanged.
    mot = {c: (50.0,) + v[1:] for c, v in mot.items()}

    # coupled torque histories only if their estimated cost (from the cached coupled jobs) fits the budget
    use_coupled = COUPLED_AERO_POWER
    if use_coupled:
        walls = np.array([res[("coupled",) + c]["wall_s"] for c in conds])
        est = 1.5 * max(walls.max(), walls.sum() / max(1, ctx.num_workers))
        elapsed = time.time() - t_start
        if elapsed + est > TIME_SAFETY * ctx.time_budget_s:
            use_coupled = False
        print(f"[c04] coupled histories: estimated {est:.0f} s at {elapsed:.0f} s elapsed -> use_coupled={use_coupled}",
              flush=True)
    ex_jobs = [("exact",) + c + tuple(round(v, 6) for v in mot[c][:3]) + (use_coupled,) for c in conds]
    ex, n_new = _run_jobs(ex_jobs, ctx.num_workers, cache_dir)
    print(f"[c04] exact/history jobs: {n_new} new, {time.time() - t_start:.1f} s", flush=True)
    E = {j[1:4]: ex[j] for j in ex_jobs}
    lift = {c: C[c].ratio * E[c]["lift_rigid_mN"] for c in conds}

    # DC-motor law on the shaft-power history
    H = {}
    for c in conds:
        e = E[c]
        pa = np.asarray(e["pa_coupled"] if use_coupled else e["pa_rigid"])
        ph = cv * np.asarray(e["theta_d"]) ** 2 if HINGE_POWER else np.zeros_like(pa)
        H[c] = dict(pw=pa + np.asarray(e["p_inert"]) + ph, pa=pa, ph=ph,
                    pdd=np.asarray(e["phi_d"]) * np.asarray(e["phi_dd"]), wm=2 * np.pi * c[2])
    i_ref = max(wc["I_flap"] for wc in wcs.values())
    jm_grid = i_ref * (JM_REL if MECH_INERTIA else np.zeros(1))

    def design_row(c, jm):
        h = H[c]
        return [1.0, float(c[2]), float(h["pw"].mean()),
                float(np.mean((h["pw"] + jm * h["pdd"]) ** 2)) / (h["wm"] * features.TAU_REF) ** 2]

    yp = np.array([r["power_mean_W"] for r in ctx.train])
    swp = 1.0 / np.sqrt(np.array([r["power_sd_W"] for r in ctx.train]) ** 2 + POWER_SD_FLOOR ** 2)
    best = None
    for jm in jm_grid:
        X = np.array([design_row(cid(r), jm) for r in ctx.train])
        Xw = X * swp[:, None]
        sc = np.linalg.norm(Xw, axis=0)
        sc[sc == 0] = 1.0
        # a = A_MIN + a', a' >= 0: fit the input in excess of A_MIN x shaft work
        b, rn = nnls(Xw / sc, (yp - A_MIN * X[:, 2]) * swp)
        b = b / sc
        b[2] += A_MIN
        if best is None or rn < best[2]:
            best = (jm, b, rn)
    jm, coef, rn = best
    ppred = {c: float(np.dot(coef, design_row(c, jm))) for c in conds}

    out = [dict(wing=r["wing"], freq_Hz=r["freq_Hz"], lift_mN=float(lift[cid(r)]),
                power_W=float(max(ppred[cid(r)], 1e-3))) for r in ctx.test]
    p_train_rmse = float(np.sqrt(np.mean((np.array([ppred[cid(r)] for r in ctx.train]) - yp) ** 2)))
    print(f"[c04] fold={getattr(ctx, 'fold', '?')} motor law: P_0={coef[0]:.4f} W k_f={coef[1]:.5f} W/Hz "
          f"a(1/eta)={coef[2]:.4f} c_R={coef[3]:.5f} W J_m={jm:.3e} kg m^2 (J_m/I_flap_max={jm / i_ref:.3f}, "
          f"I_flap_max={i_ref:.3e}) wloss={rn ** 2:.3f} train_power_rmse={p_train_rmse:.4f} W "
          f"use_coupled={use_coupled} HINGE={HINGE_POWER} MECH_INERTIA={MECH_INERTIA}", flush=True)
    for c in conds:
        h, row_ = H[c], design_row(c, jm)
        print(f"[c04] {c[0]:12s} {c[1]:6s} {c[2]:2d} Hz <p_aero>={h['pa'].mean():.4f} rigid={E[c]['pa']:.4f} "
              f"<p_hinge>={h['ph'].mean():.4f} <p_w>={row_[2]:.4f} F2={row_[3]:.3f} P_pred={ppred[c]:.4f} W "
              f"terms=[{coef[0]:.3f},{coef[1] * c[2]:.3f},{coef[2] * row_[2]:.3f},{coef[3] * row_[3]:.3f}] "
              f"hist_conv={E[c].get('hist_converged', None)}", flush=True)
    coef_log = np.r_[coef, jm]

    train_rmse = float(np.sqrt(np.mean((np.array([lift[cid(r)] for r in ctx.train]) - y) ** 2)))
    sur_rmse = float(np.sqrt(np.mean((np.array([C[cid(r)].lift_surrogate([mot[cid(r)][0]], [mot[cid(r)][1]],
                                                                          [mot[cid(r)][2]])[0] for r in ctx.train]) - y) ** 2)))
    print(f"[c03] fold={getattr(ctx, 'fold', '?')} T_f={tf:.4e} N m c_v={cv:.4e} N m s psi_s={ps:.2f} deg "
          f"loss={L.min():.2f} train_lift_rmse exact={train_rmse:.2f} surrogate={sur_rmse:.2f} mN "
          f"motor_coef[P0,kf,a,c,Jm]={coef_log.tolist()} AMPLITUDE_ONLY={AMPLITUDE_ONLY} INERTIAL={INERTIAL_FORCING} "
          f"wall={time.time() - t_start:.1f}s", flush=True)
    for c in conds:
        role = "test" if any(cid(r) == c for r in ctx.test) else "train"
        t, k, p, chg = mot[c]
        print(f"[c03] {role:5s} {c[0]:12s} {c[1]:6s} {c[2]:2d} Hz theta_m={t:5.1f} c_theta={k:5.2f} "
              f"phase_c-pi/2={_wrap(p - BASE_PC):+.3f} T_nd={tf / C[c].Iw2:.3f} c_nd={cv / C[c].Iw:.3f} "
              f"lift={lift[c]:6.2f} ratio={C[c].ratio:.4f} map_change={np.rad2deg(chg):.3f}deg "
              f"Pa={E[c]['pa']:.4f} Pi={E[c]['pi_i']:.4f} Q={E[c]['q_i']:.3f}", flush=True)
    return out
