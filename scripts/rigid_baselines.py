"""Rigid quasi-steady baselines (Sane-Dickinson form; Flappy normal-force form) for all eight wings,
at the original 25 deg feathering amplitude and at the 50 deg limiter. Nothing is fitted.
Adapted from the authors' baseline adapters (geometry and force functions unchanged in algebra)."""
import csv, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wing_lib"))
from wingmodel import wings
RHO = 1.225

def kinematics(f, amp_deg, n=1536, C=1 / 0.22):
    u = 2 * np.pi * np.arange(n) / n; w = 2 * np.pi * f
    A = np.deg2rad(amp_deg) / np.tanh(C); x = C * np.cos(u); h = np.tanh(x); s = 1 - h * h
    xd = -C * w * np.sin(u); xdd = -C * w * w * np.cos(u)
    k = dict(phi_rad=np.deg2rad(80) * np.sin(u), phi_dot_rad_s=np.deg2rad(80) * w * np.cos(u), phi_ddot_rad_s2=-np.deg2rad(80) * w * w * np.sin(u),
             theta_rad=A * h, theta_dot_rad_s=A * s * xd, theta_ddot_rad_s2=A * s * (xdd - 2 * h * xd * xd))
    return {name: (-v if name.startswith("phi") else v + np.pi / 2 if name == "theta_rad" else v) for name, v in k.items()}

def rotation(phi, theta):
    cp, sp, ct, st = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta)
    return np.array([[cp, sp * st, sp * ct], [0, ct, -st], [-sp, cp * st, cp * ct]])

def bounds(nodes, faces, x):
    vals = []
    for tri in nodes[faces]:
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            if abs(b[0] - a[0]) > 1e-12 and min(a[0], b[0]) - 1e-10 <= x <= max(a[0], b[0]) + 1e-10:
                vals.append(a[1] + (x - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))
    return min(vals), max(vals)

def geometry(planform, n=256, root_offset=0.006):
    mesh = wings.planform_mesh(planform); nodes, faces = mesh.points0, mesh.faces
    edges = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), n + 1); mid = (edges[1:] + edges[:-1]) / 2
    lim = np.array([bounds(nodes, faces, x) for x in mid])
    return dict(mid=mid + root_offset, chord=lim[:, 1] - lim[:, 0], dr=np.diff(edges), leading=lim[:, 0], midchord=lim.mean(1))

def sd_lift(g, k):
    c, r, dr, ym = g["chord"], g["mid"], g["dr"], g["midchord"]; area = c * dr; out = []
    for j in range(len(k["phi_rad"])):
        th, pdot, tdot = k["theta_rad"][j], k["phi_dot_rad_s"][j], k["theta_dot_rad_s"][j]
        R = rotation(k["phi_rad"][j], th); span, chat, normal = R[:, 0], R[:, 1], R[:, 2]
        points = np.column_stack([r, ym, np.zeros(len(r))]) @ R.T
        flow = np.sign(pdot) * np.array([np.sin(k["phi_rad"][j]), 0, np.cos(k["phi_rad"][j])]); speed = abs(pdot) * r; lift = np.cross(span, flow)
        a = np.arctan2(flow @ normal, flow @ chat); acute = np.rad2deg(np.arctan2(abs(flow @ normal), abs(flow @ chat)))
        cl = (0.225 + 1.58 * np.sin(np.deg2rad(2.13 * acute - 7.2))) * np.sign(np.sin(2 * a)); cd = 1.92 - 1.55 * np.cos(np.deg2rad(2.04 * acute - 9.82))
        ft = np.sum(0.5 * RHO * (speed**2 * area)[:, None] * (cd * flow + cl * lift), axis=0)
        fr = np.sum(-RHO * np.pi * (0.75 + g["leading"] / c) * c * c * dr * speed * tdot * np.sign(lift @ normal)) * normal
        omf = pdot * np.array([0.0, 1.0, 0.0]); omt = tdot * span; om = omf + omt
        alpha = k["phi_ddot_rad_s2"][j] * np.array([0.0, 1.0, 0.0]) + k["theta_ddot_rad_s2"][j] * span + np.cross(omf, omt)
        acc = np.cross(alpha, points) + np.cross(om, np.cross(om, points))
        fa = np.sum(-RHO * np.pi * c * c * dr / 4 * (acc @ normal)) * normal
        out.append(-(ft[1] + fr[1] + fa[1]))
    return float(np.mean(out))

def flappy_lift(g, k):
    alpha = -k["theta_rad"] + np.sign(k["phi_dot_rad_s"]) * np.pi / 2
    cn = 1.8 * np.sin(2 * alpha) * np.cos(alpha) + (1.95 - 1.5 * np.cos(2 * alpha)) * np.sin(alpha)
    return float(np.mean(0.5 * RHO * cn * k["phi_dot_rad_s"] ** 2 * np.sum(g["mid"] ** 2 * g["chord"] * g["dr"]) * np.sin(k["theta_rad"])))

def main():
    cond = list(csv.DictReader(open(ROOT / "measurements" / "conditions.csv")))
    geo = {p: geometry(p) for p in {wings.WINGS[w][0] for w in wings.WINGS}}
    rows = []
    for r in cond:
        if r["wing"] == "Bone": continue
        g = geo[r["planform"]]
        for amp in (25, 50):
            k = kinematics(int(r["freq_Hz"]), amp)
            rows.append(dict(wing=r["wing"], freq_Hz=int(r["freq_Hz"]), theta_m_deg=amp, sd_mN=round(1e3 * sd_lift(g, k), 4), flappy_mN=round(1e3 * flappy_lift(g, k), 4),
                             measured_mN=float(r["lift_mean_mN"])))
    out = ROOT / "results" / "analyses" / "rigid_baselines.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    des = ("Opt-K", "S-Opt-K")
    for amp in (25, 50):
        for key in ("sd_mN", "flappy_mN"):
            s = [x for x in rows if x["theta_m_deg"] == amp]
            e = lambda sel: np.sqrt(np.mean([(x[key] - x["measured_mN"]) ** 2 for x in s if sel(x)]))  # noqa: E731
            meas = np.array([x["measured_mN"] for x in s])
            print("%-9s theta_m=%d: lift RMSE all %.2f ref %.2f designed %.2f | designed bias %+.1f | lift skill %.3f" % (
                key, amp, e(lambda x: True), e(lambda x: x["wing"] not in des), e(lambda x: x["wing"] in des),
                np.mean([x[key] - x["measured_mN"] for x in s if x["wing"] in des]), 1 - e(lambda x: True) / np.sqrt(np.mean(meas**2))))

if __name__ == "__main__":
    main()
