"""Rebuild every experimental number from the raw acquisitions (no Origin sheets, no manual edits).

Protocol (identical for all 162 acquisitions, fixed before looking at outcomes; it follows the
authors' own 2026-09-19 raw reprocessing):
  * force: raw 1200 Hz "Filter OFF" Fz, signed;
  * window: complete shaft revolutions (encoder gates) inside 1-4 s of each acquisition;
  * run lift = mean of the per-cycle means; run power = window-weighted mean of the logged
    500 ms power windows with 1000 <= elapsed_ms <= 4000;
  * condition statistics over the three technical repeats: mean, SD (ddof=1), 95 % t interval.
Outputs: measurements/runs.csv, measurements/conditions.csv, measurements/cycle_waveforms.csv
"""
import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
WINGS = {  # label: (folder, planform, membrane)
    "Rect-K": ("data_0831/0831_rect_capton", "rectangular", "kapton"),
    "Rect-T": ("data_0831/0831_rect_tpu", "rectangular", "tpu"),
    "Tri-K": ("data_0831/0831_tri_capton", "triangular", "kapton"),
    "Tri-T": ("data_0831/0831_tri_tpu", "triangular", "tpu"),
    "Curve-K": ("data_0831/0831_curve_capton", "curve", "kapton"),
    "Curve-T": ("data_0831/0831_curve_tpu", "curve", "tpu"),
    "Bone": ("data_0831/0831_bone_capton", "frame-only", "none"),
    "Opt-K": ("0910_curve_optimized_capton", "opt", "kapton"),
    "S-Opt-K": ("0911_curve_s_optimized_capton", "s-opt", "kapton"),
}
NAME = re.compile(r"_(?P<rep>\d\d)_(?P<f>\d+)Hz_(?P<v>[\dp]+)V_(?P<i>[\dp]+)A")
NBIN = 96


def gates(enc_t, enc_deg, window=(1.0, 4.0)):
    ang = -enc_deg
    lo, hi = np.interp(window, enc_t, ang)
    revs = np.arange(np.ceil(lo / 360), np.floor(hi / 360) + 1)
    return np.interp(360 * revs, ang, enc_t)


def piecewise_integrals(tm, y, bounds):
    w = np.diff(tm)
    prim = np.r_[0.0, np.cumsum(w * (y[:-1] + y[1:]) / 2)]
    idx = np.clip(np.searchsorted(tm, bounds, side="right") - 1, 0, len(tm) - 2)
    off = bounds - tm[idx]
    slope = np.diff(y)[idx] / w[idx]
    return np.diff(prim[idx] + off * y[idx] + 0.5 * slope * off**2)


def stats(x):
    x = np.asarray(x, float)
    n = len(x)
    sd = x.std(ddof=1)
    return x.mean(), sd, student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)


def main():
    runs, waves = [], {}
    for wing, (folder, planform, membrane) in WINGS.items():
        files = sorted((RAW / folder).glob("*/*.csv.csv"))
        assert len(files) == 18, (wing, len(files))
        for path in files:
            m = NAME.search(path.name)
            f, rep = int(m["f"]), int(m["rep"])
            stem = path.name.removesuffix(".csv.csv")
            assert b"Filter,OFF" in path.read_bytes()[:200]
            tm, fz = np.loadtxt(path, delimiter=",", skiprows=4, usecols=(0, 3), encoding="cp932").T
            enc = np.genfromtxt(path.with_name(stem + "_encoder.csv"), delimiter=",", names=True)
            g = gates(enc["t_s"], enc["shaft_deg"])
            cyc = 1000 * piecewise_integrals(tm, fz, g) / np.diff(g)
            pw = np.genfromtxt(path.with_name(stem + "_power.csv"), delimiter=",", names=True)
            sel = (pw["used"] == 1) & (pw["elapsed_ms"] >= 1000) & (pw["elapsed_ms"] <= 4000)
            power = np.average(pw["power_w"][sel], weights=pw["window_ms"][sel])
            label_power = float(m["v"].replace("p", ".")) * float(m["i"].replace("p", "."))
            # Phase-averaged waveform on the shaft-revolution phase. Force and encoder clocks are not
            # hardware-synchronised; the phase origin is the encoder zero, not a measured stroke reversal.
            ph_edges = np.linspace(0, 1, NBIN + 1)
            wave = np.zeros(NBIN)
            for a, b in zip(g[:-1], g[1:]):
                wave += 1000 * piecewise_integrals(tm, fz, a + (b - a) * ph_edges) / ((b - a) / NBIN)
            wave /= len(g) - 1
            waves.setdefault((wing, f), []).append(wave)
            tail = 1000 * fz[tm > tm[-1] - 0.3].mean()
            runs.append(dict(
                wing=wing, planform=planform, membrane=membrane, freq_nominal_Hz=f, replicate=rep,
                freq_actual_Hz=round(1 / np.diff(g).mean(), 4), n_cycles=len(cyc), lift_mN=round(cyc.mean(), 5),
                lift_cycle_sd_mN=round(cyc.std(ddof=1), 5), power_W=round(power, 6),
                label_power_W=round(label_power, 4), ratio_mN_per_W=round(cyc.mean() / power, 5),
                post_stop_offset_mN=round(tail, 3), source=str(path.relative_to(RAW)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()[:16]))
    with open(ROOT / "measurements" / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    conds = []
    for wing in WINGS:
        for f in range(10, 16):
            rs = [r for r in runs if r["wing"] == wing and r["freq_nominal_Hz"] == f]
            assert len(rs) == 3
            L, P, G = (stats([r[k] for r in rs]) for k in ("lift_mN", "power_W", "ratio_mN_per_W"))
            conds.append(dict(
                wing=wing, planform=rs[0]["planform"], membrane=rs[0]["membrane"], freq_Hz=f, n=3,
                freq_actual_Hz=round(np.mean([r["freq_actual_Hz"] for r in rs]), 4),
                lift_mean_mN=round(L[0], 4), lift_sd_mN=round(L[1], 4), lift_ci95_mN=round(L[2], 4),
                power_mean_W=round(P[0], 5), power_sd_W=round(P[1], 5), power_ci95_W=round(P[2], 5),
                ratio_mean_mN_per_W=round(G[0], 4), ratio_sd=round(G[1], 4), ratio_ci95=round(G[2], 4),
                label_power_mean_W=round(np.mean([r["label_power_W"] for r in rs]), 4)))
    with open(ROOT / "measurements" / "conditions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(conds[0]))
        w.writeheader()
        w.writerows(conds)
    with open(ROOT / "measurements" / "cycle_waveforms.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wing", "freq_Hz", "phase", "force_mean_mN", "force_sd_mN"])
        for (wing, f), ws in waves.items():
            a = np.array(ws)
            for k in range(NBIN):
                w.writerow([wing, f, round((k + 0.5) / NBIN, 5), round(a[:, k].mean(), 3), round(a[:, k].std(ddof=1), 3)])
    print("runs", len(runs), "conditions", len(conds))
    print("wing      " + "".join(f"{f:>22d} Hz" for f in range(10, 16)))
    for wing in WINGS:
        cs = [c for c in conds if c["wing"] == wing]
        print("%-8s L " % wing + "".join("%13.2f ± %5.2f mN" % (c["lift_mean_mN"], c["lift_sd_mN"]) for c in cs))
        print("%-8s P " % "" + "".join("%13.3f ± %5.3f W " % (c["power_mean_W"], c["power_sd_W"]) for c in cs))
        print("%-8s G " % "" + "".join("%13.2f ± %5.2f    " % (c["ratio_mean_mN_per_W"], c["ratio_sd"]) for c in cs))


if __name__ == "__main__":
    main()
