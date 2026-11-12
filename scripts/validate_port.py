"""Check the Python port against the stored MATLAB outputs (36 cases, 144 phases each)."""
import csv, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wing_lib"))
from wingmodel import model as M

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "reference" / "geometry"

def ref_mesh(shape, material):
    if shape == "rectangular":
        return M.rectangular_mesh(material=material)
    lead = np.loadtxt(GEO / f"{shape}_leading_edge_m.csv", delimiter=",", skiprows=1)
    man = json.loads((GEO / "planform_manifest.json").read_text())[shape]
    return M.single_tip_mesh(lead, man["root_chord_mm"] * 1e-3, material=material, name=shape)

def one_case(job):
    mat_name, shape, f = job
    mat = M.KAPTON if mat_name == "kapton" else M.TPU
    mesh = ref_mesh(shape, mat)
    ref = np.genfromtxt(ROOT / "reference" / "matlab_proposed_48" / f"{shape}_{mat_name}_{f:02d}Hz.csv", delimiter=",", names=True)
    t0 = time.time()
    r = M.run_cycle(mesh, mat, M.Motion(freq=f))
    h = r["history"]
    keep = ref["cycle"] >= 2
    mref = ref["lift_N"][keep].mean() * 1e3; mpy = r["mean_lift"] * 1e3
    rms = np.sqrt(np.mean((h[:, 2] - ref["lift_N"]) ** 2)) * 1e3
    return (shape, mat_name, f, mref, mpy, mpy - mref, rms, ref["max_displacement_m"][keep].max() * 1e3, r["max_disp"] * 1e3, r["all_converged"], time.time() - t0)

def main(freqs):
    # geometry and analytic-gradient checks
    for shape in ("rectangular", "triangular", "curve"):
        m = ref_mesh(shape, M.KAPTON)
        P = np.loadtxt(GEO / f"{shape}_nodes.csv", delimiter=","); F = np.loadtxt(GEO / f"{shape}_faces.csv", delimiter=",", dtype=int) - 1
        c = np.loadtxt(GEO / f"{shape}_panel_chord.csv"); x = np.loadtxt(GEO / f"{shape}_panel_pitch_fraction.csv")
        print(f"{shape}: nodes {np.abs(m.points0-P).max():.1e} faces {int(np.abs(m.faces-F).max())} chord {np.abs(m.chord-c).max():.1e} pitch {np.abs(m.pitch_frac-x).max():.1e} area {m.area*1e6:.2f} mm2")
        rng = np.random.default_rng(0); q = rng.normal(0, 5e-4, m.n_free); fb = rng.normal(0, 1e-3, m.points0.shape)
        e, g = M.structural_energy(q, m, M.KAPTON, fb); num = np.zeros_like(g)
        for i in range(len(q)):
            d = np.zeros_like(q); d[i] = 1e-8
            num[i] = (M.structural_energy(q + d, m, M.KAPTON, fb)[0] - M.structural_energy(q - d, m, M.KAPTON, fb)[0]) / 2e-8
        print(f"   gradient check: max rel err {np.abs(g-num).max()/np.abs(num).max():.2e}")
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(mn, sh, f) for mn in ("kapton", "tpu") for sh in ("rectangular", "triangular", "curve") for f in freqs]
    with ProcessPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(one_case, jobs))
    for r in rows:
        print("%-11s %-6s %2d Hz  mean MATLAB %8.4f  Python %8.4f  diff %+.4f mN  phase-RMS %.4f mN  maxdisp %.3f/%.3f mm conv=%s  %.1fs" % r, flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "port_validation.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["shape", "material", "freq_Hz", "matlab_mean_mN", "python_mean_mN", "diff_mN", "phase_rms_mN", "matlab_maxdisp_mm", "python_maxdisp_mm", "python_converged", "seconds"]); w.writerows(rows)
    d = np.array([r[5] for r in rows]); print("max |mean diff| = %.4f mN, max phase RMS = %.4f mN" % (np.abs(d).max(), max(r[6] for r in rows)))

if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or list(range(10, 16)))
