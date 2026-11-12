"""Evaluator: can a membrane-frame wing model predict measured lift and electrical input of wings it has not seen?

    python evaluate.py --program_path <program_dir>/src/program.py --results_dir <dir>

The candidate defines ``fit_predict(ctx)``. For each fold the evaluator starts a fresh child process
and gives it the measured training conditions (``ctx.train``) and the specifications of the held-out
conditions (``ctx.test``: wing label, planform, membrane, frequency; no measurements). The child
returns predicted cycle-mean lift [mN] and electrical input [W] for every held-out condition.

Folds (all measurements come from measurements/conditions.csv, rebuilt from the raw acquisitions):
  planform folds  leave one reference planform out: train on the other two planforms (Kapton and
                  TPU, 24 conditions), predict the held-out planform's two wings (12 conditions);
  design fold     train on the six reference wings (36 conditions), predict the two designed wings
                  Opt-K and S-Opt-K (12 conditions), which were fabricated and measured later.
Every wing is therefore predicted exactly once from a model that never saw its measurements.

Score on the 48 out-of-fold predictions:
  lift_skill   = max(0, 1 - RMSE_lift / RMS(measured lift))
  power_skill  = max(0, 1 - RMSE_power / RMS(measured power))
  rank_skill   = mean over the six frequencies of (Spearman rho + 1) / 2 between predicted and
                 measured lift-to-power ratio across the eight wings
  combined_score = 0.4 lift_skill + 0.3 power_skill + 0.3 rank_skill
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

FOLD_BUDGET_S = 900
GRACE_S = 60
REFERENCE_PLANFORMS = ["rectangular", "triangular", "curve"]
DESIGNED = ["Opt-K", "S-Opt-K"]
WEIGHTS = dict(lift=0.4, power=0.3, rank=0.3)


def task_dir() -> Path:
    for candidate in (os.environ.get("TASK_SOURCE_DIR"), str(Path(__file__).resolve().parent)):
        if candidate and (Path(candidate) / "wing_lib" / "wingmodel" / "model.py").is_file():
            return Path(candidate).resolve()
    raise SystemExit("cannot locate wing_lib (set TASK_SOURCE_DIR)")


def num_workers() -> int:
    return max(1, min(16, len(os.sched_getaffinity(0))))


def load_conditions() -> list[dict]:
    rows = []
    with open(task_dir() / "measurements" / "conditions.csv") as fh:
        for r in csv.DictReader(fh):
            row = {k: (v if k in ("wing", "planform", "membrane") else float(v)) for k, v in r.items()}
            row["freq_Hz"] = int(row["freq_Hz"])
            rows.append(row)
    return rows


def folds(conditions: list[dict]) -> list[dict]:
    wings = [r for r in conditions if r["wing"] != "Bone"]
    reference = [r for r in wings if r["wing"] not in DESIGNED]
    frame_only = [r for r in conditions if r["wing"] == "Bone"]
    out = []
    for planform in REFERENCE_PLANFORMS:
        out.append(dict(name=f"planform:{planform}", train=[r for r in reference if r["planform"] != planform],
                        test=[r for r in reference if r["planform"] == planform], frame_only=frame_only))
    out.append(dict(name="design", train=reference, test=[r for r in wings if r["wing"] in DESIGNED],
                    frame_only=frame_only))
    return out


TRAIN_KEYS = ["wing", "planform", "membrane", "freq_Hz", "n", "freq_actual_Hz", "lift_mean_mN", "lift_sd_mN",
              "power_mean_W", "power_sd_W", "ratio_mean_mN_per_W", "ratio_sd"]
TEST_KEYS = ["wing", "planform", "membrane", "freq_Hz"]

_CHILD = r"""
import importlib, json, sys, types
payload = json.load(open(sys.argv[1]))
mod = importlib.import_module(payload["module"])   # importable by name, so worker processes can unpickle it
ctx = types.SimpleNamespace(train=payload["train"], test=payload["test"], frame_only=payload["frame_only"],
                            fold=payload["fold"], work_dir=payload["work_dir"], num_workers=payload["num_workers"],
                            time_budget_s=payload["time_budget_s"])
out = mod.fit_predict(ctx)
json.dump(out, open(sys.argv[2], "w"))
"""


def run_fold(program: Path, fold: dict, work_dir: Path, logs: Path) -> tuple[list[dict] | None, str, float]:
    tag = fold["name"].replace(":", "_")
    payload = dict(
        module=program.stem, fold=fold["name"],
        train=[{k: r[k] for k in TRAIN_KEYS} for r in fold["train"]],
        test=[{k: r[k] for k in TEST_KEYS} for r in fold["test"]],
        frame_only=[{k: r[k] for k in TRAIN_KEYS} for r in fold["frame_only"]],
        work_dir=str(work_dir), num_workers=num_workers(), time_budget_s=FOLD_BUDGET_S)
    pin, pout = work_dir / f"{tag}_in.json", work_dir / f"{tag}_out.json"
    pin.write_text(json.dumps(payload))
    path = os.pathsep.join([str(program.parent), str(task_dir() / "wing_lib")] + [p for p in [os.environ.get("PYTHONPATH")] if p])
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1",
               PYTHONPATH=path)
    t0 = time.time()
    with open(logs / f"{tag}.log", "w") as log:
        proc = subprocess.Popen([sys.executable, "-c", _CHILD, str(pin), str(pout)], stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=str(work_dir), start_new_session=True)
        try:
            proc.wait(timeout=FOLD_BUDGET_S + GRACE_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
            return None, f"fold {fold['name']} exceeded {FOLD_BUDGET_S + GRACE_S} s", time.time() - t0
    elapsed = time.time() - t0
    if proc.returncode != 0 or not pout.is_file():
        tail = (logs / f"{tag}.log").read_text(errors="replace")[-1500:]
        return None, f"fold {fold['name']} failed (exit {proc.returncode}): {tail}", elapsed
    try:
        preds = json.loads(pout.read_text())
        got = {(p["wing"], int(p["freq_Hz"])): (float(p["lift_mN"]), float(p["power_W"])) for p in preds}
    except (ValueError, KeyError, TypeError) as exc:
        return None, f"fold {fold['name']} returned malformed predictions: {exc}", elapsed
    rows = []
    for r in fold["test"]:
        key = (r["wing"], r["freq_Hz"])
        if key not in got or not all(np.isfinite(got[key])) or got[key][1] <= 0:
            return None, f"fold {fold['name']}: missing, non-finite or non-positive-power prediction for {key}", elapsed
        rows.append(dict(fold=fold["name"], wing=r["wing"], planform=r["planform"], membrane=r["membrane"],
                         freq_Hz=r["freq_Hz"], lift_pred_mN=got[key][0], power_pred_W=got[key][1],
                         lift_meas_mN=r["lift_mean_mN"], lift_sd_mN=r["lift_sd_mN"], power_meas_W=r["power_mean_W"],
                         power_sd_W=r["power_sd_W"]))
    return rows, "", elapsed


def _spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _nrmse(pred, meas) -> tuple[float, float]:
    pred, meas = np.asarray(pred), np.asarray(meas)
    rmse = float(np.sqrt(np.mean((pred - meas) ** 2)))
    return rmse, rmse / float(np.sqrt(np.mean(meas**2)))


def score(rows: list[dict]) -> dict:
    lp, lm = [r["lift_pred_mN"] for r in rows], [r["lift_meas_mN"] for r in rows]
    pp, pm = [r["power_pred_W"] for r in rows], [r["power_meas_W"] for r in rows]
    lift_rmse, lift_n = _nrmse(lp, lm)
    power_rmse, power_n = _nrmse(pp, pm)
    freqs = sorted({r["freq_Hz"] for r in rows})
    rho = []
    for f in freqs:
        sub = [r for r in rows if r["freq_Hz"] == f]
        rho.append(_spearman([r["lift_pred_mN"] / r["power_pred_W"] for r in sub],
                             [r["lift_meas_mN"] / r["power_meas_W"] for r in sub]))
    skills = dict(lift=max(0.0, 1 - lift_n), power=max(0.0, 1 - power_n), rank=float(np.mean([(x + 1) / 2 for x in rho])))
    combined = sum(WEIGHTS[k] * skills[k] for k in WEIGHTS)

    def subset(pred):
        sub = [r for r in rows if pred(r)]
        l_rmse, l_n = _nrmse([r["lift_pred_mN"] for r in sub], [r["lift_meas_mN"] for r in sub])
        p_rmse, p_n = _nrmse([r["power_pred_W"] for r in sub], [r["power_meas_W"] for r in sub])
        bias = float(np.mean([r["lift_pred_mN"] - r["lift_meas_mN"] for r in sub]))
        return dict(n=len(sub), lift_rmse_mN=round(l_rmse, 3), lift_nrmse=round(l_n, 4), lift_bias_mN=round(bias, 3),
                    power_rmse_W=round(p_rmse, 4), power_nrmse=round(p_n, 4))

    per_wing = {w: subset(lambda r, w=w: r["wing"] == w) for w in sorted({r["wing"] for r in rows})}
    # Measurement repeatability for context: RMS over conditions of the standard error of the mean (n = 3).
    noise = float(np.sqrt(np.mean([(r["lift_sd_mN"] / np.sqrt(3)) ** 2 for r in rows])))
    return dict(
        combined_score=round(combined, 6), lift_skill=round(skills["lift"], 6), power_skill=round(skills["power"], 6),
        rank_skill=round(skills["rank"], 6), lift_rmse_mN=round(lift_rmse, 4), lift_nrmse=round(lift_n, 4),
        power_rmse_W=round(power_rmse, 5), power_nrmse=round(power_n, 4),
        ratio_spearman_by_frequency={str(f): round(x, 4) for f, x in zip(freqs, rho)},
        reference_wings=subset(lambda r: r["wing"] not in DESIGNED), designed_wings=subset(lambda r: r["wing"] in DESIGNED),
        per_wing=per_wing, lift_measurement_sem_rms_mN=round(noise, 3), n_predictions=len(rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    program = Path(args.program_path).resolve()
    results = Path(args.results_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    work = results / "work"
    logs = results / "logs"
    work.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    metrics: dict = {"combined_score": 0.0}
    correct, message, rows, timing = False, "", [], {}
    try:
        for i, fold in enumerate(folds(load_conditions()), 1):
            print(f"[{i}/4] fold {fold['name']}: train {len(fold['train'])} conditions, predict {len(fold['test'])}", flush=True)
            fold_rows, message, elapsed = run_fold(program, fold, work, logs)
            timing[fold["name"]] = round(elapsed, 1)
            if fold_rows is None:
                break
            rows += fold_rows
            print(f"      done in {elapsed:.0f} s", flush=True)
        else:
            metrics = score(rows)
            metrics["fold_seconds"] = timing
            correct = True
            print(f"combined_score {metrics['combined_score']:.4f} = 0.4 x lift {metrics['lift_skill']:.3f} + "
                  f"0.3 x power {metrics['power_skill']:.3f} + 0.3 x rank {metrics['rank_skill']:.3f}; "
                  f"lift RMSE {metrics['lift_rmse_mN']:.2f} mN (designed wings {metrics['designed_wings']['lift_rmse_mN']:.2f}), "
                  f"power RMSE {metrics['power_rmse_W']:.3f} W", flush=True)
    except Exception as exc:  # noqa: BLE001 - the evaluator must always write its result files
        message = f"evaluator error: {type(exc).__name__}: {exc}"
    if not correct:
        metrics = {"combined_score": 0.0, "error": message[:1000]}
        print(message[:2000], flush=True)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (results / "correct.json").write_text(json.dumps({"correct": bool(correct), "error": None if correct else message[:1000]}))
    if rows:
        with open(results / "predictions.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved {results / 'metrics.json'} and correct.json (correct={correct})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
