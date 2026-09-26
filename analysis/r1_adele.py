from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260926
MIN_PANEL_MODELS_FOR_ITEM_2PL = 20
DEMANDS = [
    "AS", "CEc", "CEe", "CL", "KNn", "KNa", "KNc", "KNf", "KNs",
    "MCt", "MCu", "MCr", "MS", "QLq", "QLl", "SNs", "VO", "AT",
]
META = ["instance_id", "benchmark", "task", "UG"]
OUTCOME = "verification_final"


def clip_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_data(data_dir: Path, battery_file: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    files = sorted(data_dir.glob("*.csv"))
    if len(files) != 15:
        raise RuntimeError(f"Expected 15 ADeLe subject files; found {len(files)}")

    # The official repository publishes one canonical battery table containing
    # the task/demand annotations and separate subject-specific files containing
    # responses. Some subject-specific CSVs carry stale/different copies of a
    # subset of annotations (detected before any R1 outcome relationship was
    # computed). Therefore annotations come only from the canonical battery.
    base = pd.read_csv(
        battery_file,
        usecols=META + DEMANDS,
        low_memory=False,
    )
    ys: list[np.ndarray] = []
    names: list[str] = []

    for f in files:
        d = pd.read_csv(f, usecols=["instance_id", OUTCOME], low_memory=False)
        d[OUTCOME] = pd.to_numeric(d[OUTCOME], errors="coerce")
        if not np.array_equal(d["instance_id"].astype(str).values,
                              base["instance_id"].astype(str).values):
            raise RuntimeError(f"Instance ordering mismatch in {f.name}")
        ys.append(d[OUTCOME].to_numpy(dtype=float))
        names.append(f.stem)

    for c in ["UG"] + DEMANDS:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    Y = np.vstack(ys)
    keep = (
        np.isfinite(Y).all(axis=0)
        & base[["UG"] + DEMANDS].notna().all(axis=1).to_numpy()
        & (base["UG"].to_numpy(dtype=float) >= 75.0)
    )
    base = base.loc[keep].reset_index(drop=True)
    Y = Y[:, keep]
    if not np.isin(Y, [0.0, 1.0]).all():
        raise RuntimeError("Non-binary verification_final values after filtering")
    return base, Y.astype(int), names


def numeric_model(cols: list[str]) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=4000,
            random_state=SEED,
        )),
    ])


def domain_model() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), ["panel_logit"]),
        ("benchmark", OneHotEncoder(handle_unknown="ignore"), ["benchmark"]),
    ])
    return Pipeline([
        ("features", pre),
        ("logit", LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=4000,
            random_state=SEED,
        )),
    ])


def positive_prob(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["logit"].classes_)
    if 1 not in classes:
        return np.full(len(X), float(classes[0] == 1))
    return model.predict_proba(X)[:, classes.index(1)]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], domain: bool = False) -> np.ndarray:
    if train["y"].nunique() < 2:
        return np.full(len(test), float(train["y"].mean()))
    mdl = domain_model() if domain else numeric_model(features)
    mdl.fit(train[features if not domain else ["panel_logit", "benchmark"]], train["y"])
    Xtest = test[features if not domain else ["panel_logit", "benchmark"]]
    return positive_prob(mdl, Xtest)


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def run_group_holdout(base: pd.DataFrame, Y: np.ndarray, model_names: list[str], group_col: str) -> tuple[pd.DataFrame, dict]:
    groups = sorted(base[group_col].astype(str).unique())
    rows = []

    for mi, model_name in enumerate(model_names):
        panel_idx = [j for j in range(len(model_names)) if j != mi]
        # This implementation supplement predeclares a conservative minimum of
        # 20 independent calibration models for estimating an item-specific 2PL
        # slope. ADeLe leaves only 14 panel models after excluding the target,
        # so every item uses the frozen protocol's allowed 1PL/Rasch-like
        # fallback: Jeffreys-smoothed external-panel item easiness.
        panel_success = Y[panel_idx].sum(axis=0)
        panel_p = (panel_success + 0.5) / (len(panel_idx) + 1.0)
        panel_logit = clip_logit(panel_p)

        frame = base[["instance_id", "benchmark", "task"] + DEMANDS].copy()
        frame["panel_logit"] = panel_logit
        frame["y"] = Y[mi]

        for g in groups:
            test_mask = frame[group_col].astype(str).eq(g)
            train = frame.loc[~test_mask]
            test = frame.loc[test_mask].copy()
            if test.empty:
                continue

            p0 = fit_predict(train, test, ["panel_logit"])
            p1 = fit_predict(train, test, ["panel_logit"] + DEMANDS)
            pdomain = fit_predict(train, test, ["panel_logit"], domain=True)

            for k, idx in enumerate(test.index):
                rows.append({
                    "model": model_name,
                    "instance_id": frame.at[idx, "instance_id"],
                    "benchmark": frame.at[idx, "benchmark"],
                    "task": frame.at[idx, "task"],
                    "heldout_group": g,
                    "y": int(frame.at[idx, "y"]),
                    "p_m0": float(p0[k]),
                    "p_m1": float(p1[k]),
                    "p_domain": float(pdomain[k]),
                })

    pred = pd.DataFrame(rows)
    metrics = {
        "m0": score(pred.y.values, pred.p_m0.values),
        "m1": score(pred.y.values, pred.p_m1.values),
        "domain": score(pred.y.values, pred.p_domain.values),
    }
    metrics["delta_ll_m0_minus_m1"] = metrics["m0"]["log_loss"] - metrics["m1"]["log_loss"]
    metrics["relative_ll_reduction"] = metrics["delta_ll_m0_minus_m1"] / metrics["m0"]["log_loss"]
    metrics["delta_ll_domain_minus_m1"] = metrics["domain"]["log_loss"] - metrics["m1"]["log_loss"]
    return pred, metrics


def family_bootstrap(pred: pd.DataFrame, n_boot: int = 1000) -> dict[str, object]:
    # Cluster bootstrap by task family. Repeated sampled families contribute
    # repeated complete clusters, preserving all model/item observations.
    rng = np.random.default_rng(SEED)
    fams = np.array(sorted(pred["task"].astype(str).unique()))
    grouped = {f: pred.loc[pred.task.astype(str).eq(f)] for f in fams}
    deltas = []
    relatives = []
    for _ in range(n_boot):
        sample = rng.choice(fams, size=len(fams), replace=True)
        chunks = [grouped[f] for f in sample]
        b = pd.concat(chunks, ignore_index=True)
        ll0 = log_loss(b.y, np.clip(b.p_m0, 1e-6, 1 - 1e-6), labels=[0, 1])
        ll1 = log_loss(b.y, np.clip(b.p_m1, 1e-6, 1 - 1e-6), labels=[0, 1])
        delta = float(ll0 - ll1)
        deltas.append(delta)
        relatives.append(delta / float(ll0))
    q = lambda x: [float(v) for v in np.quantile(np.asarray(x), [0.025, 0.5, 0.975])]
    return {
        "n_boot": n_boot,
        "seed": SEED,
        "delta_ll_ci_2p5_50_97p5": q(deltas),
        "relative_ll_reduction_ci_2p5_50_97p5": q(relatives),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--battery-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-benchmark-sensitivity", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base, Y, names = load_data(args.data_dir, args.battery_file)
    task_pred, task_metrics = run_group_holdout(base, Y, names, "task")
    boot = family_bootstrap(task_pred, 1000)
    task_pred.to_csv(args.out_dir / "r1_adele_task_oof_predictions.csv.gz", index=False, compression="gzip")

    result: dict[str, object] = {
        "analysis": "CDE R1 ADeLe preregistered implementation supplement",
        "seed": SEED,
        "n_models": len(names),
        "model_names": names,
        "n_instances_after_ug75_complete_case_filter": int(len(base)),
        "n_task_families": int(base.task.nunique()),
        "n_benchmarks": int(base.benchmark.nunique()),
        "demand_features": DEMANDS,
        "two_pl_min_panel_models_predeclared": MIN_PANEL_MODELS_FOR_ITEM_2PL,
        "two_pl_fallback_rate": 1.0,
        "two_pl_fallback_reason": "14 calibration models per target/item is below the implementation supplement's predeclared minimum of 20; use Jeffreys-smoothed external-panel 1PL item easiness.",
        "task_holdout": task_metrics,
        "task_family_bootstrap": boot,
    }

    if not args.skip_benchmark_sensitivity:
        bench_pred, bench_metrics = run_group_holdout(base, Y, names, "benchmark")
        bench_pred.to_csv(args.out_dir / "r1_adele_benchmark_oof_predictions.csv.gz", index=False, compression="gzip")
        result["benchmark_holdout_sensitivity"] = bench_metrics

    ll0 = task_metrics["m0"]["log_loss"]
    ll1 = task_metrics["m1"]["log_loss"]
    lld = task_metrics["domain"]["log_loss"]
    ci_lo = boot["delta_ll_ci_2p5_50_97p5"][0]
    relative = (ll0 - ll1) / ll0
    result["frozen_gate"] = {
        "relative_reduction_ge_5pct": bool(relative >= 0.05),
        "bootstrap_delta_ll_ci_excludes_zero_positive": bool(ci_lo > 0),
        "m1_not_worse_than_domain_control": bool(ll1 <= lld),
        "pass": bool(relative >= 0.05 and ci_lo > 0 and ll1 <= lld),
    }

    out = args.out_dir / "r1_adele_result.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
