from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from r1_adele import DEMANDS, SEED, clip_logit, load_data


BASE_NUMERIC = ["panel_logit"] + [f"panel_resp_{i:02d}" for i in range(14)]


def make_model(numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("benchmark", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["benchmark"]),
    ], sparse_threshold=0.0)
    # Fixed, deliberately moderate-capacity nonlinear model. No tuning is
    # performed on the outcomes. The paired baseline and +demand model share
    # every hyperparameter; the only difference is the addition of demand
    # columns to the input.
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        max_bins=63,
        early_stopping=False,
        random_state=SEED,
    )
    return Pipeline([("features", pre), ("hgb", clf)])


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    if train["y"].nunique() < 2:
        return np.full(len(test), float(train["y"].mean()))
    cols = numeric_cols + ["benchmark"]
    model = make_model(numeric_cols)
    model.fit(train[cols], train["y"].to_numpy(int))
    return model.predict_proba(test[cols])[:, 1]


def target_frame(base: pd.DataFrame, y: np.ndarray, target_idx: int, model_names: list[str]) -> pd.DataFrame:
    panel_idx = [j for j in range(len(model_names)) if j != target_idx]
    panel = y[panel_idx].T.astype(float)
    panel_success = panel.sum(axis=1)
    panel_p = (panel_success + 0.5) / (len(panel_idx) + 1.0)
    frame = base[["instance_id", "benchmark", "task"] + DEMANDS].copy()
    frame["panel_logit"] = clip_logit(panel_p)
    for j in range(panel.shape[1]):
        frame[f"panel_resp_{j:02d}"] = panel[:, j]
    frame["y"] = y[target_idx]
    return frame


def run_target(target_idx: int, base: pd.DataFrame, y: np.ndarray, model_names: list[str]) -> list[dict]:
    frame = target_frame(base, y, target_idx, model_names)
    rows: list[dict] = []
    for group in sorted(frame["task"].astype(str).unique()):
        test_mask = frame["task"].astype(str).eq(group)
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        p_base = fit_predict(train, test, BASE_NUMERIC)
        p_demand = fit_predict(train, test, BASE_NUMERIC + DEMANDS)
        for pos, idx in enumerate(test.index):
            rows.append({
                "model": model_names[target_idx],
                "instance_id": frame.at[idx, "instance_id"],
                "benchmark": frame.at[idx, "benchmark"],
                "task": frame.at[idx, "task"],
                "heldout_group": group,
                "y": int(frame.at[idx, "y"]),
                "p_nonlinear_fullpanel": float(p_base[pos]),
                "p_nonlinear_fullpanel_demand": float(p_demand[pos]),
            })
    return rows


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def cluster_bootstrap(pred: pd.DataFrame, n_boot: int = 1000) -> dict:
    y = pred["y"].to_numpy(float)
    p0 = np.clip(pred["p_nonlinear_fullpanel"].to_numpy(float), 1e-6, 1 - 1e-6)
    p1 = np.clip(pred["p_nonlinear_fullpanel_demand"].to_numpy(float), 1e-6, 1 - 1e-6)
    l0 = -(y * np.log(p0) + (1 - y) * np.log(1 - p0))
    l1 = -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
    diff = l0 - l1
    tasks = np.array(sorted(pred["task"].astype(str).unique()))
    tmp = pd.DataFrame({"task": pred["task"].astype(str), "diff": diff})
    sums = tmp.groupby("task")["diff"].sum()
    counts = tmp.groupby("task").size()
    s = np.array([float(sums.loc[t]) for t in tasks])
    c = np.array([int(counts.loc[t]) for t in tasks], float)
    rng = np.random.default_rng(SEED)
    vals = np.empty(n_boot, float)
    for i in range(n_boot):
        ix = rng.integers(0, len(tasks), len(tasks))
        vals[i] = s[ix].sum() / c[ix].sum()
    point = float(diff.mean())
    return {
        "delta_ll_baseline_minus_demand": point,
        "relative_ll_reduction": point / float(l0.mean()),
        "ci_2p5_50_97p5": [float(x) for x in np.quantile(vals, [0.025, 0.5, 0.975])],
        "n_boot": n_boot,
        "seed": SEED,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--battery-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base, y, names = load_data(args.data_dir, args.battery_file)
    outputs = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(run_target)(i, base, y, names) for i in range(len(names))
    )
    pred = pd.DataFrame([r for block in outputs for r in block])
    pred.to_csv(args.out_dir / "r1_nonlinear_robustness_oof.csv.gz", index=False, compression="gzip")
    metrics = {
        "nonlinear_fullpanel": score(pred.y, pred.p_nonlinear_fullpanel),
        "nonlinear_fullpanel_plus_demands": score(pred.y, pred.p_nonlinear_fullpanel_demand),
    }
    boot = cluster_bootstrap(pred, 1000)
    result = {
        "analysis": "CDE R1 submission-readiness nonlinear robustness diagnostic",
        "status": "supplementary; frozen before this nonlinear analysis was run; does not rewrite the original R1 verdict",
        "n_models": len(names),
        "n_items": int(len(base)),
        "n_task_families": int(base.task.nunique()),
        "model": {
            "family": "HistGradientBoostingClassifier",
            "fixed_hyperparameters": {
                "learning_rate": 0.05,
                "max_iter": 100,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 40,
                "l2_regularization": 1.0,
                "max_bins": 63,
                "early_stopping": False,
                "random_state": SEED,
            },
            "baseline_features": "external-panel scalar item easiness + all 14 external-model binary item responses + benchmark identity",
            "augmented_features": "baseline + all 18 published ADeLe demand coordinates",
        },
        "leakage_guard": "For every target model, the target is excluded from all panel-derived features; held-out task-family target outcomes are never used in fitting. Hyperparameters were fixed before running this analysis and were not tuned to outcomes.",
        "metrics": metrics,
        "bootstrap": boot,
        "survival_check": {
            "criterion": "The demand increment survives this nonlinear comparator if the task-family bootstrap 95% CI for LL(response-only)-LL(response+demands) is entirely > 0.",
            "pass": bool(boot["ci_2p5_50_97p5"][0] > 0),
        },
    }
    (args.out_dir / "r1_nonlinear_robustness_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
