from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from r1_adele import DEMANDS, SEED, clip_logit, load_data


MODEL_SPECS = [
    "domain",
    "demand_domain",
    "latent_domain",
    "fullpanel_domain",
    "fullpanel_demand_domain",
]


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def make_model(numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("benchmark", OneHotEncoder(handle_unknown="ignore"), ["benchmark"]),
    ])
    return Pipeline([
        ("features", pre),
        ("logit", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=4000,
            random_state=SEED,
        )),
    ])


def positive_prob(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    classes = list(model.named_steps["logit"].classes_)
    if 1 not in classes:
        return np.full(len(x), float(classes[0] == 1))
    return model.predict_proba(x)[:, classes.index(1)]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    if train["y"].nunique() < 2:
        return np.full(len(test), float(train["y"].mean()))
    cols = numeric_cols + ["benchmark"]
    model = make_model(numeric_cols)
    model.fit(train[cols], train["y"])
    return positive_prob(model, test[cols])


def target_frame(
    base: pd.DataFrame,
    y: np.ndarray,
    target_idx: int,
    model_names: list[str],
) -> tuple[pd.DataFrame, int, list[str], list[str]]:
    panel_idx = [j for j in range(len(model_names)) if j != target_idx]
    panel_names = [model_names[j] for j in panel_idx]
    panel = y[panel_idx].T.astype(float)

    # Same leakage guard as frozen R1: all item calibration information comes
    # from the other 14 models; the target model is excluded completely.
    panel_success = panel.sum(axis=1)
    panel_p = (panel_success + 0.5) / (len(panel_idx) + 1.0)
    panel_logit = clip_logit(panel_p)

    frame = base[["instance_id", "benchmark", "task"] + DEMANDS].copy()
    frame["panel_logit"] = panel_logit
    frame["y"] = y[target_idx]

    panel_cols: list[str] = []
    for j, name in enumerate(panel_names):
        c = f"panel_resp_{j:02d}"
        frame[c] = panel[:, j]
        panel_cols.append(c)

    # MIRT-style latent item profile: remove scalar item easiness first, then
    # learn low-rank differential-response factors from external-panel outcomes
    # only. PCA dimension is the smallest number explaining >=90% variance;
    # this uses no target-model outcomes and no family labels.
    residual = panel - panel.mean(axis=1, keepdims=True)
    pca = PCA(n_components=0.90, svd_solver="full", random_state=SEED)
    latent = pca.fit_transform(residual)
    latent_cols: list[str] = []
    for k in range(latent.shape[1]):
        c = f"latent_{k:02d}"
        frame[c] = latent[:, k]
        latent_cols.append(c)

    return frame, int(latent.shape[1]), panel_cols, latent_cols


def run_target(
    target_idx: int,
    base: pd.DataFrame,
    y: np.ndarray,
    model_names: list[str],
) -> tuple[list[dict], dict]:
    frame, pca_k, panel_cols, latent_cols = target_frame(base, y, target_idx, model_names)
    groups = sorted(frame["task"].astype(str).unique())
    rows: list[dict] = []

    features = {
        "domain": ["panel_logit"],
        "demand_domain": ["panel_logit"] + DEMANDS,
        "latent_domain": ["panel_logit"] + latent_cols,
        "fullpanel_domain": ["panel_logit"] + panel_cols,
        "fullpanel_demand_domain": ["panel_logit"] + panel_cols + DEMANDS,
    }

    for group in groups:
        test_mask = frame["task"].astype(str).eq(group)
        train = frame.loc[~test_mask]
        test = frame.loc[test_mask]
        pred = {name: fit_predict(train, test, cols) for name, cols in features.items()}

        for pos, idx in enumerate(test.index):
            row = {
                "model": model_names[target_idx],
                "instance_id": frame.at[idx, "instance_id"],
                "benchmark": frame.at[idx, "benchmark"],
                "task": frame.at[idx, "task"],
                "heldout_group": group,
                "y": int(frame.at[idx, "y"]),
            }
            for name in MODEL_SPECS:
                row[f"p_{name}"] = float(pred[name][pos])
            rows.append(row)

    meta = {
        "target_model": model_names[target_idx],
        "external_panel_models": [m for i, m in enumerate(model_names) if i != target_idx],
        "latent_pca_components_90pct": pca_k,
    }
    return rows, meta


def pairwise_cluster_bootstrap(
    pred: pd.DataFrame,
    lhs: str,
    rhs: str,
    n_boot: int = 1000,
) -> dict:
    """Return LL(lhs)-LL(rhs); positive means rhs predicts better."""
    p_lhs = np.clip(pred[f"p_{lhs}"].to_numpy(float), 1e-6, 1 - 1e-6)
    p_rhs = np.clip(pred[f"p_{rhs}"].to_numpy(float), 1e-6, 1 - 1e-6)
    yy = pred["y"].to_numpy(float)
    loss_lhs = -(yy * np.log(p_lhs) + (1 - yy) * np.log(1 - p_lhs))
    loss_rhs = -(yy * np.log(p_rhs) + (1 - yy) * np.log(1 - p_rhs))
    diff = loss_lhs - loss_rhs

    fams = np.array(sorted(pred["task"].astype(str).unique()))
    tmp = pd.DataFrame({"task": pred["task"].astype(str), "diff": diff})
    sums = tmp.groupby("task", sort=False)["diff"].sum()
    counts = tmp.groupby("task", sort=False).size()
    sum_arr = np.array([float(sums.loc[f]) for f in fams])
    count_arr = np.array([int(counts.loc[f]) for f in fams], dtype=float)
    fam_index = {f: i for i, f in enumerate(fams)}

    rng = np.random.default_rng(SEED)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = rng.choice(fams, size=len(fams), replace=True)
        ii = np.fromiter((fam_index[f] for f in sampled), dtype=int, count=len(sampled))
        vals[b] = sum_arr[ii].sum() / count_arr[ii].sum()

    point = float(diff.mean())
    q = [float(x) for x in np.quantile(vals, [0.025, 0.5, 0.975])]
    lhs_ll = float(loss_lhs.mean())
    rhs_ll = float(loss_rhs.mean())
    return {
        "lhs": lhs,
        "rhs": rhs,
        "interpretation": "positive delta means rhs has lower log loss",
        "delta_ll_lhs_minus_rhs": point,
        "ci_2p5_50_97p5": q,
        "relative_ll_reduction_of_rhs_vs_lhs": point / lhs_ll,
        "lhs_log_loss": lhs_ll,
        "rhs_log_loss": rhs_ll,
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
    rows = [row for target_rows, _ in outputs for row in target_rows]
    target_meta = [meta for _, meta in outputs]
    pred = pd.DataFrame(rows)
    pred.to_csv(args.out_dir / "r1_strong_comparator_oof.csv.gz", index=False, compression="gzip")

    metrics = {
        name: score(pred["y"].to_numpy(), pred[f"p_{name}"].to_numpy())
        for name in MODEL_SPECS
    }
    comparisons = {
        "demand_increment_over_domain": pairwise_cluster_bootstrap(pred, "domain", "demand_domain"),
        "demand_vs_latent_panel": pairwise_cluster_bootstrap(pred, "latent_domain", "demand_domain"),
        "demand_vs_full_panel": pairwise_cluster_bootstrap(pred, "fullpanel_domain", "demand_domain"),
        "demand_increment_over_full_panel": pairwise_cluster_bootstrap(
            pred, "fullpanel_domain", "fullpanel_demand_domain"
        ),
    }

    unique_survival = comparisons["demand_increment_over_full_panel"]
    minimum_survival = bool(unique_survival["ci_2p5_50_97p5"][0] > 0)

    result = {
        "analysis": "CDE R1 post-confirmatory strong-comparator diagnostic",
        "status": "supplementary; does not rewrite frozen R1 verdict",
        "seed": SEED,
        "n_models": len(names),
        "n_items": int(len(base)),
        "n_task_families": int(base["task"].nunique()),
        "n_benchmarks": int(base["benchmark"].nunique()),
        "model_definitions": {
            "domain": "external-panel scalar item easiness + benchmark identity",
            "demand_domain": "domain + all 18 published ADeLe demand coordinates",
            "latent_domain": "domain + PCA90 differential item-response factors from the 14-model external panel; target model excluded",
            "fullpanel_domain": "domain + all 14 external-model binary item responses as a ridge-logistic multidimensional response-profile control",
            "fullpanel_demand_domain": "fullpanel_domain + all 18 ADeLe demand coordinates",
        },
        "leakage_guard": "For each target model, all panel features and latent factors are computed from the other 14 models only. Target-model outcomes from the held-out task family never enter fitting or calibration.",
        "target_panel_metadata": target_meta,
        "metrics": metrics,
        "comparisons": comparisons,
        "predeclared_minimum_survival_check": {
            "criterion": "ADeLe demand coordinates add unique held-out predictive information beyond the full external response profile only if the 95% task-family bootstrap CI for LL(fullpanel_domain)-LL(fullpanel_demand_domain) is entirely > 0. No new arbitrary effect-size threshold is introduced; magnitude is reported separately.",
            "pass": minimum_survival,
        },
    }
    (args.out_dir / "r1_strong_comparator_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
