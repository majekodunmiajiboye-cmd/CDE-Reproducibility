from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260926
FEATURES = [
    "observability",
    "action_granularity",
    "context_state_management",
    "verification",
    "recoverability",
    "decomposition_control",
]
FULL_COVERAGE_HARNESSES = [
    "hermes",
    "moltis-local",
    "nanobot",
    "nullclaw",
    "openclaw-local",
    "zeroclaw-local",
]
DOC_FILES = {
    "codex": "codex.txt",
    "hermes": "hermes.txt",
    "moltis-local": "moltis.txt",
    "nanobot": "nanobot.txt",
    "nullclaw": "nullclaw.txt",
    "openclaw-local": "openclaw.txt",
    "zeroclaw-local": "zeroclaw.txt",
}


def load_data(scores_path: Path, consensus_path: Path, docs_dir: Path) -> pd.DataFrame:
    payload = json.loads(scores_path.read_text())
    rows = [r for r in payload["runs"] if r.get("completion") is not None]
    df = pd.DataFrame(rows)
    df["task"] = df["task_id"].astype(str)
    # Frozen primary outcome: a task counts as verified success only when the
    # completion rubric awards full completion. Partial rubric credit is failure
    # for the binary success estimand; process score is not used as the outcome.
    df["y"] = (pd.to_numeric(df["completion"], errors="coerce") >= 1.0 - 1e-12).astype(int)

    consensus = json.loads(consensus_path.read_text())["consensus"]
    for f in FEATURES:
        df[f] = df["harness"].map(lambda h: float(consensus[h][f]))
    docs = {h: (docs_dir / fn).read_text(errors="replace") for h, fn in DOC_FILES.items()}
    df["doc"] = df["harness"].map(docs)
    # Stable row ids let us precompute the documentation representation once
    # and slice it in every cross-validation fold.  This is a pure execution
    # optimisation: HashingVectorizer is stateless, so hashing one copy of a
    # harness document and reusing that row is mathematically identical to
    # hashing the same document thousands of times.
    df["_row_id"] = np.arange(len(df), dtype=int)
    return df


def build_document_hash_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    hv = HashingVectorizer(
        n_features=256,
        alternate_sign=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        norm="l2",
    )
    # There are only seven unique harness documents.  Transform each once,
    # then expand by harness identity to the full row-aligned sparse matrix.
    unique = (
        df[["harness", "doc"]]
        .drop_duplicates(subset=["harness"])
        .sort_values("harness")
        .reset_index(drop=True)
    )
    unique_matrix = hv.transform(unique["doc"])
    pos = {h: i for i, h in enumerate(unique["harness"].tolist())}
    row_pos = np.fromiter((pos[h] for h in df["harness"]), dtype=int, count=len(df))
    return unique_matrix[row_pos].tocsr()


def matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kind: str,
    doc_hash_matrix: sparse.csr_matrix,
):
    cat_cols = ["model", "task"]
    if kind == "categorical":
        cat_cols = ["model", "task", "harness"]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    a = enc.fit_transform(train[cat_cols])
    b = enc.transform(test[cat_cols])

    if kind == "coded":
        sc = StandardScaler()
        na = sc.fit_transform(train[FEATURES])
        nb = sc.transform(test[FEATURES])
        a = sparse.hstack([a, sparse.csr_matrix(na)], format="csr")
        b = sparse.hstack([b, sparse.csr_matrix(nb)], format="csr")
    elif kind == "doc":
        ah = doc_hash_matrix[train["_row_id"].to_numpy(dtype=int)]
        bh = doc_hash_matrix[test["_row_id"].to_numpy(dtype=int)]
        a = sparse.hstack([a, ah], format="csr")
        b = sparse.hstack([b, bh], format="csr")
    return a, b


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kind: str,
    doc_hash_matrix: sparse.csr_matrix,
) -> np.ndarray:
    if train.y.nunique() < 2:
        return np.full(len(test), float(train.y.mean()))
    a, b = matrices(train, test, kind, doc_hash_matrix)
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=4000,
        random_state=SEED,
    )
    clf.fit(a, train.y.to_numpy())
    return clf.predict_proba(b)[:, list(clf.classes_).index(1)]


def row_log_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(pred: pd.DataFrame) -> dict:
    out = {}
    for k in ["noenv", "categorical", "coded", "doc"]:
        p = pred[f"p_{k}"].to_numpy()
        y = pred.y.to_numpy()
        out[k] = {
            "log_loss": float(row_log_loss(y, p).mean()),
            "brier": float(np.mean((p - y) ** 2)),
        }
    return out


def bootstrap_task_differences(pred: pd.DataFrame, n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(SEED)
    tasks = np.array(sorted(pred.task.unique()))
    names = {
        "categorical_minus_coded": ("categorical", "coded"),
        "noenv_minus_coded": ("noenv", "coded"),
        "doc_minus_coded": ("doc", "coded"),
    }
    # Pre-aggregate the exact row-level loss differences by task.  Sampling
    # those task sums/counts is algebraically identical to concatenating the
    # sampled task DataFrames on every bootstrap iteration, but avoids ~2,000
    # large pandas concatenations.
    counts = np.array([int(pred.task.eq(t).sum()) for t in tasks], dtype=float)
    task_sums = {}
    y = pred.y.to_numpy()
    for name, (lhs, rhs) in names.items():
        row_diff = (
            row_log_loss(y, pred[f"p_{lhs}"].to_numpy())
            - row_log_loss(y, pred[f"p_{rhs}"].to_numpy())
        )
        tmp = pred[["task"]].copy()
        tmp["row_diff"] = row_diff
        sums = tmp.groupby("task", sort=False)["row_diff"].sum()
        task_sums[name] = np.array([float(sums.loc[t]) for t in tasks])

    values = {n: np.empty(n_boot, dtype=float) for n in names}
    task_index = {t: i for i, t in enumerate(tasks)}
    for _ in range(n_boot):
        # Keep the original RNG operation exactly (choice over the task labels)
        # so a fixed seed reproduces the pre-optimisation bootstrap draws.
        sample = rng.choice(tasks, size=len(tasks), replace=True)
        sample_idx = np.fromiter((task_index[t] for t in sample), dtype=int, count=len(sample))
        denom = counts[sample_idx].sum()
        for n in names:
            values[n][_] = float(task_sums[n][sample_idx].sum() / denom)
    return {
        n: {
            "point": float(
                row_log_loss(pred.y, pred[f"p_{lhs}"]).mean()
                - row_log_loss(pred.y, pred[f"p_{rhs}"]).mean()
            ),
            "ci_2p5_50_97p5": [float(x) for x in np.quantile(values[n], [.025, .5, .975])],
        }
        for n, (lhs, rhs) in names.items()
    }


def cell_cv(df: pd.DataFrame, doc_hash_matrix: sparse.csr_matrix) -> pd.DataFrame:
    # Confirmatory cell test requires the held-out harness to have other models
    # in training. Codex has only one tested model in this release, so its lone
    # cell is structurally non-identifiable for this split and excluded here.
    d = df[df.harness.isin(FULL_COVERAGE_HARNESSES)].copy()
    rows = []
    for harness in FULL_COVERAGE_HARNESSES:
        for model in sorted(d.model.unique()):
            mask = d.harness.eq(harness) & d.model.eq(model)
            test = d.loc[mask].copy()
            if test.empty:
                continue
            train = d.loc[~mask].copy()
            block = test[["task", "harness", "model", "y"]].copy()
            for kind in ["noenv", "categorical", "coded", "doc"]:
                block[f"p_{kind}"] = fit_predict(train, test, kind, doc_hash_matrix)
            rows.append(block)
    return pd.concat(rows, ignore_index=True)


def harness_cv(df: pd.DataFrame, doc_hash_matrix: sparse.csr_matrix) -> pd.DataFrame:
    # Primary unseen-harness replication uses only the six harnesses with the
    # same eight-model task panel. This avoids conflating harness transfer with
    # Codex's structural single-model missingness.
    d = df[df.harness.isin(FULL_COVERAGE_HARNESSES)].copy()
    rows = []
    for harness in FULL_COVERAGE_HARNESSES:
        test = d.loc[d.harness.eq(harness)].copy()
        train = d.loc[~d.harness.eq(harness)].copy()
        block = test[["task", "harness", "model", "y"]].copy()
        for kind in ["noenv", "categorical", "coded", "doc"]:
            block[f"p_{kind}"] = fit_predict(train, test, kind, doc_hash_matrix)
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--consensus", type=Path, required=True)
    ap.add_argument("--docs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(args.scores, args.consensus, args.docs_dir)
    t0 = time.perf_counter()
    doc_hash_matrix = build_document_hash_matrix(df)
    print(f"precomputed documentation hashes: {doc_hash_matrix.shape} in {time.perf_counter()-t0:.2f}s", flush=True)

    t1 = time.perf_counter()
    cell = cell_cv(df, doc_hash_matrix)
    print(f"cell CV complete in {time.perf_counter()-t1:.2f}s", flush=True)
    t2 = time.perf_counter()
    unseen = harness_cv(df, doc_hash_matrix)
    print(f"unseen-harness CV complete in {time.perf_counter()-t2:.2f}s", flush=True)
    cell.to_csv(args.out_dir / "r4_harnessbench_cell_oof.csv.gz", index=False, compression="gzip")
    unseen.to_csv(args.out_dir / "r4_harnessbench_unseen_harness_oof.csv.gz", index=False, compression="gzip")

    cell_m = metrics(cell)
    unseen_m = metrics(unseen)
    t3 = time.perf_counter()
    cell_boot = bootstrap_task_differences(cell)
    unseen_boot = bootstrap_task_differences(unseen)
    print(f"bootstraps complete in {time.perf_counter()-t3:.2f}s", flush=True)

    checks = {
        "cell_coded_no_worse_than_categorical": bool(cell_m["coded"]["log_loss"] <= cell_m["categorical"]["log_loss"]),
        "unseen_coded_beats_noenv": bool(unseen_m["coded"]["log_loss"] < unseen_m["noenv"]["log_loss"]),
        "unseen_coded_beats_doc": bool(unseen_m["coded"]["log_loss"] < unseen_m["doc"]["log_loss"]),
    }
    result = {
        "analysis": "CDE R4 Harness-Bench blinded-feature replication",
        "outcome": "completion == 1.0",
        "n_source_rows_with_completion": int(len(df)),
        "overall_full_completion_rate": float(df.y.mean()),
        "primary_full_coverage_harnesses": FULL_COVERAGE_HARNESSES,
        "codex_exclusion_reason": "only gpt-5.4 x codex is present; no other model x codex cells, so a held model x harness cell cannot leave codex observed in training",
        "n_primary_rows": int(df.harness.isin(FULL_COVERAGE_HARNESSES).sum()),
        "n_models_primary": int(df.loc[df.harness.isin(FULL_COVERAGE_HARNESSES), "model"].nunique()),
        "n_tasks_primary": int(df.loc[df.harness.isin(FULL_COVERAGE_HARNESSES), "task"].nunique()),
        "cell_holdout": {
            "metrics": cell_m,
            "task_cluster_bootstrap_differences": cell_boot,
        },
        "unseen_harness": {
            "metrics": unseen_m,
            "task_cluster_bootstrap_differences": unseen_boot,
        },
        "frozen_checks": checks,
        "replication_pass": bool(all(checks.values())),
        "documentation_baseline": "stateless 256-d char_wb hashing of outcome-blind local harness documentation plus model and task fixed effects",
        "coded_representation": "arithmetic-mean consensus of two independently blinded 0/1/2 coders on six predeclared features",
        "seed": SEED,
        "execution_optimisation": "documentation hashes precomputed once per unique harness; task-cluster bootstrap uses exact per-task loss-sum aggregation",
    }
    (args.out_dir / "r4_harnessbench_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
