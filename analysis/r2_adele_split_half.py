from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pointbiserialr


SEED = 20260926
N_SPLITS = 1000
MIN_ITEM_DISCRIMINATION = 0.20
DEMANDS = [
    "AS", "CEc", "CEe", "CL", "KNn", "KNa", "KNc", "KNf", "KNs",
    "MCt", "MCu", "MCr", "MS", "QLq", "QLl", "SNs", "VO", "AT",
]


def load(battery: Path, response_dir: Path):
    ann = pd.read_csv(
        battery,
        usecols=["instance_id", "task", "UG"] + DEMANDS,
        low_memory=False,
    )
    for c in ["UG"] + DEMANDS:
        ann[c] = pd.to_numeric(ann[c], errors="coerce")
    files = sorted(response_dir.glob("*.csv"))
    if len(files) != 15:
        raise RuntimeError(f"expected 15 subject files, found {len(files)}")
    ys, names = [], []
    for f in files:
        d = pd.read_csv(f, usecols=["instance_id", "verification_final"], low_memory=False)
        if not np.array_equal(d.instance_id.astype(str).values, ann.instance_id.astype(str).values):
            raise RuntimeError(f"instance mismatch: {f.name}")
        ys.append(pd.to_numeric(d.verification_final, errors="coerce").to_numpy(float))
        names.append(f.stem)
    Y = np.vstack(ys)
    keep = (
        np.isfinite(Y).all(axis=0)
        & ann[["UG"] + DEMANDS].notna().all(axis=1).to_numpy()
        & (ann.UG.to_numpy(float) >= 75.0)
    )
    return ann.loc[keep].reset_index(drop=True), Y[:, keep].astype(int), names


def corrected_item_discrimination(Y: np.ndarray) -> np.ndarray:
    """Point-biserial item discrimination against leave-one-item-out total score."""
    totals = Y.sum(axis=1).astype(float)
    n_items = Y.shape[1]
    out = np.full(n_items, np.nan)
    for i in range(n_items):
        y = Y[:, i]
        if y.min() == y.max():
            continue
        theta = (totals - y) / max(1, n_items - 1)
        out[i] = pointbiserialr(y, theta).statistic
    return out


def equal_level_ability(Y: np.ndarray, levels: np.ndarray, idx: np.ndarray) -> np.ndarray:
    # Model-free SCC area proxy: equal weight to each observed positive demand
    # level (1..5), preventing the item frequency distribution from defining the
    # score. This is used only for the R2 invariance diagnostic.
    parts = []
    for lv in range(1, 6):
        use = idx[levels[idx] == lv]
        if len(use):
            parts.append(Y[:, use].mean(axis=1))
    if len(parts) < 3:
        return np.full(Y.shape[0], np.nan)
    return np.mean(np.vstack(parts), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-file", type=Path, required=True)
    ap.add_argument("--response-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ann, Y, names = load(args.battery_file, args.response_dir)
    discrim = corrected_item_discrimination(Y)
    rng = np.random.default_rng(SEED)
    all_d = ann[DEMANDS].to_numpy(float)
    max_d = all_d.max(axis=1)

    result = {
        "analysis": "CDE R2 ADeLe split-half invariance",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "min_item_discrimination": MIN_ITEM_DISCRIMINATION,
        "n_models": len(names),
        "models": names,
        "dimensions": {},
    }

    for j, d in enumerate(DEMANDS):
        levels = all_d[:, j]
        eligible = np.where(
            (levels >= 1) &
            (levels == max_d) &
            np.isfinite(discrim) &
            (discrim >= MIN_ITEM_DISCRIMINATION)
        )[0]
        level_counts = {str(lv): int(np.sum(levels[eligible] == lv)) for lv in range(1, 6)}
        taus = []
        rank_diffs = []
        for _ in range(N_SPLITS):
            left, right = [], []
            for lv in range(1, 6):
                inds = eligible[levels[eligible] == lv].copy()
                rng.shuffle(inds)
                cut = len(inds) // 2
                left.extend(inds[:cut])
                right.extend(inds[cut:])
            left = np.asarray(left, dtype=int)
            right = np.asarray(right, dtype=int)
            a = equal_level_ability(Y, levels, left)
            b = equal_level_ability(Y, levels, right)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            tau = kendalltau(a, b, variant="b").statistic
            if np.isfinite(tau):
                taus.append(float(tau))
            ra = pd.Series(a).rank(method="average", ascending=False).to_numpy()
            rb = pd.Series(b).rank(method="average", ascending=False).to_numpy()
            rank_diffs.append(float(np.mean(np.abs(ra - rb))))

        if taus:
            tau_q = np.quantile(taus, [0.025, 0.5, 0.975]).tolist()
            rd_q = np.quantile(rank_diffs, [0.025, 0.5, 0.975]).tolist()
            passes_tau = bool(tau_q[1] >= 0.90)
        else:
            tau_q = [None, None, None]
            rd_q = [None, None, None]
            passes_tau = False
        result["dimensions"][d] = {
            "eligible_items": int(len(eligible)),
            "level_counts": level_counts,
            "valid_splits": len(taus),
            "kendall_tau_q2p5_50_97p5": tau_q,
            "mean_abs_rank_difference_q2p5_50_97p5": rd_q,
            "median_tau_ge_0p90": passes_tau,
        }

    dims = list(result["dimensions"].values())
    result["summary"] = {
        "dimensions_passing_tau": int(sum(x["median_tau_ge_0p90"] for x in dims)),
        "dimensions_total": len(dims),
        "all_dimensions_pass_tau": bool(all(x["median_tau_ge_0p90"] for x in dims)),
        "dif_status": "NOT_ESTIMATED_IN_THIS_RUN",
        "dif_reason": "ADeLe supplies only 15 model subjects; the frozen programme already treats this as too few for a defensible family/size/reasoning-regime DIF rate. R2 cannot receive an overall PASS from split-half evidence alone.",
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    for d, x in result["dimensions"].items():
        print(d, x["eligible_items"], x["kendall_tau_q2p5_50_97p5"], x["median_tau_ge_0p90"])


if __name__ == "__main__":
    main()
