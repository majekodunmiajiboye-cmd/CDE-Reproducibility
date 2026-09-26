from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


SEED = 20260926
D_FEATURES = [
    "d_algorithmic_difficulty",
    "d_nl_log_tokens",
    "d_spec_log_tokens",
    "d_spec_logic_density",
    "d_interface_complexity",
]


def build_matrix(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, kind: str):
    tr = df.iloc[train_idx]
    te = df.iloc[test_idx]
    scaler = StandardScaler().fit(tr[D_FEATURES])
    dtr = scaler.transform(tr[D_FEATURES])
    dte = scaler.transform(te[D_FEATURES])

    def rows(x: pd.DataFrame, d: np.ndarray) -> np.ndarray:
        model = x["model_is_a"].to_numpy(float)[:, None]
        spec = x["arm_spec"].to_numpy(float)[:, None]
        both = x["arm_both"].to_numpy(float)[:, None]
        family = x["family_cf"].to_numpy(float)[:, None]
        scalar = d[:, [0]]
        base = np.hstack([model, spec, both, model * spec, model * both, family, scalar])
        if kind == "b1":
            return base
        # Strong profile/compatibility baseline: all five outcome-blind task
        # features plus model×demand, so D×environment cannot merely substitute
        # for ordinary task difficulty or model-specific task profile.
        profile = np.hstack([base, d, model * d])
        if kind == "profile":
            return profile
        if kind == "cde":
            return np.hstack([profile, d * spec, d * both])
        raise ValueError(kind)

    return rows(tr, dtr), rows(te, dte)


def loto_predictions(df: pd.DataFrame, kind: str) -> np.ndarray:
    p = np.empty(len(df), dtype=float)
    tasks = sorted(df["problem"].unique())
    for task in tasks:
        test_idx = np.flatnonzero(df["problem"].to_numpy() == task)
        train_idx = np.flatnonzero(df["problem"].to_numpy() != task)
        xtr, xte = build_matrix(df, train_idx, test_idx, kind)
        ytr = df.iloc[train_idx]["y"].to_numpy(int)
        if len(np.unique(ytr)) < 2:
            p[test_idx] = float(ytr.mean())
            continue
        clf = LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear", max_iter=4000,
            random_state=SEED,
        )
        clf.fit(xtr, ytr)
        p[test_idx] = clf.predict_proba(xte)[:, list(clf.classes_).index(1)]
    return np.clip(p, 1e-6, 1 - 1e-6)


def task_bootstrap_delta(df: pd.DataFrame, p_profile: np.ndarray, p_cde: np.ndarray, n=1000):
    y = df["y"].to_numpy(float)
    lp = -(y * np.log(p_profile) + (1-y) * np.log(1-p_profile))
    lc = -(y * np.log(p_cde) + (1-y) * np.log(1-p_cde))
    diff = lp - lc
    tasks = np.array(sorted(df.problem.unique()))
    sums = np.array([diff[df.problem.to_numpy()==t].sum() for t in tasks])
    counts = np.array([(df.problem.to_numpy()==t).sum() for t in tasks], float)
    rng = np.random.default_rng(SEED)
    vals = np.empty(n)
    for i in range(n):
        ix = rng.integers(0, len(tasks), size=len(tasks))
        vals[i] = sums[ix].sum() / counts[ix].sum()
    return {
        "point": float(diff.mean()),
        "ci_2p5_50_97p5": [float(x) for x in np.quantile(vals, [.025,.5,.975])],
        "n_boot": n,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot-result",type=Path,required=True)
    ap.add_argument("--task-features",type=Path,required=True)
    ap.add_argument("--model-a",required=True)
    ap.add_argument("--model-b",required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    pr=json.loads(args.pilot_result.read_text())
    feats=json.loads(args.task_features.read_text())
    fidx={r["problem"]:r for r in feats["rows"] if r["split"]=="pilot"}
    rows=[]
    for r in pr["results"]:
        if r["model"] not in {args.model_a,args.model_b}: continue
        fr=fidx[r["problem"]]
        row={
            "model":r["model"],"problem":r["problem"],"family":r["family"],"arm":r["arm"],
            "y":int(bool(r["eval"].get("semantic_pass",False))),
            "model_is_a":int(r["model"]==args.model_a),
            "arm_spec":int(r["arm"]=="spec"),"arm_both":int(r["arm"]=="both"),
            "family_cf":int(r["family"]=="codeforces"),
        }
        row.update({d:fr[d] for d in D_FEATURES}); rows.append(row)
    df=pd.DataFrame(rows)
    expected=2*3*24
    if len(df)!=expected or df.problem.nunique()!=24:
        raise RuntimeError(f"expected full 24-task x2-model x3-arm pilot ({expected} rows), got {len(df)} rows / {df.problem.nunique()} tasks")

    rates={}
    for m in [args.model_a,args.model_b]:
        rates[m]={arm:float(df[(df.model==m)&(df.arm==arm)].y.mean()) for arm in ["nl","spec","both"]}
        rates[m]["overall"]=float(df[df.model==m].y.mean())

    contrasts={}
    for arm in ["spec","both"]:
        ea=rates[args.model_a][arm]-rates[args.model_a]["nl"]
        eb=rates[args.model_b][arm]-rates[args.model_b]["nl"]
        contrasts[f"{arm}_minus_nl"]={"model_a_effect":ea,"model_b_effect":eb,"difference_in_differences_a_minus_b":ea-eb}
    selected=max(contrasts, key=lambda k:(abs(contrasts[k]["difference_in_differences_a_minus_b"]),k))
    did=float(contrasts[selected]["difference_in_differences_a_minus_b"])

    preds={k:loto_predictions(df,k) for k in ["b1","profile","cde"]}
    lls={k:float(log_loss(df.y,p,labels=[0,1])) for k,p in preds.items()}
    boot=task_bootstrap_delta(df,preds["profile"],preds["cde"],1000)

    competence={m:bool(.10<=rates[m]["overall"]<=.90) for m in [args.model_a,args.model_b]}
    gates={
        "both_models_nonfloor_nonceiling":bool(all(competence.values())),
        "selected_environment_did_abs_ge_0p10":bool(abs(did)>=.10),
        "cde_loto_ll_better_than_strong_profile_baseline":bool(lls["cde"]<lls["profile"]),
    }
    gates["proceed_to_confirmation_preparation"]=bool(all(gates.values()))
    direction=(f"{args.model_a} benefits more than {args.model_b} from {selected.replace('_minus_nl','')} relative to NL"
               if did>0 else
               f"{args.model_b} benefits more than {args.model_a} from {selected.replace('_minus_nl','')} relative to NL")
    out={
        "analysis":"CDE R4b development-pilot frozen analysis",
        "models":{"a":args.model_a,"b":args.model_b},
        "n_tasks":24,"n_rows":len(df),"arm_success_rates":rates,
        "environment_contrasts":contrasts,"selected_secondary_contrast":selected,
        "selected_difference_in_differences":did,"frozen_secondary_direction_if_gate_passes":direction,
        "loto_log_loss":lls,
        "profile_minus_cde_task_bootstrap_delta_ll":boot,
        "model_definitions":{
            "b1":"model + environment + model×environment + source family + public scalar difficulty",
            "profile":"B1 + all five frozen demand proxies + model×demand terms (strong compatibility/profile baseline)",
            "cde":"profile + demand×environment terms for spec and both vs NL; no feature selection",
        },
        "gates":gates,
        "gate_rule":"Proceed only if both models have 10-90% overall semantic success, the largest prespecified spec-vs-NL or both-vs-NL model difference-in-differences is at least 10 percentage points, and CDE improves leave-one-task-out log loss over the stronger profile baseline. Pilot bootstrap is descriptive; confirmation remains independent.",
        "seed":SEED,
    }
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=='__main__': main()
