from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from r1_adele import DEMANDS, SEED, clip_logit, load_data


EMBED_MODEL = "nomic-embed-text:latest"
TEXT_DIMS = 18


def embed_questions(texts: list[str], cache: Path, batch_size: int = 64) -> np.ndarray:
    if cache.exists():
        arr = np.load(cache)["embeddings"]
        if len(arr) != len(texts):
            raise RuntimeError("embedding cache row count mismatch")
        return arr.astype(np.float32, copy=False)

    blocks = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = {
            "model": EMBED_MODEL,
            "input": batch,
            "truncate": True,
            "keep_alive": "30m",
        }
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/embed",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1800) as r:
            out = json.loads(r.read())
        block = np.asarray(out["embeddings"], dtype=np.float32)
        if len(block) != len(batch):
            raise RuntimeError("embedding API returned wrong batch size")
        blocks.append(block)
        if start % (batch_size * 20) == 0:
            print(f"embedded {min(start+batch_size,len(texts))}/{len(texts)}", flush=True)
    arr = np.vstack(blocks)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, embeddings=arr)
    return arr


def model(numeric_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("benchmark", OneHotEncoder(handle_unknown="ignore"), ["benchmark"]),
    ])
    return Pipeline([
        ("features", pre),
        ("logit", LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=4000,
            random_state=SEED,
        )),
    ])


def predict(train: pd.DataFrame, test: pd.DataFrame, numeric_cols: list[str]) -> np.ndarray:
    if train.y.nunique() < 2:
        return np.full(len(test), float(train.y.mean()))
    cols = numeric_cols + ["benchmark"]
    mdl = model(numeric_cols)
    mdl.fit(train[cols], train.y)
    return mdl.predict_proba(test[cols])[:, list(mdl.named_steps["logit"].classes_).index(1)]


def score(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    return {
        "log_loss": float(log_loss(y, p, labels=[0,1])),
        "brier": float(brier_score_loss(y, p)),
    }


def bootstrap(pred: pd.DataFrame, lhs: str, rhs: str, n_boot=1000):
    """LL(lhs)-LL(rhs); positive means rhs is better."""
    y = pred.y.to_numpy(float)
    a = np.clip(pred[f"p_{lhs}"].to_numpy(float), 1e-6, 1-1e-6)
    b = np.clip(pred[f"p_{rhs}"].to_numpy(float), 1e-6, 1-1e-6)
    la = -(y*np.log(a)+(1-y)*np.log(1-a))
    lb = -(y*np.log(b)+(1-y)*np.log(1-b))
    diff = la-lb
    tasks = np.array(sorted(pred.task.astype(str).unique()))
    sums = np.array([diff[pred.task.astype(str).to_numpy()==t].sum() for t in tasks])
    counts = np.array([(pred.task.astype(str).to_numpy()==t).sum() for t in tasks], float)
    rng = np.random.default_rng(SEED)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, len(tasks), len(tasks))
        vals[i] = sums[ix].sum()/counts[ix].sum()
    point = float(diff.mean())
    return {
        "lhs": lhs, "rhs": rhs,
        "delta_ll_lhs_minus_rhs": point,
        "ci_2p5_50_97p5": [float(x) for x in np.quantile(vals,[.025,.5,.975])],
        "relative_ll_reduction_of_rhs_vs_lhs": point/float(la.mean()),
        "n_boot": n_boot, "seed": SEED,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--battery-file", type=Path, required=True)
    ap.add_argument("--embedding-cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base, Y, names = load_data(args.data_dir, args.battery_file)
    raw = pd.read_csv(args.battery_file, usecols=["instance_id","question"], low_memory=False)
    qmap = raw.set_index(raw.instance_id.astype(str))["question"]
    questions = [str(qmap.loc[str(x)]) if str(x) in qmap.index else "" for x in base.instance_id]
    emb = embed_questions(questions, args.embedding_cache)
    if emb.shape[1] != 768:
        raise RuntimeError(f"expected 768-d nomic embeddings, got {emb.shape}")

    groups = sorted(base.task.astype(str).unique())
    rows=[]
    fold_meta=[]
    for gi,g in enumerate(groups,1):
        test_mask = base.task.astype(str).eq(g).to_numpy()
        tr_idx=np.flatnonzero(~test_mask); te_idx=np.flatnonzero(test_mask)
        pca=PCA(n_components=TEXT_DIMS, svd_solver="randomized", random_state=SEED)
        ztr=pca.fit_transform(emb[tr_idx]); zte=pca.transform(emb[te_idx])
        bench=str(base.iloc[te_idx[0]].benchmark)
        benchmark_seen=bool((base.iloc[tr_idx].benchmark.astype(str)==bench).any())
        fold_meta.append({"task":g,"benchmark":bench,"benchmark_seen_in_training":benchmark_seen,
                          "text_pca_explained_variance":float(pca.explained_variance_ratio_.sum()),
                          "n_test_items":int(len(te_idx))})

        for mi,name in enumerate(names):
            panel_idx=[j for j in range(len(names)) if j!=mi]
            panel=Y[panel_idx].T.astype(float)
            panel_p=(panel.sum(axis=1)+.5)/(len(panel_idx)+1.0)
            frame=base[["instance_id","benchmark","task"]+DEMANDS].copy()
            frame["panel_logit"]=clip_logit(panel_p); frame["y"]=Y[mi]
            panel_cols=[]
            for j in range(panel.shape[1]):
                c=f"panel_resp_{j:02d}"; frame[c]=panel[:,j]; panel_cols.append(c)
            text_cols=[]
            for k in range(TEXT_DIMS):
                c=f"text_pc_{k:02d}"; text_cols.append(c)
                frame.loc[tr_idx,c]=ztr[:,k]; frame.loc[te_idx,c]=zte[:,k]

            train=frame.iloc[tr_idx]; test=frame.iloc[te_idx]
            base_cols=["panel_logit"]+panel_cols
            specs={
                "response":base_cols,
                "demand":base_cols+DEMANDS,
                "text":base_cols+text_cols,
                "text_demand":base_cols+text_cols+DEMANDS,
            }
            pp={k:predict(train,test,v) for k,v in specs.items()}
            for pos,idx in enumerate(te_idx):
                rows.append({"model":name,"instance_id":frame.iloc[idx].instance_id,
                             "benchmark":frame.iloc[idx].benchmark,"task":frame.iloc[idx].task,
                             "y":int(frame.iloc[idx].y),
                             **{f"p_{k}":float(v[pos]) for k,v in pp.items()}})
        print(f"fold {gi:02d}/{len(groups)} {g}", flush=True)

    pred=pd.DataFrame(rows)
    pred.to_csv(args.out_dir/"r1_text_control_oof.csv.gz",index=False,compression="gzip")
    metrics={k:score(pred.y,pred[f"p_{k}"]) for k in ["response","demand","text","text_demand"]}
    comps={
        "demand_increment_over_response":bootstrap(pred,"response","demand"),
        "text_increment_over_response":bootstrap(pred,"response","text"),
        "demand_vs_text":bootstrap(pred,"text","demand"),
        "demand_increment_over_text":bootstrap(pred,"text","text_demand"),
        "text_increment_over_demand":bootstrap(pred,"demand","text_demand"),
    }
    result={
        "analysis":"CDE submission-readiness generic text control",
        "status":"supplementary reviewer control; exact analysis frozen before text-control outcomes were run",
        "n_models":len(names),"n_items":int(len(base)),"n_task_families":len(groups),
        "text_representation":{
            "source":"ADeLe item `question` field only",
            "embedding_model":EMBED_MODEL,
            "ollama_model_id":"0a109f422b47",
            "architecture":"nomic-bert","parameters":"137M","quantization":"F16",
            "raw_embedding_dims":768,"reduced_dims":TEXT_DIMS,
            "reduction":"PCA fitted within each leave-one-task-family-out training fold; held-out text is transform-only",
        },
        "model":"L2 logistic regression, C=1, liblinear, same response-profile and benchmark features in every arm",
        "metrics":metrics,"comparisons":comps,"fold_metadata":fold_meta,
        "benchmark_identity_check":{
            "n_folds_total":len(fold_meta),
            "n_folds_benchmark_unseen_in_training":sum(not x["benchmark_seen_in_training"] for x in fold_meta),
            "encoding":"OneHotEncoder(handle_unknown='ignore'); unseen held-out benchmark maps to the all-zero benchmark block",
        },
        "primary_interpretation_rule":"Demand-specific signal beyond generic text is supported only if LL(text)-LL(text+demand) has a 95% task-family bootstrap CI entirely >0. Direct demand-vs-text performance is reported separately.",
    }
    (args.out_dir/"r1_text_control_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
