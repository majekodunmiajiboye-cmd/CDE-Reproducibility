#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
SEED=20260926

def close(a,b,tol=5e-5):
    if abs(a-b)>tol: raise AssertionError(f"{a} != {b}")

def task_boot(df,lhs,rhs,n=1000):
    y=df.y.to_numpy(float); a=np.clip(df[lhs].to_numpy(float),1e-6,1-1e-6); b=np.clip(df[rhs].to_numpy(float),1e-6,1-1e-6)
    la=-(y*np.log(a)+(1-y)*np.log(1-a)); lb=-(y*np.log(b)+(1-y)*np.log(1-b)); d=la-lb
    tasks=np.array(sorted(df.task.astype(str).unique())); key=df.task.astype(str).to_numpy()
    sums=np.array([d[key==t].sum() for t in tasks]); counts=np.array([(key==t).sum() for t in tasks],float)
    rng=np.random.default_rng(SEED); vals=[]
    for _ in range(n):
        ix=rng.integers(0,len(tasks),len(tasks)); vals.append(sums[ix].sum()/counts[ix].sum())
    return float(d.mean()), [float(x) for x in np.quantile(vals,[.025,.5,.975])]

# Headline ADeLe values.
r1=json.load(open(ROOT/'results/adele/r1_primary/r1_adele_result.json'))
close(r1['task_holdout']['relative_ll_reduction'],0.0674810586)
strong=json.load(open(ROOT/'results/adele/r1_strong_comparator/result.json'))
close(strong['comparisons']['demand_increment_over_full_panel']['relative_ll_reduction_of_rhs_vs_lhs'],0.0173836819)
nonlin=json.load(open(ROOT/'results/adele/r1_nonlinear/result.json'))
close(nonlin['bootstrap']['relative_ll_reduction'],0.0213731773)
text=json.load(open(ROOT/'results/adele/r1_text_control/result.json'))
close(text['comparisons']['text_increment_over_response']['relative_ll_reduction_of_rhs_vs_lhs'],0.0149167011)

# Benchmark-seen text-control sensitivity used in the manuscript.
oof=pd.read_csv(ROOT/'results/adele/r1_text_control/oof.csv.gz')
seen={x['task'] for x in text['fold_metadata'] if x['benchmark_seen_in_training']}
s=oof[oof.task.astype(str).isin(seen)].copy()
delta,ci=task_boot(s,'p_text','p_text_demand')
close(delta,0.0018825197); close(ci[0],0.0001500159)

# R4b direct B1-vs-CDE bootstrap used in manuscript.
# Recompute predictions from the frozen analysis helper to avoid storing another ad-hoc result.
import importlib.util
spec=importlib.util.spec_from_file_location('r4a',ROOT/'analysis/r4b_analyze_pilot.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
pr=json.load(open(ROOT/'results/r4b/full_pilot/pilot_result.json')); feats=json.load(open(ROOT/'manifests/r4b_task_features.json'))
fa={r['problem']:r for r in feats['rows'] if r['split']=='pilot'}
ma='devstral-small-2:24b-instruct-2512-q4_K_M'; mb='gemma3:12b'; rows=[]
for r in pr['results']:
    if r['model'] not in {ma,mb}: continue
    fr=fa[r['problem']]; row={'model':r['model'],'problem':r['problem'],'family':r['family'],'arm':r['arm'],'y':int(bool(r['eval'].get('semantic_pass',False))),'model_is_a':int(r['model']==ma),'arm_spec':int(r['arm']=='spec'),'arm_both':int(r['arm']=='both'),'family_cf':int(r['family']=='codeforces')}
    row.update({d:fr[d] for d in m.D_FEATURES}); rows.append(row)
df=pd.DataFrame(rows); pb=m.loto_predictions(df,'b1'); pc=m.loto_predictions(df,'cde'); boot=m.task_bootstrap_delta(df,pb,pc,1000)
close(boot['point'],-0.0998867415); close(boot['ci_2p5_50_97p5'][0],-0.1900600160)
print('All manuscript headline and supplementary values verified.')
