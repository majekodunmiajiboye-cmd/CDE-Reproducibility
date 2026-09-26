from __future__ import annotations

import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED=20260926
MODELS=['Nemotron-3 30B','Nemotron-3 120B','Nemotron-3 550B','Mistral-3.5-128B']
BENCH_N={'SWE':500,'Terminal':89}
TIER={'T0':(0,0,0),'T1':(1,0,0),'T2':(1,1,0),'T3':(0,0,1),'T4':(1,1,1)}

def clean_num(s):
    s=re.sub(r'\\textbf\{([^}]*)\}',r'\1',s)
    s=re.sub(r'\\hphantom\{[^}]*\}','',s)
    s=re.sub(r'\\textsuperscript\{\*\}','',s)
    m=re.search(r'-?\d+(?:\.\d+)?',s)
    return float(m.group()) if m else None

def parse_table(path:Path, bench:str):
    rows=[]; window=None
    for line in path.read_text().splitlines():
        if '&' not in line or 'SR (' in line: continue
        wm=re.search(r'\\multirow\{\d+\}\{\*\}\{(\d+)K\}',line)
        if wm: window=int(wm.group(1))
        parts=[x.strip() for x in line.split('&')]
        if len(parts)<10: continue
        setting=re.sub(r'\\.*','',parts[1]).strip()
        if not setting.startswith('T'): continue
        vals=[clean_num(x) for x in parts[2:10]]
        if any(v is None for v in vals): continue
        for k,m in enumerate(MODELS):
            sr,cost=vals[2*k],vals[2*k+1]
            rows.append({'benchmark':bench,'n':BENCH_N[bench],'model':m,'window_k':window,'setting_label':setting,'sr_pct':sr,'cost':cost})
    return rows

def enrich(df):
    out=df.copy(); ids=[]; desc=[]
    feats=[]
    for _,r in out.iterrows():
        label=r.setting_label
        tier='T4' if label.startswith('T4 ') else label
        e,rec,s=TIER[tier]
        planning=0 if 'w/o plan' in label else 1
        structured=0 if 'bash only' in label else 1
        sid=f"{int(r.window_k)}K_{label.replace(' ','_')}"
        ids.append(sid)
        desc.append(f"context window {int(r.window_k)}k; elision {e}; recall {rec}; summarization {s}; planning {planning}; action space {'structured tools' if structured else 'bash only'}")
        feats.append((math.log2(r.window_k/32),e,rec,s,planning,structured))
    out['setting_id']=ids; out['doc']=desc
    for j,c in enumerate(['window_log2','elision','recall','summarization','planning','structured_tools']): out[c]=[x[j] for x in feats]
    out['successes']=[int(round(p*n/100)) for p,n in zip(out.sr_pct,out.n)]
    out['failures']=out.n-out.successes
    return out

def expanded_train(df):
    rows=[]; y=[]; w=[]
    for _,r in df.iterrows():
        d=r.to_dict()
        if r.successes>0: rows.append(d); y.append(1); w.append(r.successes)
        if r.failures>0: rows.append(d); y.append(0); w.append(r.failures)
    return pd.DataFrame(rows),np.array(y),np.array(w,float)

def cat_matrix(train,test,cat_cols,num_cols=None):
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    A=enc.fit_transform(train[cat_cols]); B=enc.transform(test[cat_cols])
    if num_cols:
        sc=StandardScaler(); na=sc.fit_transform(train[num_cols]); nb=sc.transform(test[num_cols])
        A=sparse.hstack([A,sparse.csr_matrix(na)],format='csr'); B=sparse.hstack([B,sparse.csr_matrix(nb)],format='csr')
    return A,B

def fit_predict(train,test,kind):
    tr,y,w=expanded_train(train)
    if kind=='noenv': A,B=cat_matrix(tr,test,['model','benchmark'])
    elif kind=='categorical': A,B=cat_matrix(tr,test,['model','benchmark','setting_id'])
    elif kind=='coded': A,B=cat_matrix(tr,test,['model','benchmark'],['window_log2','elision','recall','summarization','planning','structured_tools'])
    elif kind=='doc':
        A0,B0=cat_matrix(tr,test,['model','benchmark'])
        hv=HashingVectorizer(n_features=256,alternate_sign=False,analyzer='char_wb',ngram_range=(3,5),norm='l2')
        Ah=hv.transform(tr.doc); Bh=hv.transform(test.doc)
        A=sparse.hstack([A0,Ah],format='csr'); B=sparse.hstack([B0,Bh],format='csr')
    else: raise ValueError(kind)
    clf=LogisticRegression(C=1.0,max_iter=5000,solver='liblinear',random_state=SEED)
    clf.fit(A,y,sample_weight=w)
    return clf.predict_proba(B)[:,list(clf.classes_).index(1)]

def agg_ll(df,p):
    p=np.clip(np.asarray(p),1e-8,1-1e-8)
    return float(-(df.successes*np.log(p)+df.failures*np.log(1-p)).sum()/df.n.sum())

def cv(df,mode):
    kinds=['noenv','categorical','coded','doc']
    store={k:[] for k in kinds}; truth=[]
    if mode=='cell': groups=sorted(set(zip(df.model,df.setting_id)))
    else: groups=sorted(df.setting_id.unique())
    for g in groups:
        mask=(df.model.eq(g[0]) & df.setting_id.eq(g[1])) if mode=='cell' else df.setting_id.eq(g)
        tr=df.loc[~mask]; te=df.loc[mask]
        truth.append(te)
        for k in kinds: store[k].extend(fit_predict(tr,te,k).tolist())
    alltest=pd.concat(truth,ignore_index=True)
    return {k:agg_ll(alltest,np.array(v)) for k,v in store.items()},alltest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-dir',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    rows=parse_table(args.source_dir/'tables/results_swe.tex','SWE')+parse_table(args.source_dir/'tables/results_terminal.tex','Terminal')
    df=enrich(pd.DataFrame(rows))
    if len(df)!=176: raise RuntimeError(f'expected 176 setting rows, got {len(df)}')
    cell,_=cv(df,'cell'); unseen,_=cv(df,'setting')
    out={'analysis':'CDE R4 Fan et al. 176-setting reanalysis','n_rows':len(df),'n_models':df.model.nunique(),'n_settings':df.setting_id.nunique(),'benchmarks':sorted(df.benchmark.unique()),'coded_features':['window_log2','elision','recall','summarization','planning','structured_tools'],'leave_one_model_setting_cell_out_log_loss':cell,'leave_one_setting_out_log_loss':unseen}
    out['frozen_checks']={'cell_coded_no_worse_than_categorical':bool(cell['coded']<=cell['categorical']),'unseen_coded_beats_noenv':bool(unseen['coded']<unseen['noenv']),'unseen_coded_beats_doc_hash':bool(unseen['coded']<unseen['doc'])}
    out['fan_component_pass']=all(out['frozen_checks'].values())
    args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=='__main__': main()
