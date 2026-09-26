from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from r1_adele import DEMANDS, SEED, load_data


MIN_LEVEL_N = 20
REVERSE_EFFECT = 0.05
ALPHA = 0.05
LOGISTIC_TOL = 0.01


def bh(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any(): return out
    v = p[ok]; order = np.argsort(v); sv = v[order]
    q = sv * len(sv) / np.arange(1, len(sv)+1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    tmp = np.empty_like(v); tmp[order] = np.clip(q,0,1); out[ok]=tmp
    return out


def two_prop_reverse_p(s1,n1,s2,n2):
    # One-sided H1: higher demand level (2) has higher success than level 1.
    if min(n1,n2) <= 0: return np.nan
    p1=s1/n1; p2=s2/n2; pooled=(s1+s2)/(n1+n2)
    se=np.sqrt(max(pooled*(1-pooled)*(1/n1+1/n2),1e-12))
    z=(p2-p1)/se
    return float(1-norm.cdf(z))


def fit_link(levels, succ, counts, link):
    x=np.asarray(levels,float); s=np.asarray(succ,float); n=np.asarray(counts,float)
    overall=(s.sum()+.5)/(n.sum()+1.0)
    a0=float(np.log(overall/(1-overall)))
    def nll(par):
        a,b=par
        eta=a+b*x
        p=expit(eta) if link=='logistic' else norm.cdf(eta)
        p=np.clip(p,1e-8,1-1e-8)
        return float(-(s*np.log(p)+(n-s)*np.log(1-p)).sum())
    res=minimize(nll,[a0,-.4],method='L-BFGS-B',bounds=[(-20,20),(-20,0)])
    return res.x


def predict_link(par, x, link):
    eta=par[0]+par[1]*x
    return float(expit(eta) if link=='logistic' else norm.cdf(eta))


def ll_for_bin(s,n,p):
    p=float(np.clip(p,1e-8,1-1e-8))
    return float(-(s*np.log(p)+(n-s)*np.log(1-p)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--battery-file',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    base,Y,names=load_data(args.data_dir,args.battery_file)
    X=base[DEMANDS].to_numpy(float)
    result={}
    for j,d in enumerate(DEMANDS):
        other=np.max(np.delete(X,j,axis=1),axis=1)
        idx=np.where(X[:,j]>=other)[0]
        lv=X[idx,j].astype(int)
        levels=sorted(set(lv.tolist()))
        reverse_rows=[]
        model_curve=[]
        pooled_losses={k:0.0 for k in ['logistic','probit','isotonic','pchip']}
        pooled_n=0
        for mi,name in enumerate(names):
            y=Y[mi,idx]
            stats={L:(int(y[lv==L].sum()),int((lv==L).sum())) for L in levels}
            # raw monotonicity pairs
            for L1,L2 in zip(levels[:-1],levels[1:]):
                s1,n1=stats[L1]; s2,n2=stats[L2]
                if min(n1,n2)<MIN_LEVEL_N: continue
                diff=s2/n2-s1/n1
                reverse_rows.append({'model':name,'lo':L1,'hi':L2,'diff':diff,'p':two_prop_reverse_p(s1,n1,s2,n2)})
            losses={k:0.0 for k in pooled_losses}; ntest=0
            for held in levels:
                sh,nh=stats[held]
                if nh<MIN_LEVEL_N: continue
                train=[L for L in levels if L!=held and stats[L][1]>=MIN_LEVEL_N]
                if len(train)<3: continue
                succ=[stats[L][0] for L in train]; counts=[stats[L][1] for L in train]
                for link in ['logistic','probit']:
                    par=fit_link(train,succ,counts,link)
                    losses[link]+=ll_for_bin(sh,nh,predict_link(par,held,link))
                rates=np.array([stats[L][0]/stats[L][1] for L in train],float)
                weights=np.array([stats[L][1] for L in train],float)
                ir=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(np.array(train,float),rates,sample_weight=weights)
                p_iso=float(ir.predict([held])[0])
                losses['isotonic']+=ll_for_bin(sh,nh,p_iso)
                fitvals=ir.predict(np.array(train,float))
                pp=PchipInterpolator(np.array(train,float),fitvals,extrapolate=True)
                p_pc=float(np.clip(pp(float(held)),1e-8,1-1e-8))
                losses['pchip']+=ll_for_bin(sh,nh,p_pc)
                ntest+=nh
            if ntest:
                per_item={k:v/ntest for k,v in losses.items()}
                model_curve.append({'model':name,'n_test_item_predictions':ntest,'log_loss':per_item})
                for k,v in losses.items(): pooled_losses[k]+=v
                pooled_n+=ntest
        # BH across all adjacent pair tests within this dimension.
        q=bh([r['p'] for r in reverse_rows])
        for r,qi in zip(reverse_rows,q): r['q']=float(qi) if np.isfinite(qi) else None
        flagged=[r for r in reverse_rows if r.get('q') is not None and r['q']<ALPHA and r['diff']>=REVERSE_EFFECT]
        pair_models={}
        for r in flagged: pair_models.setdefault((r['lo'],r['hi']),set()).add(r['model'])
        reproducible=[{'lo':a,'hi':b,'n_models':len(ms),'models':sorted(ms)} for (a,b),ms in pair_models.items() if len(ms)>=2]
        pll={k:(v/pooled_n if pooled_n else float('nan')) for k,v in pooled_losses.items()}
        best_nonparam=min(pll['isotonic'],pll['pchip']) if pooled_n else float('nan')
        result[d]={
            'n_dominant_items':int(len(idx)),
            'n_raw_pair_tests':len(reverse_rows),
            'n_flagged_reverse_pairs':len(flagged),
            'reproducible_reverse_level_pairs':reproducible,
            'monotone_raw_pass':len(reproducible)==0,
            'leave_one_level_out_pooled_log_loss':pll,
            'best_monotone_nonparametric_log_loss':best_nonparam,
            'logistic_within_0p01_of_best_nonparametric':bool(np.isfinite(best_nonparam) and pll['logistic'] <= best_nonparam+LOGISTIC_TOL),
            'model_curve_scores':model_curve,
        }
    out={
        'analysis':'CDE R3 ADeLe raw monotonicity then curve-form comparison',
        'seed':SEED,
        'minimum_items_per_level':MIN_LEVEL_N,
        'raw_nonmonotonicity_flag':'higher adjacent level success >=5 percentage points, BH q<0.05; reproducible if same level pair flags in >=2 models',
        'curve_cv':'per model x dimension leave-one-demand-level-out; decreasing logistic, decreasing probit, isotonic and monotone PCHIP; pooled item-weighted log loss',
        'logistic_reporting_rule':'accept logistic form only if pooled held-out log loss <= best monotone nonparametric + 0.01',
        'dimensions':result,
        'n_dimensions_raw_monotone_pass':int(sum(v['monotone_raw_pass'] for v in result.values())),
        'n_dimensions_logistic_adequate':int(sum(v['logistic_within_0p01_of_best_nonparametric'] for v in result.values())),
    }
    args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='dimensions'},indent=2,sort_keys=True))
    for d,v in result.items():
        if not v['monotone_raw_pass'] or not v['logistic_within_0p01_of_best_nonparametric']:
            print(d,'monotone',v['monotone_raw_pass'],'logistic_ok',v['logistic_within_0p01_of_best_nonparametric'],'loss',v['leave_one_level_out_pooled_log_loss'],'reverse',v['reproducible_reverse_level_pairs'])

if __name__=='__main__': main()
