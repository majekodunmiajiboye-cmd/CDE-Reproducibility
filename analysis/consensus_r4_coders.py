from __future__ import annotations

import json
import re
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


HARNESS = ['codex','hermes','moltis-local','nanobot','nullclaw','openclaw-local','zeroclaw-local']
FEATURES = ['observability','action_granularity','context_state_management','verification','recoverability','decomposition_control']


def load(path: Path):
    txt = path.read_text().strip()
    if txt.startswith('```'):
        txt = re.sub(r'^```(?:json)?\s*', '', txt)
        txt = re.sub(r'\s*```$', '', txt)
    return json.loads(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coder-a', type=Path, required=True)
    ap.add_argument('--coder-b', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    files = {'claude': args.coder_a, 'codex': args.coder_b}
    d = {k: load(v) for k,v in files.items()}
    for coder, obj in d.items():
        if set(obj['harnesses']) != set(HARNESS):
            raise RuntimeError(f'{coder} harness mismatch: {obj["harnesses"].keys()}')

    a=[]; b=[]; cells=[]
    consensus={}
    for h in HARNESS:
        consensus[h]={}
        for f in FEATURES:
            sa=int(d['claude']['harnesses'][h][f]['score'])
            sb=int(d['codex']['harnesses'][h][f]['score'])
            if sa not in (0,1,2) or sb not in (0,1,2): raise RuntimeError((h,f,sa,sb))
            a.append(sa); b.append(sb)
            # Outcome-blind consensus rule: retain the arithmetic mean rather than
            # adjudicating toward whichever discrete score happens to fit results.
            # This preserves coder disagreement as half-steps.
            consensus[h][f]=(sa+sb)/2.0
            cells.append({'harness':h,'feature':f,'claude':sa,'codex':sb,'consensus':(sa+sb)/2.0})
    a=np.array(a); b=np.array(b)
    exact=float(np.mean(a==b))
    kap=float(cohen_kappa_score(a,b,weights='quadratic'))
    mad=float(np.mean(np.abs(a-b)))
    out={
        'coders': {k:v['coder'] for k,v in d.items()},
        'n_cells':len(a),
        'exact_agreement':exact,
        'quadratic_weighted_kappa':kap,
        'mean_absolute_score_difference':mad,
        'consensus_rule':'arithmetic mean of the two outcome-blind ordinal scores; no post-outcome adjudication',
        'features':FEATURES,
        'harnesses':HARNESS,
        'consensus':consensus,
        'cells':cells,
    }
    p=args.out; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=='__main__': main()
