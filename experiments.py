"""Reproduce the calibration, bounded search, confirmation and control study."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import source
from shaper import simulate
from metrics import measure, feasible, aggregate

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'data/results'


def save_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def resource(row):
    p=row['parameters']
    return (p['mu'],p['Bi']+p['Bo'],p['rho'],p['b'],p['Bi'])


def run(trace, parameters, duration):
    res=simulate(trace['times'],trace['lengths'],trace['ids'],parameters)
    return measure(trace,res,duration)


def shortlist(rows, limit=8):
    good=sorted([row for row in rows if feasible(row['metrics'])],key=resource)
    frontier=[row for row in good if not any(
        resource(other)[0]<=resource(row)[0] and resource(other)[1]<=resource(row)[1]
        and resource(other)[:2]!=resource(row)[:2] for other in good)]
    chosen=frontier[:4]
    rest=sorted([r for r in good if r not in chosen],key=lambda r:(r['metrics']['bad_5'],r['metrics']['loss'],resource(r)))
    return (chosen+rest)[:limit]


def generate(model,duration,seed,kind='onoff',rate=None):
    return source.generate(model,duration,seed,kind=kind,matched_rate=rate)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true',help='reuse checked stages already saved in data/results')
    args=parser.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True)
    model=source.fit()
    started=time.perf_counter()
    stage=RESULTS/'calibration.json'
    if args.resume and stage.exists():
        calibration=json.loads(stage.read_text())
    else:
        rows=[]; burst_bytes=[]
        for seed in range(2026111300,2026111310):
            tr=generate(model,3600.,seed)
            bs=[c['bytes'] for c in tr['cycles'] if c['complete_on']]
            burst_bytes.extend(bs)
            rows.append(dict(seed=seed,packets=len(tr['times']),bytes=int(tr['lengths'].sum()),complete_on=len(bs)))
        calibration=dict(r=sum(row['bytes'] for row in rows)/36000.,M=float(np.mean(burst_bytes)),
                         runs=rows,complete_bursts=len(burst_bytes),duration_s=3600.)
        save_json(stage,calibration)
    r,M=calibration['r'],calibration['M']; Lmax=1452
    print(f'Calibration r={r:.3f} B/s, M={M:.1f} bytes',flush=True)
    tr=generate(model,900.,2026111400)
    stage=RESULTS/'screening.json'
    if args.resume and stage.exists():
        screen=json.loads(stage.read_text())
    else:
        screen=[]
        def screen_grid(rate_scale=1.,memory_scale=1.):
            for a,c,bm,sm,f in itertools.product((1.1,1.5,2.),(2.,4.,8.),(0.,.25,1.),(.5,1.,2.),(.25,.5,.75)):
                S=math.ceil(sm*M*memory_scale)
                p=dict(rho=a*r*rate_scale,mu=c*r*rate_scale,b=math.ceil(max(Lmax,bm*M)),Bi=math.floor(f*S),Bo=S-math.floor(f*S))
                score=run(tr,p,900.)
                screen.append(dict(id=f'C{len(screen):03d}',parameters=p,metrics=score,stage='coarse' if rate_scale==1 else 'expanded'))
                if len(screen)%27==0:print('Screened',len(screen),flush=True)
        screen_grid()
        if not any(feasible(row['metrics']) for row in screen):screen_grid(2.,2.)
        initial=shortlist(screen)
        # Refine mu only: four interior points below each of up to eight promising designs.
        tried={tuple(sorted(row['parameters'].items())) for row in screen}
        extra=[]
        for row in initial:
            p=row['parameters']; lower=max([x['parameters']['mu'] for x in screen if x['parameters']['mu']<p['mu']],default=r)
            for mu in np.linspace(lower,p['mu'],6)[1:-1]:
                q=dict(p,mu=float(mu)); key=tuple(sorted(q.items()))
                if q['mu']<q['rho'] or key in tried:continue
                tried.add(key)
                extra.append(dict(id=f'R{len(extra):03d}',parameters=q,metrics=run(tr,q,900.),stage='refined'))
        screen.extend(extra)
        save_json(stage,screen)
    candidates=shortlist(screen)
    print('Screening complete:',len(screen),'configurations;',len(candidates),'shortlisted',flush=True)
    if not candidates:raise RuntimeError('No screening-feasible design in the bounded search; inspect screening.json.')
    # Retain a less provisioned tested neighbor for controls even if it is not feasible.
    stage=RESULTS/'evaluation.json'
    if args.resume and stage.exists():
        evaluated=json.loads(stage.read_text())
        if [x['id'] for x in evaluated]!=[x['id'] for x in candidates]:raise ValueError('Saved shortlist differs; do not mix studies.')
    else:
        evaluated=[dict(id=c['id'],parameters=c['parameters'],runs=[]) for c in candidates]
        for j,seed in enumerate(range(2026111500,2026111520)):
            tt=generate(model,3600.,seed)
            for candidate in evaluated:
                candidate['runs'].append(dict(seed=seed,**run(tt,candidate['parameters'],3600.)))
            print('Independent evaluation',j+1,'/20',flush=True)
        for c in evaluated:c['summary']=aggregate(c['runs'])
        save_json(stage,evaluated)
    supported=sorted([c for c in evaluated if c['summary']['interval_supported']],key=resource)
    observed=sorted([c for c in evaluated if c['summary']['observed_feasible']],key=resource)
    eligible=supported or observed
    if not eligible:raise RuntimeError('No independently observed-feasible design; report study limits instead of inventing success.')
    chosen=eligible[0]
    # Fix winner before confirmation; do not re-optimize using confirmation samples.
    others=sorted([c for c in evaluated if c['id']!=chosen['id']],key=lambda c:(not c['summary']['observed_feasible'],resource(c)))
    confirm_designs=[chosen]+others[:1]
    stage=RESULTS/'confirmation.json'
    if args.resume and stage.exists():
        confirmation=json.loads(stage.read_text())
    else:
        confirmation=[dict(id=c['id'],parameters=c['parameters'],runs=[]) for c in confirm_designs]
        for j,seed in enumerate(range(2026111600,2026111610)):
            tt=generate(model,7200.,seed)
            for c in confirmation:c['runs'].append(dict(seed=seed,**run(tt,c['parameters'],7200.)))
            print('Horizon confirmation',j+1,'/10',flush=True)
        for c in confirmation:c['summary']=aggregate(c['runs'])
        save_json(stage,confirmation)
    less=[row for row in screen if resource(row)[:2]<resource(chosen)[:2] and row['parameters']['mu']<=chosen['parameters']['mu']]
    neighbor=min(less,key=lambda row:(abs(row['parameters']['mu']-chosen['parameters']['mu'])/r,
        abs((row['parameters']['Bi']+row['parameters']['Bo'])-(chosen['parameters']['Bi']+chosen['parameters']['Bo']))/M,
        abs(row['parameters']['rho']-chosen['parameters']['rho'])/r,abs(row['parameters']['b']-chosen['parameters']['b'])/M)) if less else others[0]
    stage=RESULTS/'controls.json'
    if args.resume and stage.exists():
        controls=json.loads(stage.read_text())
    else:
        controls=[]
        for kind in ('onoff','poisson'):
            entries=[dict(id=c['id'],parameters=c['parameters'],model=kind,runs=[]) for c in (chosen,neighbor)]
            for j,seed in enumerate(range(2026111700,2026111720)):
                tt=generate(model,3600.,seed,kind,r if kind=='poisson' else None)
                for c in entries:c['runs'].append(dict(seed=seed,**run(tt,c['parameters'],3600.)))
                if (j+1)%5==0:print(kind,'control',j+1,'/20',flush=True)
            for c in entries:c['summary']=aggregate(c['runs'])
            controls.extend(entries)
        save_json(stage,controls)
    # Required trace pair: deterministic first evaluation realization, not cherry-picked.
    tt=generate(model,3600.,2026111500)
    result=simulate(tt['times'],tt['lengths'],tt['ids'],chosen['parameters'])
    traces=ROOT/'data/traces';traces.mkdir(parents=True,exist_ok=True)
    source.write_trace(traces/'Input_Trace_File.txt',tt)
    source.write_trace(traces/'Output_Trace_File.txt',tt,times=result['completion'],mask=result['drop']==0)
    save_json(ROOT/'config/selected.json',chosen['parameters'])
    save_json(ROOT/'config/experiment.json',dict(calibration_seeds=[2026111300,2026111309],screen_seed=2026111400,
        evaluation_seeds=[2026111500,2026111519],confirmation_seeds=[2026111600,2026111609],control_seeds=[2026111700,2026111719],
        loss_target=.001,bad_target=.05,deadline_s=5.,rate_grid=[1.1,1.5,2.],mu_grid=[2,4,8],bucket_grid=[0,.25,1],
        memory_grid=[.5,1,2],split_grid=[.25,.5,.75],refinement_max=32,selection='interval-supported if available, otherwise observed; then mu,total memory,rho,b,Bi'))
    # Diagnostic curves on the first full ON and its following OFF from the screening seed.
    selected_sim=simulate(tr['times'],tr['lengths'],tr['ids'],chosen['parameters'],record_events=True)
    cycle=next(c for c in tr['cycles'] if c['start']>=0 and c['end']<=900.)
    arrays=dict(times=tr['times'],lengths=tr['lengths'],completion=selected_sim['completion'],drop=selected_sim['drop'],
                interval=np.array([cycle['start'],cycle['end']]))
    for key,value in selected_sim['events'].items():arrays['event_'+key]=value
    np.savez_compressed(RESULTS/'diagnostic.npz',**arrays)
    # Isolated diagnostics demonstrate limits without altering the main search.
    p=chosen['parameters']; total=int(tr['lengths'].sum())
    isolated_lbf=dict(p,b=total,Bi=total)
    isolated_tbf=dict(p,Bo=total,mu=max(p['mu'],1000*r))
    diagnostics={name:run(tr,pars,900.) for name,pars in [('isolated_lbf',isolated_lbf),('isolated_tbf',isolated_tbf),('composite',p)]}
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in traces.glob('*.txt')}
    summary=dict(selected=chosen['id'],selection_evidence='nominal intervals' if supported else 'observed fractions; no joint certificate',
                 parameters=p,calibration=calibration,screening_configurations=len(screen),shortlist=[c['id'] for c in evaluated],
                 neighbor=neighbor['id'],confirmation_passed=confirmation[0]['summary']['observed_feasible'],diagnostics=diagnostics,
                 trace_seed=2026111500,trace_duration_s=3600.,trace_metrics=measure(tt,result,3600.),trace_sha256=hashes,
                 python=platform.python_version(),elapsed_this_invocation_s=time.perf_counter()-started)
    save_json(RESULTS/'summary.json',summary)
    print('FINISHED',json.dumps(dict(selected=chosen['id'],parameters=p,confirmation_passed=summary['confirmation_passed'])),flush=True)


if __name__=='__main__':main()
