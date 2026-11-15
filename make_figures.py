"""Regenerate report figures solely from saved numerical results."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'data/results'


def token_curve(event_times, post_tokens, rho, capacity, begin, end):
    """Draw continuous refill, saturation, and vertical post-event deductions.

    The event log stores the state after all actions at an instant. Connecting
    those samples directly would smear packet deductions across quiet periods.
    The right endpoint shows its left limit, excluding actions at ``end``.
    """
    event_times=np.asarray(event_times)
    post_tokens=np.asarray(post_tokens)
    previous=np.searchsorted(event_times,begin,side='left')-1
    initial=(min(capacity,post_tokens[previous]+rho*(begin-event_times[previous]))
             if previous>=0 else capacity)
    xs,ys=[float(begin)],[float(initial)]

    def refill_to(time):
        old_time,old_tokens=xs[-1],ys[-1]
        cap_time=old_time+max(0.,capacity-old_tokens)/rho
        if old_time<cap_time<time:
            xs.append(float(cap_time));ys.append(float(capacity))
        xs.append(float(time));ys.append(float(min(capacity,old_tokens+rho*(time-old_time))))

    for i in np.flatnonzero((event_times>=begin)&(event_times<end)):
        refill_to(event_times[i])
        xs.append(float(event_times[i]));ys.append(float(post_tokens[i]))
    refill_to(end)
    return np.asarray(xs),np.asarray(ys)


def queue_curve(event_times, values, begin, end):
    """Return post-step vertices spanning the complete half-open interval."""
    event_times=np.asarray(event_times)
    values=np.asarray(values)
    previous=np.searchsorted(event_times,begin,side='right')-1
    initial=values[previous] if previous>=0 else 0
    inside=(event_times>begin)&(event_times<end)
    xs=np.r_[begin,event_times[inside],end]
    states=np.r_[initial,values[inside]]
    return xs,np.r_[states,states[-1]]


def interval_rate(times, lengths, bins):
    """Count all work timestamped in [bins[0], bins[-1]), in Mbit/s."""
    times=np.asarray(times)
    lengths=np.asarray(lengths)
    inside=(times>=bins[0])&(times<bins[-1])
    return np.histogram(times[inside],bins,weights=lengths[inside])[0]*8/np.diff(bins)/1e6


def main():
    out=ROOT/'figure';out.mkdir(exist_ok=True)
    summary=json.loads((RESULTS/'summary.json').read_text())
    rows=json.loads((RESULTS/'screening.json').read_text())
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.grid':True,'grid.alpha':.18})
    z=np.load(RESULTS/'diagnostic.npz')
    begin,end=z['interval']; width=.1
    bins=np.r_[np.arange(begin,end,width),end]
    centers=(bins[:-1]+bins[1:])/2-begin
    ti=z['times'];lengths=z['lengths']; ok=z['drop']==0
    incoming=interval_rate(ti,lengths,bins)
    outgoing=interval_rate(z['completion'][ok],lengths[ok],bins)
    fig,axes=plt.subplots(3,1,figsize=(8.1,3.8),sharex=True,constrained_layout=True)
    axes[0].stairs(incoming,bins-begin,color='.55',lw=1.2,label='Input')
    axes[0].stairs(outgoing,bins-begin,color='#126b8c',lw=1.5,label='Output')
    axes[0].set_ylabel('Rate (Mbit/s)');axes[0].legend(loc='upper right',ncol=2)
    et=z['event_time']
    qit,qiv=queue_curve(et,z['event_qi'],begin,end)
    qot,qov=queue_curve(et,z['event_qo'],begin,end)
    axes[1].step(qit-begin,qiv/1e6,where='post',label='Input queue',color='#bb6633')
    axes[1].step(qot-begin,qov/1e6,where='post',label='Output queue',color='#126b8c')
    axes[1].set_ylabel('Waiting (MB)');axes[1].legend(loc='upper right',ncol=2)
    p=summary['parameters']
    token_times,tokens=token_curve(et,z['event_tokens'],p['rho'],p['b'],begin,end)
    axes[2].plot(token_times-begin,tokens/1e6,color='#47734c',lw=1.)
    axes[2].set_ylabel('Tokens (million)');axes[2].set_xlabel('Time since first complete ON begins (s)')
    axes[2].set_xlim(0,end-begin)
    for ext in ('pdf','png'):fig.savefig(out/f'mechanism.{ext}',dpi=180)
    plt.close(fig)
    with (out/'rate-curves.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['time_since_on_s','input_Mbps','output_Mbps']);w.writerows(zip(centers,incoming,outgoing))

    fig,axes=plt.subplots(1,2,figsize=(8.1,3.1),constrained_layout=True)
    good=np.array([row['metrics']['loss']<=.001 and row['metrics']['bad_5']<=.05 for row in rows])
    xs=np.array([row['parameters']['mu']*8/1e6 for row in rows]);ys=np.array([(row['parameters']['Bi']+row['parameters']['Bo'])/1e6 for row in rows])
    for mask,color,label,marker in [(~good,'.72','Fails screening','x'),(good,'#126b8c','Passes screening','o')]:
        axes[0].scatter(xs[mask],ys[mask],c=color,s=25,marker=marker,alpha=.65,label=label)
    p=summary['parameters'];axes[0].scatter([p['mu']*8/1e6],[(p['Bi']+p['Bo'])/1e6],s=130,marker='*',c='#b55428',label='Selected after evaluation',zorder=4)
    axes[0].set_xlabel('LBF rate (Mbit/s)');axes[0].set_ylabel('Total waiting memory (MB)')
    axes[0].legend(fontsize=7,loc='best')
    # Select an actual coarse-grid comparison with identical rho,b,S,mu across splits.
    coarse=[row for row in rows if row['stage']=='coarse']
    groups={}
    for row in coarse:
        q=row['parameters'];key=(q['rho'],q['b'],q['mu'],q['Bi']+q['Bo'])
        groups.setdefault(key,[]).append(row)
    valid=[g for g in groups.values() if len(g)==3]
    # Prespecified proximity to selected design, with variation as a deterministic tie-break.
    def group_distance(g):
        q=g[0]['parameters'];r=summary['calibration']['r'];M=summary['calibration']['M']
        return (abs(q['rho']-p['rho'])/r+abs(q['b']-p['b'])/M+abs(q['mu']-p['mu'])/r+abs(q['Bi']+q['Bo']-p['Bi']-p['Bo'])/M,)
    comparison=min(valid,key=group_distance)
    comparison.sort(key=lambda row:row['parameters']['Bi'])
    f=np.array([row['parameters']['Bi']/(row['parameters']['Bi']+row['parameters']['Bo']) for row in comparison])
    loss=[100*row['metrics']['loss'] for row in comparison]
    late=[100*row['metrics']['bad_5'] for row in comparison]
    axes[1].plot(f,late,'o-',c='#126b8c',label='Lost or >5 s late')
    axes[1].plot(f,loss,'s--',c='#bb6633',label='Lost')
    axes[1].axhline(5,color='#126b8c',lw=.8,ls=':');axes[1].axhline(.1,color='#bb6633',lw=.8,ls=':')
    axes[1].set_xlabel('Input share of fixed waiting memory');axes[1].set_ylabel('Generated packets (%)')
    axes[1].set_xticks([.25,.5,.75]);axes[1].legend(fontsize=8)
    q=comparison[0]['parameters']
    axes[1].set_title(f"S={(q['Bi']+q['Bo'])/1e6:.2f} MB; "+r'$\mu$'+f"={q['mu']*8/1e6:.2f} Mbit/s",fontsize=9)
    for ext in ('pdf','png'):fig.savefig(out/f'resources.{ext}',dpi=180)
    plt.close(fig)
    # Retain exact comparison provenance and numerical curves for inspection.
    (out/'figure-data.json').write_text(json.dumps(dict(buffer_comparison=comparison,diagnostic_interval_s=[float(begin),float(end)],
        diagnostic_seed=2026111400,selection=summary['selected']),indent=2)+'\n',encoding='utf-8')
    print('Figures written to',out)


if __name__=='__main__':main()
