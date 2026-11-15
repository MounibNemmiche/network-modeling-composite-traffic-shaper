"""Packet-based service metrics and independent-run summaries."""
import numpy as np
from scipy.stats import t


def measure(trace, result, duration):
    lengths = trace['lengths']
    n = len(lengths)
    ok = result['drop'] == 0
    delay = result['completion'][ok] - trace['times'][ok]
    end = max(duration, float(np.max(result['completion'][ok])) if ok.any() else duration)
    assert n == int(ok.sum()) + int((result['drop'] == 1).sum()) + int((result['drop'] == 2).sum())
    assert np.all(delay >= -1e-10)
    out = dict(packets=n, delivered=int(ok.sum()), input_drops=int((result['drop']==1).sum()),
               output_drops=int((result['drop']==2).sum()), bytes_in=int(lengths.sum()),
               bytes_out=int(lengths[ok].sum()), duration_s=duration, drain_s=end-duration,
               offered_Bps=float(lengths.sum()/duration), throughput_Bps=float(lengths[ok].sum()/end),
               loss=float((n-int(ok.sum()))/n) if n else None,
               byte_loss=float(lengths[~ok].sum()/lengths.sum()) if n else None,
               mean_delay=float(delay.mean()) if len(delay) else None,
               p95_delay=float(np.quantile(delay,.95)) if len(delay) else None,
               max_qi=int(result['max_qi']), max_qo=int(result['max_qo']))
    for d in (1,5,10):
        out[f'bad_{d}'] = float((n-int(ok.sum())+np.sum(delay>d))/n) if n else None
    return out


def feasible(score):
    return score['loss'] is not None and score['loss'] <= .001 and score['bad_5'] <= .05


def aggregate(rows):
    output = {}
    for key in ('loss','bad_1','bad_5','bad_10','mean_delay','p95_delay','offered_Bps','throughput_Bps','drain_s','byte_loss'):
        vals = np.array([r[key] for r in rows if r[key] is not None],dtype=float)
        if not len(vals):
            output[key] = None
            continue
        sd = float(vals.std(ddof=1)) if len(vals)>1 else None
        half = float(t.ppf(.975,len(vals)-1)*sd/np.sqrt(len(vals))) if len(vals)>1 else None
        output[key] = dict(mean=float(vals.mean()),sd=sd,lo=float(vals.mean()-half) if half is not None else None,
                           hi=float(vals.mean()+half) if half is not None else None,min=float(vals.min()),max=float(vals.max()),
                           interval_informative=bool(sd is not None and sd>0))
    output['runs'] = len(rows)
    output['packets'] = sum(r['packets'] for r in rows)
    output['observed_feasible'] = feasible({k:output[k]['mean'] if output[k] else None for k in ('loss','bad_5')}) if output['loss'] and output['bad_5'] else False
    # A degenerate zero-loss sample is an observation, never a confidence certificate.
    output['interval_supported'] = all(output[k] and output[k]['interval_informative'] and output[k]['hi']<=target
                                       for k,target in [('loss',.001),('bad_5',.05)])
    return output
