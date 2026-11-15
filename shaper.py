"""Packet-level token-bucket shaper followed by a finite FIFO output server.

The two stages can be evaluated separately because output drops do not apply
backpressure or refund tokens. Both stages advance in logical simulated time.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path

import numpy as np


def _inputs(times, lengths, ids, parameters):
    times = np.asarray(times, dtype=np.float64)
    lengths_raw, ids_raw = np.asarray(lengths), np.asarray(ids)
    if times.ndim != 1 or lengths_raw.ndim != 1 or ids_raw.ndim != 1:
        raise ValueError("Trace arrays must be one-dimensional")
    if not (len(times) == len(lengths_raw) == len(ids_raw)):
        raise ValueError("Trace arrays must have equal lengths")
    if not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError("Times must be finite, nonnegative and nondecreasing")
    for values, name in [(lengths_raw, "lengths"), (ids_raw, "identifiers")]:
        if values.size and (values.dtype.kind not in "iu" or np.any(values <= 0)):
            raise ValueError(f"Packet {name} must be positive integers")
    lengths = lengths_raw.astype(np.int64, copy=False)
    ids = ids_raw.astype(np.int64, copy=False)
    if np.any(lengths <= 0) or np.any(ids <= 0):
        raise ValueError("Packet integers exceed supported signed 64-bit range")
    if len(ids) > 1 and not np.all(np.diff(ids) > 0) and len(np.unique(ids)) != len(ids):
        raise ValueError("Packet identifiers must be unique")
    p = dict(parameters)
    for key in ["rho", "b", "mu"]:
        if key not in p or isinstance(p[key], bool):
            raise ValueError(f"Missing or invalid {key}")
        p[key] = float(p[key])
        if not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ["Bi", "Bo"]:
        if key not in p or isinstance(p[key], bool) or not isinstance(p[key], (int, np.integer)) or p[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer byte capacity")
        p[key] = int(p[key])
    if len(lengths) and p["b"] < int(lengths.max()):
        raise ValueError("Token bucket capacity b must accommodate the largest input packet")
    return times, lengths, ids, p


def simulate(times, lengths, ids, parameters, record_events=False):
    """Return per-input-packet timing arrays and drop codes (0/1/2).

    Drop codes mean delivered, input overflow, and output overflow respectively.
    Missing times are NaN. Output capacity counts waiting packets only. At ties,
    existing input releases precede new arrivals, and output completions precede
    releases. Optional logs contain state after all actions at each timestamp.
    """
    times, lengths, ids, p = _inputs(times, lengths, ids, parameters)
    n = len(times)
    rho, capacity, mu = p["rho"], p["b"], p["mu"]
    release = np.full(n, np.nan)
    start = np.full(n, np.nan)
    completion = np.full(n, np.nan)
    drop = np.zeros(n, dtype=np.int8)
    tb_log = [(0.0, capacity, 0)] if record_events else None
    out_log = [(0.0, 0)] if record_events else None

    # Stage 1: admissions and token-ready events only. A waiting head is always
    # considered before a new arrival at its exact ready time.
    queue = deque()
    occupancy = max_qi = 0
    tokens, now, i = capacity, 0.0, 0
    eps = np.finfo(np.float64).eps
    while i < n or queue:
        arrival = float(times[i]) if i < n else math.inf
        if queue:
            head_length = int(lengths[queue[0]])
            deficit = max(0.0, head_length - tokens)
            ready = now + deficit / rho
            if deficit > 0 and ready == now:
                ready = math.nextafter(now, math.inf)
        else:
            ready = math.inf
        event_time = min(arrival, ready)
        if not math.isfinite(event_time):
            raise ValueError("Rates/timestamps cannot represent a finite service event")
        tokens = min(capacity, tokens + rho * (event_time - now))
        now = event_time
        if ready <= arrival:
            j = queue.popleft()
            size = int(lengths[j])
            # The computed ready time can round by an ulp. Correct only that
            # machine-scale error; a significant deficit indicates a bug.
            tolerance = 8 * eps * max(capacity, size, rho * max(1.0, abs(now)))
            if tokens + tolerance < size:
                raise ArithmeticError("Computed token-ready event has insufficient credit")
            tokens = max(0.0, tokens - size)
            occupancy -= size
            release[j] = now
        else:
            size = int(lengths[i])
            tolerance = 8 * eps * max(capacity, size, rho * max(1.0, abs(now)))
            if not queue and tokens + tolerance >= size:
                tokens = max(0.0, tokens - size)
                release[i] = now
            elif occupancy + size <= p["Bi"]:
                queue.append(i)
                occupancy += size
                max_qi = max(max_qi, occupancy)
            else:
                drop[i] = 1
            i += 1
        if record_events:
            tb_log.append((now, tokens, occupancy))

    # Stage 2: releases are in input order. Completion times for accepted FIFO
    # packets are fixed at admission; the deque tracks which are still present.
    # Popping a completion advances the next waiting packet into service.
    pending = deque()
    occupancy = max_qo = 0
    for j in np.flatnonzero(np.isfinite(release)):
        arrival = float(release[j])
        while pending and completion[pending[0]] <= arrival:
            finished = pending.popleft()
            if pending:
                occupancy -= int(lengths[pending[0]])
            if record_events:
                out_log.append((float(completion[finished]), occupancy))
        size = int(lengths[j])
        if not pending:
            start[j] = arrival
        elif occupancy + size <= p["Bo"]:
            start[j] = completion[pending[-1]]
            occupancy += size
            max_qo = max(max_qo, occupancy)
        else:
            drop[j] = 2
            if record_events:
                out_log.append((arrival, occupancy))
            continue
        completion[j] = start[j] + size / mu
        if not math.isfinite(completion[j]) or completion[j] <= start[j]:
            raise ValueError("Rate/timestamps cannot represent a positive packet transmission")
        pending.append(j)
        if record_events:
            out_log.append((arrival, occupancy))
    if record_events:
        while pending:
            finished = pending.popleft()
            if pending:
                occupancy -= int(lengths[pending[0]])
            out_log.append((float(completion[finished]), occupancy))

    result = {"release": release, "start": start, "completion": completion,
              "drop": drop, "max_qi": max_qi, "max_qo": max_qo,
              "events": None}
    if record_events:
        tb = np.asarray(tb_log, dtype=float)
        lb = np.asarray(out_log, dtype=float)
        event_times = np.union1d(tb[:, 0], lb[:, 0])
        ti = np.searchsorted(tb[:, 0], event_times, side="right") - 1
        oi = np.searchsorted(lb[:, 0], event_times, side="right") - 1
        result["events"] = {
            "time": event_times,
            "tokens": np.minimum(capacity, tb[ti, 1] + rho * (event_times - tb[ti, 0])),
            "qi": tb[ti, 2].astype(np.int64),
            "qo": lb[oi, 1].astype(np.int64),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from source import read_trace, write_trace
    trace = read_trace(args.input)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parameters = config.get("parameters", config)
    result = simulate(trace["times"], trace["lengths"], trace["ids"], parameters)
    write_trace(args.output, trace, times=result["completion"], mask=result["drop"] == 0)
    print(json.dumps({"generated": len(trace["ids"]), "delivered": int(np.sum(result["drop"] == 0)),
                      "input_drops": int(np.sum(result["drop"] == 1)),
                      "output_drops": int(np.sum(result["drop"] == 2))}))


if __name__ == "__main__":
    main()
