"""Generate the HW1 empirical ON-OFF source or a mean-matched Poisson control.

Only packet descriptors are generated. Times are logical seconds and lengths are
original IP bytes. The ON-OFF RNG sequence deliberately matches the HW1 generator.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_FITTING = ROOT / "data" / "fitting" / "packets.csv"


@dataclass
class SourceModel:
    on: np.ndarray
    off: np.ndarray
    active_gaps: np.ndarray
    sizes: np.ndarray
    size_probabilities: np.ndarray
    active_counts: np.ndarray
    fitting_packets: int
    fitting_bytes: int
    rate: float
    threshold: float = 0.1

    @property
    def mean_size(self):
        return float(np.dot(self.sizes, self.size_probabilities))

    @property
    def cycle_lengths(self):
        return self.on[:, None] + self.off[None, :]

    @property
    def initial_cycle_probabilities(self):
        lengths = self.cycle_lengths
        return lengths.ravel() / lengths.sum()


def fit(path=DEFAULT_FITTING):
    """Fit only capture time [60,270), preserving HW1's boundary rules."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([float(row["capture_time_s"]) for row in rows])
    lengths = np.array([int(row["ip_length_bytes"]) for row in rows], dtype=np.int64)
    mask = (times >= 60) & (times < 270)
    times, lengths = times[mask] - 60, lengths[mask]
    if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("Fitting times must be finite and strictly increasing.")
    if np.any(lengths <= 0):
        raise ValueError("Fitting packet lengths must be positive.")
    threshold, duration = 0.1, 210.0
    split = np.flatnonzero(np.diff(times) > threshold) + 1
    starts, ends = np.r_[0, split], np.r_[split - 1, len(times) - 1]
    complete = (times[starts] >= threshold) & (times[ends] <= duration - threshold)
    indices = [j for j in range(len(starts) - 1) if complete[j]]
    if len(indices) < 3:
        raise ValueError("Too few complete fitting activity cycles.")
    on = np.array([times[ends[j]] - times[starts[j]] for j in indices])
    off = np.array([times[starts[j + 1]] - times[ends[j]] for j in indices])
    gaps = np.concatenate([
        np.rint(np.diff(times[starts[j]:ends[j] + 1]) * 1e9) / 1e9 for j in indices
    ])
    if np.any(on <= 0) or np.any(off <= 0) or not len(gaps) or np.any(gaps <= 0):
        raise ValueError("Source requires positive ON/OFF durations and active gaps.")
    sizes, counts = np.unique(lengths, return_counts=True)
    active_counts = np.array([ends[j] - starts[j] + 1 for j in indices], dtype=np.int64)
    return SourceModel(on, off, gaps, sizes, counts / counts.sum(), active_counts,
                       len(times), int(lengths.sum()), len(times) / duration, threshold)


def generate(model, duration, seed, kind="onoff", matched_rate=None):
    """Return times, lengths, sequential IDs and cycle metadata without payloads.

    ``matched_rate`` is the Poisson control's expected byte rate. It does not
    rescale ON-OFF timestamps. Cycle metadata consumes no additional RNG draws.
    """
    duration = float(duration)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("Duration must be finite and nonnegative.")
    if kind not in {"onoff", "poisson"}:
        raise ValueError("Model must be onoff or poisson.")
    if kind == "onoff" and matched_rate is not None:
        raise ValueError("A matched rate applies only to the Poisson control.")
    rng = np.random.default_rng(seed)
    cycles = []
    if kind == "poisson":
        rate = model.rate if matched_rate is None else float(matched_rate) / model.mean_size
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Poisson rate must be finite and positive.")
        count = rng.poisson(rate * duration)
        times = np.sort(rng.uniform(0, duration, count))
    else:
        cycle_lengths = model.cycle_lengths
        pair = rng.choice(cycle_lengths.size, p=model.initial_cycle_probabilities)
        i, j = np.unravel_index(pair, cycle_lengths.shape)
        on, off = model.on[i], model.off[j]
        start = -rng.uniform(0, on + off)
        parts, packet_offset = [], 0
        while start < duration:
            accepted = np.empty(0, dtype=np.float64)
            if start + on >= 0:
                positions = [np.array([0.0])]
                current = 0.0
                block_size = max(1024, round(on / model.active_gaps.mean() * 1.3))
                while current < on:
                    candidate = current + np.cumsum(rng.choice(model.active_gaps, size=block_size))
                    positions.append(candidate[candidate < on])
                    current = candidate[-1]
                local = np.concatenate(positions) + start
                accepted = local[(local >= 0) & (local < duration)]
                parts.append(accepted)
            cycles.append({"start": float(start), "on_end": float(start + on),
                           "end": float(start + on + off),
                           "complete_on": bool(start >= 0 and start + on <= duration),
                           "packet_start": packet_offset,
                           "packet_stop": packet_offset + len(accepted)})
            packet_offset += len(accepted)
            start += on + off
            on, off = rng.choice(model.on), rng.choice(model.off)
        times = np.concatenate(parts) if parts else np.empty(0)
    lengths = rng.choice(model.sizes, size=len(times), p=model.size_probabilities)
    for cycle in cycles:
        cycle["bytes"] = int(lengths[cycle["packet_start"]:cycle["packet_stop"]].sum())
    return {"times": np.asarray(times, dtype=np.float64),
            "lengths": np.asarray(lengths, dtype=np.int64),
            "ids": np.arange(1, len(times) + 1, dtype=np.int64), "cycles": cycles}


def read_trace(path):
    """Read a strict headerless ID/time/length trace; preserve tied input order."""
    ids, times, lengths = [], [], []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"Line {line_number}: expected exactly three columns.")
            try:
                packet_id, timestamp, length = int(fields[0]), float(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: invalid ID, timestamp or length.") from exc
            if packet_id <= 0 or packet_id in seen or packet_id > np.iinfo(np.int64).max:
                raise ValueError(f"Line {line_number}: ID must be a unique positive int64.")
            if not np.isfinite(timestamp) or timestamp < 0 or (times and timestamp < times[-1]):
                raise ValueError(f"Line {line_number}: times must be finite, nonnegative and ordered.")
            if length <= 0 or length > np.iinfo(np.int64).max:
                raise ValueError(f"Line {line_number}: length must be a positive int64.")
            seen.add(packet_id)
            ids.append(packet_id)
            times.append(timestamp)
            lengths.append(length)
    return {"ids": np.array(ids, dtype=np.int64), "times": np.array(times, dtype=np.float64),
            "lengths": np.array(lengths, dtype=np.int64), "cycles": []}


def write_trace(path, trace, times=None, mask=None):
    """Write full-precision descriptor rows, optionally selecting delivered packets."""
    actual_times = np.asarray(trace["times"] if times is None else times)
    ids, lengths = np.asarray(trace["ids"]), np.asarray(trace["lengths"])
    if not len(actual_times) == len(ids) == len(lengths):
        raise ValueError("Trace columns must have equal length.")
    if mask is not None:
        actual_times, ids, lengths = actual_times[mask], ids[mask], lengths[mask]
    if np.any(~np.isfinite(actual_times)) or np.any(actual_times < 0):
        raise ValueError("Written timestamps must be finite and nonnegative.")
    if np.any(np.diff(actual_times) < 0):
        raise ValueError("Written timestamps must be nondecreasing.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for packet_id, timestamp, length in zip(ids, actual_times, lengths):
            handle.write(f"{int(packet_id)} {float(timestamp):.17g} {int(length)}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", choices=["onoff", "poisson"], default="onoff")
    parser.add_argument("--rate", type=float, help="Expected Poisson-control byte rate")
    parser.add_argument("--fitting", type=Path, default=DEFAULT_FITTING)
    parser.add_argument("--output", type=Path, default=Path("Input_Trace_File.txt"))
    args = parser.parse_args()
    try:
        trace = generate(fit(args.fitting), args.duration, args.seed, args.model, args.rate)
        write_trace(args.output, trace)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(trace['times'])} packets to {args.output}")


if __name__ == "__main__":
    main()
