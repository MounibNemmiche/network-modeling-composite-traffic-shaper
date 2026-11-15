"""Check metric populations, draining and independent-run aggregation."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import aggregate, feasible, measure


def score(times, lengths, completion, drop, duration):
    trace = dict(times=np.array(times, dtype=float), lengths=np.array(lengths, dtype=np.int64))
    result = dict(completion=np.array(completion, dtype=float), drop=np.array(drop, dtype=np.int8),
                  max_qi=1000, max_qo=1000)
    return measure(trace, result, duration)


def test_hand_example_populations_and_drain():
    m = score([0, 0, 0, 0], [1000]*4, [1, 2, 3, np.nan], [0, 0, 0, 1], 1)
    assert m['packets'] == 4
    assert m['delivered'] == 3
    assert m['input_drops'] == 1 and m['output_drops'] == 0
    assert m['loss'] == .25
    assert m['bad_1'] == .75  # A completion exactly at the deadline is on time.
    assert m['bad_5'] == .25  # Dropped packets still count as failures.
    assert m['mean_delay'] == 2
    assert m['drain_s'] == 2
    assert m['throughput_Bps'] == 1000  # 3000 delivered bytes / 3 seconds.


def test_exact_loss_target_is_accepted():
    m = score(np.zeros(1000), np.full(1000, 100), np.r_[np.ones(999), np.nan],
              np.r_[np.zeros(999, dtype=int), 1], 1)
    assert m['loss'] == 1/1000
    assert feasible(m)


def test_all_dropped_and_byte_fraction():
    m = score([0, 0], [100, 300], [np.nan, np.nan], [1, 2], 2)
    assert m['loss'] == 1 and m['byte_loss'] == 1
    assert m['mean_delay'] is None and m['p95_delay'] is None
    assert m['bad_1'] == m['bad_5'] == m['bad_10'] == 1
    assert m['throughput_Bps'] == 0 and m['drain_s'] == 0


def test_empty_trace_has_unavailable_fractions():
    m = score([], [], [], [], 2)
    assert m['packets'] == m['delivered'] == m['bytes_out'] == 0
    assert m['loss'] is None and m['bad_5'] is None and m['mean_delay'] is None
    assert m['throughput_Bps'] == 0
    assert not feasible(m)


def test_aggregate_means_runs_not_packets():
    first = score(np.zeros(10), np.full(10, 100), np.r_[np.ones(9), np.nan],
                  np.r_[np.zeros(9, dtype=int), 1], 2)
    second = score(np.zeros(1000), np.full(1000, 100), np.r_[np.ones(990), np.full(10, np.nan)],
                   np.r_[np.zeros(990, dtype=int), np.ones(10, dtype=int)], 2)
    summary = aggregate([first, second])
    assert summary['loss']['mean'] == pytest.approx(.055)
    assert summary['loss']['mean'] != pytest.approx(11/1010)
    assert summary['packets'] == 1010 and summary['runs'] == 2
    assert summary['loss']['sd'] == pytest.approx(.09/np.sqrt(2))


def test_zero_observed_is_not_interval_certificate():
    m = score([0], [100], [1], [0], 2)
    summary = aggregate([m]*20)
    assert summary['observed_feasible']
    assert summary['loss']['mean'] == 0
    assert not summary['loss']['interval_informative']
    assert not summary['interval_supported']
