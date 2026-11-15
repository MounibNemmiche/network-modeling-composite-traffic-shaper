"""Independent small-trace checks for packet admission and service semantics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shaper import simulate


def run(times, lengths=None, **overrides):
    if lengths is None:
        lengths = [1000] * len(times)
    parameters = dict(rho=1000, b=2000, Bi=1000, mu=1000, Bo=1000)
    parameters.update(overrides)
    return simulate(np.array(times, float), np.array(lengths, np.int64),
                    np.arange(1, len(times) + 1, dtype=np.int64), parameters,
                    record_events=True)


def test_hand_worked_four_packets():
    result = run([0, 0, 0, 0])
    np.testing.assert_array_equal(result["drop"], [0, 0, 0, 1])
    np.testing.assert_allclose(result["completion"][:3], [1, 2, 3], rtol=0, atol=0)
    np.testing.assert_allclose(result["release"][:3], [0, 0, 1], rtol=0, atol=0)
    assert result["max_qi"] == result["max_qo"] == 1000


def test_larger_bucket_moves_overflow_to_output():
    result = run([0, 0, 0, 0], b=3000)
    np.testing.assert_array_equal(result["drop"], [0, 0, 2, 0])
    np.testing.assert_allclose(result["completion"][[0, 1, 3]], [1, 2, 3])


def test_isolated_lbf_against_recurrence():
    times = np.array([0, 0, .1, 1.2, 10, 10, 10.01], float)
    lengths = np.array([100, 230, 50, 72, 300, 200, 100], np.int64)
    result = run(times, lengths, rho=10000, b=100000, mu=700, Bo=100000)
    expected = []
    finish = 0
    for arrival, length in zip(times, lengths):
        finish = max(arrival, finish) + length / 700
        expected.append(finish)
    np.testing.assert_allclose(result["completion"], expected, rtol=0, atol=0)


def test_fifo_head_blocks_smaller_later_packet():
    result = run([0, 0, .1], [1000, 1000, 100], rho=1000, b=1000,
                 Bi=2000, mu=100000, Bo=2000)
    np.testing.assert_allclose(result["release"], [0, 1, 1.1])


def test_completion_precedes_tied_arrival_with_zero_waiting_capacity():
    result = run([0, 1], b=2000, Bi=0, Bo=0)
    np.testing.assert_array_equal(result["drop"], [0, 0])
    np.testing.assert_allclose(result["completion"], [1, 2])


def test_zero_capacity_allows_immediate_service_only():
    result = run([0, 0, 0], b=1000, Bi=0, Bo=0)
    np.testing.assert_array_equal(result["drop"], [0, 1, 1])
    result = run([0, 0], b=2000, Bi=0, Bo=0)
    np.testing.assert_array_equal(result["drop"], [0, 2])


def test_exact_input_fit_and_one_byte_over():
    result = run([0, 0, 0], [1000, 1000, 1], b=1000, Bi=1000)
    np.testing.assert_array_equal(result["drop"], [0, 0, 1])


def test_exact_output_fit_and_one_byte_over():
    result = run([0, 0, 0], [1000, 1000, 1], b=3000, Bo=1000)
    np.testing.assert_array_equal(result["drop"], [0, 0, 2])


def test_output_drop_does_not_refund_tokens():
    result = run([0, 0, .1], b=2000, Bo=0, Bi=2000, mu=10000)
    assert result["drop"][1] == 2
    assert result["release"][2] == pytest.approx(1)


def test_idle_refill_saturates_and_tail_drains():
    result = run([0, 0, 100, 100, 100], rho=1, b=2000, Bi=4000,
                 Bo=4000, mu=1000)
    np.testing.assert_allclose(result["release"], [0, 0, 1000, 2000, 3000])
    assert result["completion"][-1] == 3001
    assert np.all(result["drop"] == 0)
    events = result["events"]
    assert events["qi"][-1] == events["qo"][-1] == 0
    assert np.all((events["tokens"] >= 0) & (events["tokens"] <= 2000))


def test_empty_trace():
    result = run([])
    assert len(result["completion"]) == 0
    assert result["max_qi"] == result["max_qo"] == 0
    assert result["events"]["time"].tolist() == [0]


def test_decimal_refill_boundary_does_not_create_false_drops():
    times = np.arange(100, dtype=float) / 10
    result = run(times, np.ones(100, dtype=np.int64), rho=10, b=1,
                 Bi=0, mu=1000, Bo=0)
    assert np.all(result["drop"] == 0)
    np.testing.assert_array_equal(result["release"], times)


def test_long_idle_refills_to_capacity():
    result = run([0, 100, 100], rho=100, b=2000, Bi=0, Bo=2000)
    np.testing.assert_array_equal(result["drop"], [0, 0, 0])
    np.testing.assert_array_equal(result["release"], [0, 100, 100])


def test_existing_input_waiter_precedes_tied_arrival():
    result = run([0, 0, 1], b=1000, Bi=1000, Bo=0, mu=1000)
    np.testing.assert_array_equal(result["drop"], [0, 0, 0])
    np.testing.assert_array_equal(result["release"], [0, 1, 2])


@pytest.mark.parametrize("override", [dict(rho=0), dict(mu=-1), dict(b=999),
                         dict(Bi=-1), dict(Bo=1.5), dict(rho=float("nan")),
                         dict(mu=float("inf"))])
def test_invalid_parameters(override):
    with pytest.raises(ValueError):
        run([0], **override)


def test_invalid_arrays():
    parameters = dict(rho=10, b=1000, Bi=1000, mu=1000, Bo=1000)
    for times, lengths, ids in [([1, 0], [1, 1], [1, 2]),
                                ([0, 1], [1, 1], [1, 1]),
                                ([0], [0], [1]),
                                ([float("nan")], [1], [1])]:
        with pytest.raises(ValueError):
            simulate(times, np.asarray(lengths, dtype=np.int64),
                     np.asarray(ids, dtype=np.int64), parameters)


def test_random_small_trace_conservation_and_credit():
    rng = np.random.default_rng(317)
    for _ in range(10):
        times = np.cumsum(rng.exponential(.01, 100))
        lengths = rng.integers(1, 1001, 100)
        result = run(times, lengths, rho=1000, b=1500, Bi=10000,
                     mu=2000, Bo=3000)
        delivered = result["drop"] == 0
        assert len(result["drop"]) == np.count_nonzero(delivered) + np.count_nonzero(result["drop"])
        assert np.all(result["completion"][delivered] >= times[delivered])
        assert np.allclose(result["completion"][delivered] - result["start"][delivered], lengths[delivered]/2000)
        released = np.flatnonzero(np.isfinite(result["release"]))
        # Independent credit replay, including packets dropped downstream.
        credit, previous = 1500., 0.
        for j in released:
            time = result["release"][j]
            credit = min(1500., credit + 1000*(time-previous)) - lengths[j]
            assert credit >= -1e-7
            previous = time
        assert result["max_qi"] <= 10000 and result["max_qo"] <= 3000


def test_two_pass_matches_exact_global_event_oracle():
    """Compare against a separate global clock using exact rational arithmetic."""
    from collections import deque
    from fractions import Fraction as F

    def oracle(ticks, sizes, p):
        n = len(ticks)
        t, tokens = F(0), F(p["b"])
        qi, qo = deque(), deque()
        bi = bo = i = 0
        release, start, completion = ([None] * n for _ in range(3))
        drop = [0] * n
        active = due = None

        def send(j):
            nonlocal active, due, bo
            if active is None:
                active, start[j] = j, t
                due = t + F(int(sizes[j]), p["mu"])
            elif bo + int(sizes[j]) <= p["Bo"]:
                qo.append(j)
                bo += int(sizes[j])
            else:
                drop[j] = 2

        while i < n or qi or active is not None:
            arrival = F(int(ticks[i]), 100) if i < n else None
            ready = t + max(F(0), F(int(sizes[qi[0]])) - tokens) / p["rho"] if qi else None
            new_time = min(v for v in (arrival, ready, due) if v is not None)
            tokens = min(F(p["b"]), tokens + p["rho"] * (new_time - t))
            t = new_time
            if due == t:
                completion[active] = t
                active = due = None
                if qo:
                    j = qo.popleft()
                    bo -= int(sizes[j])
                    send(j)
            while qi and tokens >= int(sizes[qi[0]]):
                j = qi.popleft()
                bi -= int(sizes[j])
                tokens -= int(sizes[j])
                release[j] = t
                send(j)
            while i < n and F(int(ticks[i]), 100) == t:
                if not qi and tokens >= int(sizes[i]):
                    tokens -= int(sizes[i])
                    release[i] = t
                    send(i)
                elif bi + int(sizes[i]) <= p["Bi"]:
                    qi.append(i)
                    bi += int(sizes[i])
                else:
                    drop[i] = 1
                i += 1
        return release, start, completion, drop

    rng = np.random.default_rng(942)
    for _ in range(30):
        ticks = np.cumsum(rng.integers(0, 20, 50))
        sizes = rng.integers(1, 100, 50, dtype=np.int64)
        p = dict(rho=int(rng.integers(10, 1000)), b=int(rng.integers(100, 500)),
                 Bi=int(rng.integers(0, 500)), mu=int(rng.integers(10, 1000)),
                 Bo=int(rng.integers(0, 500)))
        actual = run(ticks / 100, sizes, **p)
        expected = oracle(ticks, sizes, p)
        np.testing.assert_array_equal(actual["drop"], expected[3])
        for key, values in zip(["release", "start", "completion"], expected[:3]):
            wanted = np.array([float(v) if v is not None else np.nan for v in values])
            np.testing.assert_allclose(actual[key], wanted, atol=1e-10, rtol=1e-13, equal_nan=True)
