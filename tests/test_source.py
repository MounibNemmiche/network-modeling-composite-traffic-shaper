"""Source and trace-format checks, including the preserved HW1 realization."""
import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import source


@pytest.fixture(scope="module")
def model():
    return source.fit()


def test_fitting_reference(model):
    assert model.fitting_packets == 32768
    assert model.fitting_bytes == 47399900
    assert len(model.on) == 9
    assert model.mean_size == 1446.5301513671875
    assert model.sizes.max() == 1452
    assert model.on.mean() == pytest.approx(0.40098644444443543, abs=1e-14)
    assert model.off.mean() == pytest.approx(22.073692233333343, abs=1e-14)


def test_saved_hw1_realization(model):
    trace = source.generate(model, 90, 20281020)
    assert len(trace["times"]) == 15351
    digest = hashlib.sha256(trace["times"].tobytes() + trace["lengths"].tobytes()).hexdigest()
    assert digest == "2851d121426b9391458614f543945955e48a4480ef93def3c6086aff79aeece0"
    assert np.all(np.diff(trace["times"]) > 0)
    assert np.all((trace["times"] >= 0) & (trace["times"] < 90))
    assert np.array_equal(trace["ids"], np.arange(1, 15352))
    assert sum(c["bytes"] for c in trace["cycles"]) == trace["lengths"].sum()


def test_against_original_hw1_when_available(model):
    hw1 = ROOT.parents[1] / "network-modeling-youtube-source-model" / "scripts" / "analyze_trace.py"
    if not hw1.exists():
        pytest.skip("Optional integration check requires the original HW1 workspace.")
    spec = importlib.util.spec_from_file_location("original_hw1_models", hw1)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    with source.DEFAULT_FITTING.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([float(r["capture_time_s"]) for r in rows]) - 60
    lengths = np.array([int(r["ip_length_bytes"]) for r in rows])
    old = original.Models(times, lengths, 210, 0.1)
    for seed in [20281020, 2026111400, 17]:
        t, w = old.onoff(np.random.default_rng(seed), 120)
        new = source.generate(model, 120, seed)
        assert np.array_equal(new["times"], t)
        assert np.array_equal(new["lengths"], w)


def test_complete_cycle_metadata(model):
    trace = source.generate(model, 200, 17)
    assert any(c["complete_on"] for c in trace["cycles"])
    for cycle in trace["cycles"]:
        assert cycle["start"] <= cycle["on_end"] < cycle["end"]
        assert cycle["complete_on"] == (cycle["start"] >= 0 and cycle["on_end"] <= 200)
        lo, hi = cycle["packet_start"], cycle["packet_stop"]
        assert cycle["bytes"] == trace["lengths"][lo:hi].sum()
        if hi > lo:
            assert np.all(trace["times"][lo:hi] >= max(0, cycle["start"]))
            assert np.all(trace["times"][lo:hi] < cycle["on_end"])


def test_trace_round_trip(tmp_path, model):
    trace = source.generate(model, 90, 20281020)
    path = tmp_path / "trace.txt"
    source.write_trace(path, trace)
    restored = source.read_trace(path)
    for key in ["ids", "times", "lengths"]:
        assert np.array_equal(restored[key], trace[key])


def test_equal_times_order_and_selection(tmp_path):
    path = tmp_path / "trace.txt"
    path.write_text("8 0 10\n2 0 20\n9 1 30\n", encoding="utf-8")
    trace = source.read_trace(path)
    assert trace["ids"].tolist() == [8, 2, 9]
    source.write_trace(path, trace, times=np.array([1., 2., 3.]), mask=np.array([True, False, True]))
    assert source.read_trace(path)["ids"].tolist() == [8, 9]


@pytest.mark.parametrize("text", [
    "1 0 10\n1 1 10\n", "0 0 10\n", "1 -1 10\n", "1 nan 10\n",
    "1 inf 10\n", "1 1 10\n2 0 10\n", "1 0 0\n", "1 0 1.5\n",
    "1 0 10 extra\n", "id time length\n", "1.5 0 10\n",
])
def test_malformed_trace_rejected(tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        source.read_trace(path)


def test_empty_trace(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    trace = source.read_trace(path)
    assert trace["times"].dtype == np.float64
    assert trace["ids"].dtype == np.int64
    assert not len(trace["times"])


def test_poisson_and_bad_arguments(model):
    trace = source.generate(model, 10, 18, "poisson", 200000)
    assert np.all(np.diff(trace["times"]) >= 0)
    assert trace["cycles"] == []
    for duration in [-1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            source.generate(model, duration, 1)
    with pytest.raises(ValueError):
        source.generate(model, 1, 1, "poisson", -1)
    with pytest.raises(ValueError):
        source.generate(model, 1, 1, "onoff", 200000)
