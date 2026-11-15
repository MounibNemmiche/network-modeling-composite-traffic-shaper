"""Numerical regression checks for the physical meaning of plotted evidence."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.figure
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_figures


def test_rate_excludes_final_endpoint_and_conserves_bytes():
    # The last bin is shorter. Integrating the rates must still recover exactly
    # the eligible packet bytes, with the next interval's first packet excluded.
    bins = np.array([2., 2.1, 2.2, 2.25])
    times = np.array([1.9, 2., 2.1, 2.24, 2.25, 2.3])
    lengths = np.array([500, 100, 200, 300, 400, 600])
    rates = make_figures.interval_rate(times, lengths, bins)
    recovered = np.sum(rates * 1e6 / 8 * np.diff(bins))
    assert recovered == pytest.approx(600.)
    np.testing.assert_allclose(rates, [0.008, 0.016, 0.048])


def test_token_path_has_refill_slope_plateau_and_vertical_deductions():
    # A ten-token bucket spends eight at t=2 and five at t=15. It must
    # refill at one token/s, saturate at t=10, and jump down at t=15.
    x, y = make_figures.token_curve([0, 2, 15], [10, 2, 5],
                                   rho=1, capacity=10, begin=0, end=22)
    assert x[0] == 0 and x[-1] == 22
    assert np.all(np.diff(x) >= 0)
    assert np.all((y >= 0) & (y <= 10))
    segments = [(a, b, c, d) for a, b, c, d in zip(x[:-1], y[:-1], x[1:], y[1:])]
    assert any(a == 2 and b == 10 and c == 2 and d == 2 for a, b, c, d in segments)
    assert any(a == 2 and b == 2 and c == 10 and d == 10 for a, b, c, d in segments)
    assert any(a == 10 and b == 10 and c == 15 and d == 10 for a, b, c, d in segments)
    assert any(a == 15 and b == 10 and c == 15 and d == 5 for a, b, c, d in segments)
    # A positive-duration segment cannot contain token consumption smeared over
    # time; its slope is either the physical refill rate or a full-bucket plateau.
    for a, b, c, d in segments:
        if c > a:
            assert (d - b) / (c - a) == pytest.approx(0 if b == d == 10 else 1)


def test_token_path_clips_inside_refill_and_excludes_right_endpoint_action():
    x, y = make_figures.token_curve([0, 2, 15, 22], [10, 2, 5, 1],
                                   rho=1, capacity=10, begin=3, end=22)
    # The prior post-event balance is two at t=2, hence three at t=3.
    assert x[0] == 3 and y[0] == 3
    # The five-token deduction at t=15 has refilled by t=20. A new deduction
    # exactly at t=22 belongs to the following interval and must not be shown.
    assert x[-1] == 22 and y[-1] == 10
    assert np.all(y[x == 22] == 10)
    assert any(a == 20 and b == 10 for a, b in zip(x, y))


def test_token_path_shows_action_at_left_boundary():
    x, y = make_figures.token_curve([0, 2, 15], [10, 2, 5],
                                   rho=1, capacity=10, begin=2, end=3)
    left_values = y[x == 2]
    assert left_values[0] == 10 and left_values[-1] == 2
    assert y[-1] == 3


def test_queue_covers_quiet_tail_and_preserves_boundary_convention():
    x, y = make_figures.queue_curve([0, 2, 15, 22], [0, 3000, 0, 4000], 3, 22)
    np.testing.assert_array_equal(x, [3, 15, 22])
    np.testing.assert_array_equal(y, [3000, 0, 0])
    # A queue change at the left endpoint is included immediately.
    x, y = make_figures.queue_curve([0, 2, 15], [0, 3000, 0], 2, 10)
    np.testing.assert_array_equal(x, [2, 10])
    np.testing.assert_array_equal(y, [3000, 3000])


def test_report_output_rate_uses_completions_not_generation_window(tmp_path, monkeypatch):
    """Run the figure pipeline and inspect its exported rate population."""
    results = tmp_path / "data" / "results"
    results.mkdir(parents=True)
    p = dict(rho=10, b=100, mu=1000, Bi=50, Bo=50)
    summary = dict(parameters=p, calibration=dict(r=100, M=1000), selected="C1")
    (results / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = [dict(id=f"C{i}", stage="coarse", parameters=dict(p, Bi=bi, Bo=100-bi),
                 metrics=dict(loss=0, bad_5=0)) for i, bi in enumerate([25, 50, 75])]
    (results / "screening.json").write_text(json.dumps(rows), encoding="utf-8")
    # Packet 1 was generated before the plotted interval but completes inside it.
    # Packet 3 is generated inside but completes outside. Packet 4 is dropped;
    # packet 5 has a completion exactly at the excluded right endpoint.
    np.savez(results / "diagnostic.npz",
             interval=np.array([2., 3.]),
             times=np.array([.5, 2.2, 2.8, 2.9, 3.]),
             lengths=np.array([1000, 2000, 3000, 5000, 4000]),
             completion=np.array([2.1, 2.6, 3.2, np.nan, 3.]),
             drop=np.array([0, 0, 0, 1, 0]),
             event_time=np.array([0., 2., 3.2]),
             event_tokens=np.array([100., 60., 40.]),
             event_qi=np.array([0, 0, 0]), event_qo=np.array([0, 3000, 0]))
    monkeypatch.setattr(make_figures, "ROOT", tmp_path)
    monkeypatch.setattr(make_figures, "RESULTS", results)
    # Pixel rendering is unrelated to this population regression. Keep the
    # actual histogram/export path and suppress only the expensive file render.
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", lambda *args, **kwargs: None)
    make_figures.main()
    with (tmp_path / "figure" / "rate-curves.csv").open(newline="", encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    output_rates = np.array([float(row["output_Mbps"]) for row in curves])
    # The expected eligible output population is packets 1 and 2: 3000 bytes.
    assert np.sum(output_rates) * .1 * 1e6 / 8 == pytest.approx(3000)
