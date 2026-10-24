"""Nonlinear adapters: outcome classes, recovery of exact synthetic fits, M5 done-check."""

from __future__ import annotations

import math

import numpy as np
import pytest
import synth

from strdcheck.engines.nonlinear import ADAPTERS, classify
from strdcheck.engines.registry import adapters, get_adapter
from strdcheck.manifest import list_datasets
from strdcheck.parse import load, parse_text
from strdcheck.results import score


def test_classify() -> None:
    assert classify(True, 5.0) == "ok"
    assert classify(True, 3.9) == "silent"
    assert classify(True, math.nan) == "silent"
    assert classify(False, 7.0) == "flagged_ok"
    assert classify(False, 0.0) == "flagged"
    assert classify(None, 9.0) == "error"
    assert classify(True, 9.0, error=True) == "error"
    assert classify(True, 5.0, tau=6) == "silent"
    assert classify(True, 2.5, tau=2) == "ok"


def test_fourteen_configurations() -> None:
    ids = [a.id for a in adapters("nonlinear")]
    assert len(ids) == 14 == len(set(ids)) == len(ADAPTERS)
    assert "scipy.least_squares.lm.jac-2point.tol-tight" in ids


@pytest.fixture(scope="module")
def exact_misra():
    """Noise-free data from y = b1*(1-exp(-b2*x)) with b = (250, 5e-4)."""
    x = np.array([50.0, 100, 200, 300, 400, 500, 600, 700])
    y = 250.0 * (1 - np.exp(-5e-4 * x))
    params = [
        ("500", "100", "2.5000000000E+02", "0.0000000000E+00"),
        ("0.0001", "0.001", "5.0000000000E-04", "0.0000000000E+00"),
    ]
    text, info = synth.nonlinear(
        [repr(float(v)) for v in y],
        [repr(float(v)) for v in x],
        params,
        "0.0000000000E+00",
        "0.0000000000E+00",
    )
    return parse_text(text, info)


@pytest.mark.parametrize("adapter", [pytest.param(a, id=a.id) for a in ADAPTERS])
def test_recovers_exact_parameters(adapter, exact_misra) -> None:
    res = adapter.run(exact_misra, start=(240.0, 4.5e-4))
    assert res.status["success"] is True
    rows = {q.name: q for q in score(exact_misra.certified, res)}
    assert min(rows["b1"].lre, rows["b2"].lre) >= 7
    assert res.values["rss"] < 1e-10
    assert set(rows) >= {"b1", "b2", "se_b1", "se_b2", "rss", "resid_sd"}


def test_default_start_is_nist_start1(exact_misra) -> None:
    a = get_adapter("nonlinear", "scipy.least_squares.trf.jac-analytic.tol-tight")
    r1 = a.run(exact_misra)
    r2 = a.run(exact_misra, start=exact_misra.starts[0])
    assert r1.values == r2.values


def test_curve_fit_runtime_error_is_reported_failure(exact_misra, monkeypatch) -> None:
    """``RuntimeError`` from curve_fit ("optimal parameters not found") is a reported
    failure with nan values, not an adapter error."""
    import scipy.optimize

    def boom(*args, **kwargs):
        raise RuntimeError("Optimal parameters not found: maxfev reached")

    monkeypatch.setattr(scipy.optimize, "curve_fit", boom)
    res = get_adapter("nonlinear", "scipy.curve_fit.default").run(exact_misra)
    assert res.status["success"] is False
    assert all(math.isnan(v) for v in res.values.values())
    rows = score(exact_misra.certified, res)
    assert classify(res.status["success"], min(q.lre for q in rows if q.name == "b1")) == "flagged"


def test_m5_done_check(strd_dir) -> None:
    """trf + analytic Jacobian + tight tolerances from Start 2: >= 6 digits on every
    lower-difficulty problem."""
    a = get_adapter("nonlinear", "scipy.least_squares.trf.jac-analytic.tol-tight")
    for info in list_datasets("nonlinear", "lower"):
        ds = load(info.name, strd_dir)
        res = a.run(ds, start=ds.starts[1])
        b = [q.lre for q in score(ds.certified, res) if q.name in ds.param_names()]
        assert min(b) >= 6, (info.name, b)


def test_boxbod_start1_silent_lm(strd_dir) -> None:
    """Regression guard for a cell the probe found: lm reports success, 0 digits."""
    ds = load("BoxBOD", strd_dir)
    res = get_adapter("nonlinear", "scipy.least_squares.lm.jac-analytic.tol-default").run(
        ds, start=ds.starts[0]
    )
    b = min(q.lre for q in score(ds.certified, res) if q.name in ds.param_names())
    assert classify(res.status["success"], b) in {"silent", "ok"}


def test_derived_names_and_rank(exact_misra) -> None:
    """least_squares SEs come from strdcheck's SVD of res.jac: marked derived, rank recorded."""
    ls = get_adapter("nonlinear", "scipy.least_squares.trf.jac-analytic.tol-default")
    res = ls.run(exact_misra, start=(240.0, 4.5e-4))
    assert res.derived == {"se_b1", "se_b2", "resid_sd"}
    assert res.status["rank"] == 2 and res.status["p"] == 2
    cf = get_adapter("nonlinear", "scipy.curve_fit.default").run(exact_misra, start=(240.0, 4.5e-4))
    assert cf.derived == {"resid_sd"}  # curve_fit SEs are its own pcov
    rows = {q.name: q for q in score(exact_misra.certified, res)}
    assert rows["se_b1"].derived and not rows["b1"].derived
