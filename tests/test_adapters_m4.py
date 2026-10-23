"""V4 adapter correctness for linear adapters on synthetic data with exact answers."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import synth

from strdcheck import exact
from strdcheck.engines import linear as lin
from strdcheck.engines.registry import adapters, get_adapter
from strdcheck.parse import LINEAR_SPECS, load, parse_text
from strdcheck.results import score


def _make(name: str, x: list[str], y: list[str]):
    """Synthetic dataset in StRD layout whose certified values are exact (15 digits)."""
    spec = LINEAR_SPECS[name]
    xs = [Fraction(v) for v in x]
    cols = ([[Fraction(1)] * len(x)] if spec.intercept else []) + [
        [v**j for v in xs] for j in range(1, spec.degree + 1)
    ]
    ref = exact.linear(cols, [Fraction(v) for v in y], spec.param_names(), intercept=spec.intercept)
    fmt = {k: f"{float(v):.14E}" for k, v in ref.items()}
    params = {nm: (fmt[nm], fmt[f"se_{nm}"]) for nm in spec.param_names()}
    text, info = synth.linear(y, x, params, fmt["resid_sd"], fmt["r2"], name=name)
    return parse_text(text, info)


@pytest.fixture(scope="module")
def datasets():
    rng = np.random.default_rng(20261024)
    x = [str(v) for v in range(1, 13)]
    noise = rng.integers(-3, 4, size=12)
    y1 = [str(3 + 2 * i + int(e)) for i, e in zip(range(1, 13), noise, strict=True)]
    y2 = [str(5 - i + i * i + int(e)) for i, e in zip(range(1, 13), noise, strict=True)]
    y0 = [str(4 * i + int(e)) for i, e in zip(range(1, 13), noise, strict=True)]
    return {
        "Norris": _make("Norris", x, y1),
        "Pontius": _make("Pontius", x, y2),
        "NoInt1": _make("NoInt1", x, y0),
    }


def _avail():
    return [pytest.param(a, id=a.id) for a in adapters("linear") if a.available()]


@pytest.mark.parametrize("adapter", _avail())
def test_linear_adapter_on_synthetic(adapter, datasets) -> None:
    ran = 0
    for ds in datasets.values():
        if not adapter.applicable(ds):
            continue
        res = adapter.run(ds)
        rows = score(ds.certified, res)
        names = {q.name for q in rows}
        assert set(ds.param_names()) <= names
        need = 4 if adapter.dtype == "f32" else 10
        for q in rows:
            if q.name.startswith("b") or not q.derived:
                assert q.lre >= need, (ds.name, q.name, q.lre)
        if "rank" in res.status:
            assert res.status["rank"] == res.status["p"]
        ran += 1
    assert ran >= 1


def test_exact_fit_has_zero_residual() -> None:
    ds = _make("Norris", ["0", "1", "2", "3"], ["1", "3", "5", "7"])
    res = get_adapter("linear", "numpy.lstsq").run(ds)
    assert res.values["b0"] == pytest.approx(1.0) and res.values["b1"] == pytest.approx(2.0)
    assert abs(res.values["resid_sd"]) < 1e-12


def test_design_matrix_powers_in_dtype(datasets) -> None:
    X, y = lin.design(datasets["Pontius"], np.float32)
    assert X.dtype == np.float32 and X.shape == (12, 3)
    np.testing.assert_array_equal(X[:, 2], X[:, 1] ** 2)
    X0, _ = lin.design(datasets["NoInt1"])
    assert X0.shape == (12, 1)


def test_applicability(datasets) -> None:
    assert get_adapter("linear", "scipy.linregress").applicable(datasets["Norris"])
    assert not get_adapter("linear", "scipy.linregress").applicable(datasets["Pontius"])
    assert not get_adapter("linear", "numpy.polyfit").applicable(datasets["NoInt1"])


def test_m4_done_check_real_data(strd_dir) -> None:
    """Normal equations lose >= 5 digits on Filip versus Polynomial.fit; ranks are recorded."""
    ds = load("Filip", strd_dir)

    def bmin(aid):
        res = get_adapter("linear", aid).run(ds)
        return min(q.lre for q in score(ds.certified, res) if q.name in ds.param_names()), res

    ne, _ = bmin("control.normal-equations.f64")
    pf, _ = bmin("numpy.Polynomial.fit")
    assert pf - ne >= 5
    _, res = bmin("numpy.lstsq")
    assert res.status["p"] == 11 and 1 <= res.status["rank"] <= 11
