"""V4 adapter correctness for univariate and ANOVA adapters on exact synthetic data."""

from __future__ import annotations

import math

import pytest
import synth

from strdcheck import exact
from strdcheck.engines import anova as anova_mod
from strdcheck.engines.base import decimal_type
from strdcheck.engines.registry import adapters, get_adapter
from strdcheck.parse import load, parse_text
from strdcheck.results import score


@pytest.fixture(scope="module")
def univ_ds():
    text, info = synth.univariate(["1", "2", "3", "4", "5"], "3", "1.58113883008419", "0.4")
    return parse_text(text, info)


@pytest.fixture(scope="module")
def anova_ds():
    groups = [1, 1, 1, 2, 2, 2, 3, 3]
    values = ["1", "2", "4", "4", "5", "9", "7", "8"]
    ref = exact.anova([str(g) for g in groups], exact.to_fractions(values))
    cert = {k: f"{float(v):.14E}" for k, v in ref.items()}
    text, info = synth.anova(groups, values, cert, (2, 5))
    return parse_text(text, info), {k: float(v) for k, v in ref.items()}


def _avail(suite):
    return [pytest.param(a, id=a.id) for a in adapters(suite) if a.available()]


@pytest.mark.parametrize("adapter", _avail("univariate"))
def test_univariate_exact_answer(adapter, univ_ds) -> None:
    res = adapter.run(univ_ds)
    want = {"mean": 3.0, "sd": math.sqrt(2.5), "r1": 0.4}
    assert {"mean", "sd"} <= set(res.values)
    rel = 1e-6 if adapter.dtype == "f32" else 1e-13
    for k, v in res.values.items():
        assert v == pytest.approx(want[k], rel=rel), k
    assert res.seconds >= 0


@pytest.mark.parametrize("adapter", _avail("anova"))
def test_anova_exact_answer(adapter, anova_ds) -> None:
    ds, want = anova_ds
    res = adapter.run(ds)
    assert res.values
    for k, v in res.values.items():
        assert v == pytest.approx(want[k], rel=1e-12), k


def test_pandas_autocorr_is_not_nist_r1(univ_ds) -> None:
    """Documents why pandas has no r1 adapter (DESIGN 3.2)."""
    pd = pytest.importorskip("pandas")
    y = univ_ds.array("y")
    assert pd.Series(y).autocorr(1) == pytest.approx(1.0)  # Pearson of the lagged series
    assert get_adapter("univariate", "numpy.f64").run(univ_ds).values["r1"] == pytest.approx(0.4)
    assert "r1" not in get_adapter("univariate", "pandas.f64").run(univ_ds).values


def test_combine_matches_exact(anova_ds) -> None:
    ds, want = anova_ds
    got = anova_mod.numpy_twopass(ds)
    for k in want:
        assert got[k] == pytest.approx(want[k], rel=1e-13)


def test_decimal_type() -> None:
    assert decimal_type(("1.5", "-22.25", "3")) == "DECIMAL(18, 2)"
    assert decimal_type(("1000000000000.4",)) == "DECIMAL(18, 1)"
    assert decimal_type(("1" * 20 + ".5",)) == "DECIMAL(38, 1)"
    with pytest.raises(ValueError):
        decimal_type(("1E5",))
    with pytest.raises(ValueError):
        decimal_type(("1" * 39,))


def test_registry_unique_and_lookup() -> None:
    ids = [(a.suite, a.id) for a in adapters()]
    assert len(ids) == len(set(ids))
    assert get_adapter("anova", "control.exact").role == "control"
    with pytest.raises(KeyError):
        get_adapter("anova", "nope")


def _lre(adapter_id, suite, ds, name):
    res = get_adapter(suite, adapter_id).run(ds)
    return {q.name: q.lre for q in score(ds.certified, res)}[name]


def test_controls_on_real_data(strd_dir) -> None:
    """M3 done-check: control.exact hits the cap on all 20 univariate/ANOVA datasets;
    the one-pass control loses at least 5 digits below the float64 ceiling on NumAcc3/4."""
    from strdcheck.manifest import list_datasets

    for suite in ("univariate", "anova"):
        for info in list_datasets(suite):
            ds = load(info.name, strd_dir)
            res = get_adapter(suite, "control.exact").run(ds)
            for q in score(ds.certified, res):
                assert q.lre >= q.cap - 1, (info.name, q.name, q.lre)
    for name in ("NumAcc3", "NumAcc4"):
        ds = load(name, strd_dir)
        sd = _lre("control.onepass.f64", "univariate", ds, "sd")
        ceiling = exact.ceilings(ds)["sd"].f64
        assert math.isnan(sd) or sd <= ceiling - 5, (name, sd, ceiling)
