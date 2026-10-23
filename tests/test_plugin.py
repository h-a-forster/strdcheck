from __future__ import annotations

import pytest


def test_strd_assert_lre_passes_and_fails(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        def test_ok(strd_assert_lre):
            worst = strd_assert_lre([1.0000001, 2.0], [1.0, 2.0], min_lre=6)
            assert 6.9 < worst < 7.1

        def test_low(strd_assert_lre):
            with pytest.raises(AssertionError, match="LRE below 6"):
                strd_assert_lre([1.001], [1.0], min_lre=6, label="mean")

        def test_nan(strd_assert_lre):
            with pytest.raises(AssertionError):
                strd_assert_lre([float("nan")], [1.0], min_lre=1)

        @pytest.mark.strd("Norris")
        def test_marker(strd_data_dir):
            assert strd_data_dir.name
        """
    )
    result = pytester.runpytest("--strict-markers")
    result.assert_outcomes(passed=4)


def test_min_lre_override(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_override(strd_assert_lre):
            strd_assert_lre([1.001], [1.0], min_lre=6)
        """
    )
    result = pytester.runpytest("--strd-min-lre", "2")
    result.assert_outcomes(passed=1)


# --- M6: dataset fixture, params, report, summary, check_* helpers -------------------------

import math  # noqa: E402

import synth  # noqa: E402

import strdcheck.testing as st  # noqa: E402

NUMACC1 = synth.univariate(
    ["10000001", "10000003", "10000002"], "10000002", "1", "-0.5", exact=True
)[0]


def test_params_ids_and_marks() -> None:
    ps = st.params("univariate")
    assert [p.id for p in ps][:2] == ["PiDigits", "Lottery"]
    assert len(ps) == 9
    assert ps[0].marks[0].name == "strd" and ps[0].marks[0].args == ("PiDigits",)
    assert len(st.params("nonlinear", "higher")) == 8


def test_strd_dataset_skips_when_missing(pytester: pytest.Pytester, tmp_path) -> None:
    pytester.makepyfile(
        """
        import pytest
        from strdcheck.testing import params

        @pytest.mark.parametrize("strd_dataset", params("univariate")[:2], indirect=True)
        def test_ds(strd_dataset):
            assert strd_dataset.n > 0
        """
    )
    result = pytester.runpytest_inprocess("--strd-data-dir", str(tmp_path / "empty"))
    result.assert_outcomes(skipped=2)


def test_strd_dataset_fetch_report_and_summary(
    pytester: pytest.Pytester, tmp_path, monkeypatch
) -> None:
    data = tmp_path / "data"
    fetched = []

    def fake_fetch(info, d, **kw):
        d.mkdir(parents=True, exist_ok=True)
        (d / info.filename).write_text(NUMACC1, newline="")
        fetched.append(info.name)
        return d / info.filename

    monkeypatch.setattr(st, "fetch_one", fake_fetch)
    monkeypatch.setattr(st, "is_cached", lambda info, d: (d / info.filename).exists())
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.parametrize("strd_dataset", ["NumAcc1"], indirect=True)
        def test_mean(strd_dataset, strd_assert_lre):
            y = strd_dataset.array("y")
            c = strd_dataset.certified["mean"]
            strd_assert_lre([y.mean()], [c.value], min_lre=10, cap=c.cap, label="mean")

        def test_weak(strd_assert_lre):
            strd_assert_lre([1.01], [1.0], min_lre=1, label="weak")
        """
    )
    report = tmp_path / "out" / "lre.csv"
    result = pytester.runpytest_inprocess(
        "--strd-data-dir", str(data), "--strd-fetch", "--strd-report", str(report)
    )
    result.assert_outcomes(passed=2)
    assert fetched == ["NumAcc1"]
    rows = report.read_text().splitlines()
    assert rows[0] == "test,label,index,lre,cap,threshold,passed"
    assert len(rows) == 3
    result.stdout.fnmatch_lines(["*strdcheck: lowest LREs*", "*2.00 (min 1)*weak*"])


def test_strd_dataset_requires_param(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        def test_bad(strd_dataset):
            pass
        """
    )
    result = pytester.runpytest_inprocess()
    result.assert_outcomes(errors=1)


def _numacc1_dir(tmp_path, monkeypatch):
    d = tmp_path / "d"
    d.mkdir()
    (d / "NumAcc1.dat").write_text(NUMACC1, newline="")
    monkeypatch.setattr(st, "is_cached", lambda info, dd: (dd / info.filename).exists())
    return d


def test_check_univariate(tmp_path, monkeypatch) -> None:
    import numpy as np

    d = _numacc1_dir(tmp_path, monkeypatch)
    table = st.check_univariate(lambda y: {"mean": y.mean(), "sd": y.std(ddof=1)}, data_dir=d)
    assert table.missing and "NumAcc1" not in table.missing
    assert {r.quantity for r in table.rows} == {"mean", "sd"}
    assert table.min() == 15 and table.min_by_difficulty() == {"lower": 15}
    bad = st.check_univariate(lambda y: {"sd": y.std()}, data_dir=d)  # ddof=0 mistake
    assert bad.min() < 1.5
    err = st.check_univariate(lambda y: 1 / 0, data_dir=d)
    assert "ZeroDivisionError" in err.errors["NumAcc1"] and math.isnan(err.min())
    assert "NumAcc1" in st.check_univariate(np.mean, datasets=["numacc1"], data_dir=d).errors
    assert "dataset" in table.format()


def test_check_linear_and_nonlinear_real(strd_dir) -> None:
    import numpy as np

    lin = st.check_linear(
        lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0],
        datasets=["Norris", "NoInt1"],
        data_dir=strd_dir,
    )
    assert lin.min() > 10 and set(lin.by_dataset()) == {"Norris", "NoInt1"}

    from scipy.optimize import least_squares

    def fit(p):
        res = least_squares(
            lambda b: p.f(b, p.x) - p.y,
            p.start,
            jac=lambda b: np.column_stack([np.broadcast_to(c, p.y.shape) for c in p.jac(b, p.x)]),
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
        )
        return res.x

    nl = st.check_nonlinear(fit, start=2, datasets=["Misra1a", "DanWood"], data_dir=strd_dir)
    assert nl.min() >= 6
