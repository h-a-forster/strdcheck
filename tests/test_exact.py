from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
import synth

from strdcheck import exact
from strdcheck.manifest import list_datasets
from strdcheck.parse import Certified, load, parse_text


def F(xs):
    return [Fraction(x) for x in xs]


def test_univariate_exact_values() -> None:
    r = exact.univariate(F([1, 2, 3, 4, 5]))
    assert r["mean"] == 3
    assert r["r1"] == Decimal("0.4")
    assert r["sd"] == exact.sqrt_fraction(Fraction(5, 2))


def test_univariate_constant_has_no_r1() -> None:
    r = exact.univariate(F([7, 7, 7]))
    assert r["sd"] == 0 and "r1" not in r
    with pytest.raises(ValueError):
        exact.univariate(F([1]))


def test_anova_exact_values() -> None:
    r = exact.anova(["a", "a", "a", "b", "b", "b"], F([1, 2, 3, 4, 5, 6]))
    assert r["ss_between"] == Decimal("13.5")
    assert r["ss_within"] == 4
    assert r["ms_within"] == 1
    assert r["f"] == Decimal("13.5")
    assert r["r2"] == exact.to_decimal(Fraction(27, 35))
    assert r["resid_sd"] == 1


def test_anova_constant_values_omit_ratio_quantities() -> None:
    r = exact.anova(["a", "b", "b"], F([1, 1, 1]))
    assert "f" not in r and "r2" not in r and r["ss_within"] == 0


def test_linear_exact_fit() -> None:
    x = F([0, 1, 2, 3])
    y = [1 + 2 * v for v in x]
    r = exact.linear([[Fraction(1)] * 4, x], y, ["b0", "b1"], intercept=True)
    assert r["b0"] == 1 and r["b1"] == 2
    assert r["se_b0"] == 0 and r["resid_sd"] == 0 and r["r2"] == 1


def test_linear_matches_numpy_on_noisy_data() -> None:
    rng = np.random.default_rng(7)
    x = rng.integers(-20, 20, size=30)
    y = 3 + 0.5 * x - 0.01 * x**2 + rng.integers(-3, 4, size=30)
    cols = [[Fraction(1)] * 30, F(x.tolist()), [Fraction(int(v)) ** 2 for v in x]]
    r = exact.linear(
        cols,
        [Fraction(float(v)).limit_denominator(1000) for v in y],
        ["b0", "b1", "b2"],
        intercept=True,
    )
    X = np.column_stack([np.ones(30), x, x**2]).astype(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    s2 = res @ res / 27
    se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
    for j in range(3):
        assert float(r[f"b{j}"]) == pytest.approx(b[j], rel=1e-10)
        assert float(r[f"se_b{j}"]) == pytest.approx(se[j], rel=1e-10)
    r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
    assert float(r["r2"]) == pytest.approx(r2, rel=1e-12)


def test_linear_no_intercept_r2_is_uncentred() -> None:
    x = F([1, 2, 3])
    y = F([2, 4, 7])
    r = exact.linear([x], y, ["b1"], intercept=False)
    b1 = Fraction(2 + 8 + 21, 14)
    rss = sum((yi - b1 * xi) ** 2 for xi, yi in zip(x, y, strict=True))
    assert r["b1"] == exact.to_decimal(b1)
    assert r["r2"] == exact.to_decimal(1 - rss / sum(v * v for v in y))


def test_invert_singular() -> None:
    with pytest.raises(ZeroDivisionError):
        exact._invert([[Fraction(1), Fraction(2)], [Fraction(2), Fraction(4)]])


def test_to_fractions_kinds() -> None:
    assert exact.to_fractions(["0.1"], "dec") == [Fraction(1, 10)]
    assert exact.to_fractions(["0.1"], "f64") == [Fraction(0.1)]
    assert exact.to_fractions(["0.1"], "f32") == [Fraction(float(np.float32(0.1)))]
    with pytest.raises(ValueError):
        exact.to_fractions(["1"], "f16")  # type: ignore[arg-type]


def test_lre_decimal() -> None:
    c = Certified.from_text("x", "1.00000000000000")
    assert exact.lre_decimal(Decimal("1.000001"), c)[0] == pytest.approx(6.0)
    assert exact.lre_decimal(Decimal(1), c) == (math.inf, 15.0)
    z = Certified.from_text("z", "0.000000000000000")
    assert exact.lre_decimal(Decimal("1e-9"), z)[0] == pytest.approx(9.0)


def test_ceiling_binds_for_large_offset() -> None:
    vals = ["1000000000000.4", "1000000000000.3", "1000000000000.5"]
    text, info = synth.univariate(vals, "1000000000000.4", "0.1", "-0.5", exact=True)
    c = exact.ceilings(parse_text(text, info))
    assert c["sd"].dec == 15 and c["r1"].dec == 15
    assert 1 < c["sd"].f64 < 6  # float64 spacing near 1e12 is 1.2e-4
    assert c["sd"].f32 == 0


def test_tiled_reference() -> None:
    y = np.array([1.5, 2.25, 10.0, -3.0])
    k1 = exact.tiled_reference(y, 1)
    ref = exact.univariate([Fraction(float(v)) for v in y])
    assert k1["mean"] == ref["mean"] and k1["sd"] == ref["sd"]
    k3 = exact.tiled_reference(y, 3)
    brute = exact.univariate([Fraction(float(v)) for v in np.tile(y, 3)])
    assert k3["mean"] == brute["mean"] and abs(k3["sd"] - brute["sd"]) < Decimal("1e-55")
    with pytest.raises(ValueError):
        exact.tiled_reference(y, 0)


def test_v1_certified_values_reproduced(strd_dir) -> None:
    """V1: exact statistics of the decimal data reproduce every certified value (>= cap - 1)."""
    for suite in ("univariate", "anova", "linear"):
        for info in list_datasets(suite):
            ds = load(info.name, strd_dir)
            for name, c in exact.ceilings(ds).items():
                cap = ds.certified[name].cap
                assert c.dec >= cap - 1, (info.name, name, c.dec)


def test_ceilings_known_values(strd_dir) -> None:
    c = exact.ceilings(load("NumAcc4", strd_dir))
    assert 8.0 < c["sd"].f64 < 8.5
    c = exact.ceilings(load("SmLs09", strd_dir))
    assert 3.5 < c["ss_within"].f64 < 4.5
    assert 3.5 < c["ss_between"].f64 < 4.5  # group means shift too
    assert exact.ceilings(load("SmLs03", strd_dir))["ss_within"].f64 == 15


@pytest.mark.slow
def test_v1_nonlinear(strd_dir) -> None:
    """Refinement from b* on decimal data reproduces all nonlinear certified values."""
    for info in list_datasets("nonlinear"):
        ds = load(info.name, strd_dir)
        ref = exact.nonlinear_refine(ds, "dec")
        assert ref.converged, info.name
        for name, cert in ds.certified.items():
            assert exact.lre_decimal(ref.values[name], cert)[1] >= 10, (info.name, name)


def test_nonlinear_at_certified_small(strd_dir) -> None:
    """RSS and SEs evaluated at b* match NIST (fast subset; Lanczos1 is excluded by design)."""
    for name in ("Misra1a", "DanWood", "BoxBOD", "MGH09", "Rat43", "Nelson", "Roszman1"):
        ds = load(name, strd_dir)
        at = exact.nonlinear_at(ds, [ds.certified[p].text for p in ds.param_names()])
        assert exact.lre_decimal(at["rss"], ds.certified["rss"])[1] >= 9
        for p in ds.param_names():
            assert exact.lre_decimal(at[f"se_{p}"], ds.certified[f"se_{p}"])[1] >= 8
