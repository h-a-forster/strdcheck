from __future__ import annotations

import math

import pytest

from strdcheck.lre import EXACT_CAP, clip_lre, lre, lre_from_string, raw_lre, significant_digits


@pytest.mark.parametrize(
    ("text", "digits"),
    [
        ("4.53480000000000", 15),
        ("-0.00355099287237972", 15),
        ("2.3894212918E+02", 11),
        ("5.5015643181E-04", 11),
        ("0.429796848199937E-03", 15),
        ("-1467.48961422980", 15),
        ("10000000.2", 9),
        ("0.1", 1),
        ("0.000000000000000", 0),
        ("1.60080000000000E+02", 15),
        ("15750.2500000000", 15),
        ("60323", 5),
    ],
)
def test_significant_digits(text: str, digits: int) -> None:
    assert significant_digits(text) == digits


@pytest.mark.parametrize("bad", ["", "Infinity", "abc", ".", "1.2.3"])
def test_significant_digits_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        significant_digits(bad)


def test_raw_lre_relative_and_absolute() -> None:
    assert raw_lre(1.0001, 1.0) == pytest.approx(4.0, abs=1e-9)
    assert raw_lre(-2.002, -2.0) == pytest.approx(3.0, abs=1e-9)
    assert raw_lre(1e-7, 0.0) == pytest.approx(7.0, abs=1e-9)  # log absolute error
    assert raw_lre(3.0, 3.0) == math.inf
    assert math.isnan(raw_lre(math.nan, 1.0))
    assert math.isnan(raw_lre(math.inf, 1.0))
    with pytest.raises(ValueError):
        raw_lre(1.0, math.inf)


def test_clip_lre() -> None:
    assert clip_lre(math.inf, 11) == 11
    assert clip_lre(17.3, 15) == 15
    assert clip_lre(0.7, 15) == 0.0  # not even the leading digit is right
    assert clip_lre(-3.0, 15) == 0.0
    assert clip_lre(4.25, 15) == 4.25
    assert math.isnan(clip_lre(math.nan, 15))
    with pytest.raises(ValueError):
        clip_lre(1.0, 0)


def test_lre_dataclass() -> None:
    r = lre(0.0, 0.0)
    assert r.value == EXACT_CAP and r.absolute
    r = lre(2.0, 1.0, cap=11)
    assert r.raw == pytest.approx(0.0) and r.value == 0.0 and not r.absolute


def test_lre_from_string_caps() -> None:
    # 11-digit nonlinear certified value: an exact match reports 11.
    assert lre_from_string(238.94212918, "2.3894212918E+02").value == 11
    # Exact certified values (NumAcc) get the full cap even if printed with few digits.
    assert lre_from_string(0.1, "0.1", exact=True).value == EXACT_CAP
    assert lre_from_string(0.1, "0.1").value == 1
    # A certified zero (Wampler1 standard errors) uses the absolute error and the exact cap.
    r = lre_from_string(1e-9, "0.000000000000000")
    assert r.absolute and r.value == pytest.approx(9.0)
