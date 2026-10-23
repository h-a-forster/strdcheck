from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import synth

from strdcheck.manifest import all_datasets
from strdcheck.parse import LINEAR_SPECS, Certified, ParseError, parse_file, parse_text


def test_univariate_synthetic() -> None:
    text, info = synth.univariate(["1", "2", "3", "4", "5"], "3", "1.58113883008419", "0.4")
    ds = parse_text(text, info)
    assert ds.columns == ("y",)
    assert ds.raw["y"] == ("1", "2", "3", "4", "5")
    assert ds.n == 5
    assert ds.certified["sd"].cap == 15
    assert ds.certified["r1"].value == 0.4
    assert ds.certified["r1"].cap == 1
    np.testing.assert_array_equal(ds.array("y"), [1, 2, 3, 4, 5])
    assert ds.array("y", np.float32).dtype == np.float32


def test_univariate_exact_flag() -> None:
    text, info = synth.univariate(["1", "3", "2"], "2", "1", "-0.5", exact=True)
    ds = parse_text(text, info)
    assert all(c.exact and c.cap == 15 for c in ds.certified.values())


def test_anova_synthetic() -> None:
    cert = {
        "ss_between": "1.35000000000000E+01",
        "ms_between": "1.35000000000000E+01",
        "f": "1.35000000000000E+01",
        "ss_within": "4.00000000000000E+00",
        "ms_within": "1.00000000000000E+00",
        "r2": "7.71428571428571E-01",
        "resid_sd": "1.00000000000000E+00",
    }
    text, info = synth.anova([1, 1, 1, 2, 2, 2], ["1", "2", "3", "4", "5", "6"], cert, (1, 4))
    ds = parse_text(text, info)
    assert ds.columns == ("instrument", "resistance")
    assert ds.response == "resistance"
    assert ds.df == {"between": 1, "within": 4}
    assert set(ds.certified) == set(cert)
    assert ds.certified["f"].value == 13.5


def test_linear_synthetic() -> None:
    params = {"b0": ("1.00000000000000", "0.0"), "b1": ("2.00000000000000", "0.0")}
    text, info = synth.linear(["3", "5", "7"], ["1", "2", "3"], params, "0.0", "1.00000000000000")
    ds = parse_text(text, info)
    assert ds.param_names() == ("b0", "b1")
    assert ds.certified["se_b0"].value == 0.0
    assert ds.certified["se_b0"].cap == 15  # certified zero
    assert ds.df == {"residual": 1}
    assert ds.response == "y"


def test_nonlinear_synthetic() -> None:
    params = [
        ("500", "250", "2.3894212918E+02", "2.7070075241E+00"),
        ("0.0001", "0.0005", "5.5015643181E-04", "7.2668688436E-06"),
    ]
    text, info = synth.nonlinear(
        ["10.07E0", "14.73E0", "17.94E0"],
        ["77.6E0", "114.9E0", "141.1E0"],
        params,
        "1.2455138894E-01",
        "1.0187876330E-01",
    )
    ds = parse_text(text, info)
    assert ds.starts == ((500.0, 0.0001), (250.0, 0.0005))
    assert ds.start_texts[1] == ("250", "0.0005")
    assert ds.certified["b1"].cap == 11
    assert ds.n_params == 2
    assert ds.param_names() == ("b1", "b2")
    assert ds.array("y")[0] == 10.07


def test_row_count_mismatch() -> None:
    text, info = synth.univariate(["1", "2", "3"], "2", "1", "0")
    with pytest.raises(ParseError, match="data rows"):
        parse_text(text, dataclasses.replace(info, n_obs=4))


def test_bad_field_and_number() -> None:
    text, info = synth.linear(
        ["3", "5"], ["1", "2"], {"b0": ("1", "0"), "b1": ("2", "0")}, "0", "1"
    )
    with pytest.raises(ParseError, match="fields"):
        parse_text(text.replace("3    1\r\n", "3    1  9\r\n"), info)
    with pytest.raises(ParseError, match="not a number"):
        parse_text(text.replace("3    1\r\n", "3    abc\r\n"), info)


def test_missing_certified_and_range() -> None:
    text, info = synth.univariate(["1", "2", "3"], "2", "1", "0")
    with pytest.raises(ParseError, match="r1"):
        parse_text(text.replace("r(1):", "r(x):"), info)
    with pytest.raises(ParseError, match="lines a to b"):
        parse_text(text.replace("lines 61", "rows 61"), info)


def test_certified_from_text() -> None:
    c = Certified.from_text("b1", "-0.358191792925910E-01")
    assert c.cap == 15 and c.value == pytest.approx(-0.0358191792925910)
    assert Certified.from_text("se", "0.000000000000000").cap == 15


def test_linear_specs_cover_manifest() -> None:
    names = {d.name for d in all_datasets() if d.suite == "linear"}
    assert names == set(LINEAR_SPECS)
    assert LINEAR_SPECS["Filip"].n_params == 11
    assert LINEAR_SPECS["NoInt1"].param_names() == ("b1",)
    assert LINEAR_SPECS["Longley"].param_names()[-1] == "b6"


def test_real_files_parse(strd_dir) -> None:
    """All 58 files parse; row counts equal the manifest (checked inside the parser)."""
    for info in all_datasets():
        ds = parse_file(strd_dir / info.filename, info)
        assert ds.n == info.n_obs
        if ds.suite == "nonlinear":
            assert len(ds.starts) == 2
            assert len(ds.certified) == 2 * ds.n_params + 2
        if ds.suite == "linear":
            assert len(ds.certified) == 2 * ds.n_params + 2
    rat43 = parse_file(strd_dir / "Rat43.dat")
    assert rat43.df == {"residual": 11, "printed_residual": 9}  # NIST header typo
