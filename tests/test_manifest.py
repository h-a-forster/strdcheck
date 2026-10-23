from __future__ import annotations

import re

import pytest

from strdcheck.manifest import all_datasets, get_dataset_info, list_datasets, manifest_meta

EXPECTED_COUNTS = {"univariate": 9, "anova": 11, "linear": 11, "nonlinear": 27}


def test_counts_per_suite() -> None:
    for suite, n in EXPECTED_COUNTS.items():
        assert len(list_datasets(suite)) == n  # type: ignore[arg-type]
    assert len(all_datasets()) == sum(EXPECTED_COUNTS.values())


def test_entries_are_well_formed() -> None:
    names = [d.name for d in all_datasets()]
    assert len(set(names)) == len(names)
    for d in all_datasets():
        assert re.fullmatch(r"[0-9a-f]{64}", d.sha256)
        assert d.url.startswith("https://www.itl.nist.gov/div898/strd/")
        assert d.url.endswith(f"/{d.name}.dat")
        assert d.difficulty in ("lower", "average", "higher")
        assert d.n_obs >= 3
        assert 1000 < d.bytes < 700_000


def test_known_facts() -> None:
    assert get_dataset_info("norris").n_obs == 36
    assert get_dataset_info("SmLs09").n_obs == 18009
    assert get_dataset_info("Filip").difficulty == "higher"
    assert get_dataset_info("Bennett5").suite == "nonlinear"
    assert len(list_datasets("nonlinear", "higher")) == 8
    assert manifest_meta()["doi"] == "10.18434/T43G6C"


def test_unknown() -> None:
    with pytest.raises(KeyError):
        get_dataset_info("nope")
    with pytest.raises(ValueError):
        list_datasets("bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        list_datasets(difficulty="hard")
