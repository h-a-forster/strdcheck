from __future__ import annotations

import strdcheck


def test_public_api() -> None:
    assert strdcheck.__version__
    for name in strdcheck.__all__:
        assert hasattr(strdcheck, name)


def test_sdist_ships_what_the_tests_read() -> None:
    """tests/test_study.py loads configs/*.toml, so the sdist must include configs/."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    inc = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["tool"]["hatch"]["build"][
        "targets"
    ]["sdist"]["include"]
    for d in ("src/strdcheck", "tests", "configs", "scripts"):
        assert d in inc
