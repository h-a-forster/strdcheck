from __future__ import annotations

from pathlib import Path

import pytest

from strdcheck import __version__
from strdcheck.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_list_and_verify_empty_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--suite", "linear", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Norris" in out and "Wampler5" in out and "cached" not in out
    assert main(["verify", "--suite", "univariate", "--data-dir", str(tmp_path)]) == 1
