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


def test_study_time_limit(tmp_path: Path) -> None:
    """--max-minutes is validated first, and a run stopped by it exits 3 without retries."""
    import subprocess
    import sys

    from strdcheck.cli import EXIT_TIME_LIMIT

    cfg = tmp_path / "c.toml"
    cfg.write_text(
        "[part_a]\nenabled = false\n[part_b]\nenabled = false\n"
        '[part_c]\ndatasets = ["NumAcc4"]\nsizes = [2002]\norders = ["tiled"]\n'
    )
    run = tmp_path / "run"
    base = ["study", "--config", str(cfg), "--run", str(run), "--data-dir", str(tmp_path)]
    assert main([*base, "--max-minutes", "0"]) == 2
    assert not run.exists()
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_study.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfg), "--run", str(run)]
        + ["--data-dir", str(tmp_path), "--backoff", "0", "--max-minutes", "1e-9"],
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == EXIT_TIME_LIMIT
    log = (run / "study.log").read_text()
    assert "attempt 1/3" in log and "attempt 2/3" not in log
    assert "stopped at the time limit: 0/" in log


def test_list_and_verify_empty_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--suite", "linear", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Norris" in out and "Wampler5" in out and "cached" not in out
    assert main(["verify", "--suite", "univariate", "--data-dir", str(tmp_path)]) == 1
