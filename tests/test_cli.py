from __future__ import annotations

import re
import subprocess
import sys
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
    # Windows on Python < 3.13 has a 16 ms monotonic clock, so up to 4 units (one worker) may
    # be queued before the limit registers; the rest stay for the next call.
    assert re.search(r"stopped at the time limit: [0-4]/[67] units", log)  # 7 with torch
    out = tmp_path / "res"
    assert main(["table", str(run), "--out", str(out)]) == 1
    assert not out.exists()


def test_study_failed_units_exit_code(tmp_path: Path) -> None:
    """Units that failed are reported by the exit code and not retried (a rerun skips them)."""
    from strdcheck.cli import EXIT_UNIT_ERRORS

    cfg = tmp_path / "c.toml"
    cfg.write_text(
        "[part_a]\nenabled = false\n[part_b]\nenabled = false\n"
        '[part_c]\ndatasets = ["NumAcc4"]\nsizes = [2002]\norders = ["tiled"]\n'
    )
    run = tmp_path / "run"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_study.py"
    proc = subprocess.run(  # the data directory is empty, so every unit fails
        [sys.executable, str(script), "--config", str(cfg), "--run", str(run)]
        + ["--data-dir", str(tmp_path), "--backoff", "0"],
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == EXIT_UNIT_ERRORS
    log = (run / "study.log").read_text()
    assert "attempt 2/3" not in log and "units failed" in log
    assert main(["table", str(run), "--out", str(tmp_path / "res")]) == 0


def test_list_and_verify_empty_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--suite", "linear", "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Norris" in out and "Wampler5" in out and "cached" not in out
    assert main(["verify", "--suite", "univariate", "--data-dir", str(tmp_path)]) == 1
