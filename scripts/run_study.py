"""Run a study with automatic resume on failure (DESIGN 5.5, 6.5).

Each attempt runs ``strdcheck study`` in a fresh subprocess, so even a crashed worker or
interpreter does not lose finished units; a rerun skips them. Progress is in
``<run>/progress.log`` and ``<run>/study.log``.

Usage:
    uv run python scripts/run_study.py --config configs/quick.toml --run runs/quick
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=10.0)
    args = ap.parse_args(argv)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    log = run / "study.log"
    cmd = [
        sys.executable,
        "-m",
        "strdcheck.cli",
        "study",
        "--config",
        args.config,
        "--run",
        args.run,
    ]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    for attempt in range(1, args.attempts + 1):
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} run_study: attempt {attempt}/{args.attempts}\n")
        t0 = time.perf_counter()
        code = subprocess.call(cmd)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(
                f"{datetime.now(UTC).isoformat(timespec='seconds')} run_study: attempt "
                f"{attempt} exit {code} after {time.perf_counter() - t0:.0f} s\n"
            )
        if code == 0:
            return 0
        if attempt < args.attempts:
            time.sleep(args.backoff)
    return 1


if __name__ == "__main__":
    sys.exit(main())
