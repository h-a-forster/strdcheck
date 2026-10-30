"""Run a study with automatic resume on failure (DESIGN 5.5, 6.5).

Each attempt runs ``strdcheck study`` in a fresh subprocess, so even a crashed worker or
interpreter does not lose finished units; a rerun skips them. Progress is in
``<run>/progress.log`` and ``<run>/study.log``. With ``--max-minutes`` an attempt queues no
unit later than that after its setup (up to 4 per worker queued before then still run); a
run stopped there exits with code 3 (not retried)
and the next call resumes it, which splits a long study into bounded chunks.

Usage:
    uv run python scripts/run_study.py --config configs/quick.toml --run runs/quick
    uv run python scripts/run_study.py --config configs/full.toml --run runs/full --max-minutes 40
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from strdcheck.cli import EXIT_TIME_LIMIT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=10.0)
    ap.add_argument("--max-minutes", type=float, default=None)
    args = ap.parse_args(argv)
    if args.max_minutes is not None and args.max_minutes <= 0:
        ap.error("--max-minutes must be positive")
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
    if args.max_minutes is not None:
        cmd += ["--max-minutes", str(args.max_minutes)]
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
        if code in (0, EXIT_TIME_LIMIT):
            return code
        if attempt < args.attempts:
            time.sleep(args.backoff)
    return 1


if __name__ == "__main__":
    sys.exit(main())
