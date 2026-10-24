"""Resumable one-file-per-unit result store (``docs/DESIGN.md`` sections 5.5 and 6.5).

Layout of a run directory::

    runs/<run>/manifest.json   config hash, config, environment record (of the first attempt)
    runs/<run>/attempts.jsonl  one line per open: time and code/package provenance
    runs/<run>/units/*.json    one UnitRecord per finished unit (atomic write)
    runs/<run>/progress.log    one line per finished unit: time, unit id, status, seconds
    runs/<run>/study.log       free-form log of the study command

A rerun skips units whose file exists and refuses to continue a run directory created with
a different configuration hash. A rerun whose code or package versions differ from the
first attempt (``meta["environment"]``, compared with :func:`strdcheck.env.provenance`) is
allowed but logged as a warning in ``study.log`` and recorded in ``attempts.jsonl``, so a
run mixing units from different code is visible to the analysis. :func:`execute` runs the
remaining units in at most ``MAX_WORKERS`` spawned processes and writes each record as soon
as it arrives, so an interrupted run loses at most the units in flight.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from strdcheck.results import UnitRecord

MAX_WORKERS = 3


class ConfigMismatch(RuntimeError):
    """The run directory was created with a different configuration."""


def config_hash(config: Any) -> str:
    """Stable hash of a JSON-serialisable configuration."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def unit_filename(unit_id: str) -> str:
    """Readable, filesystem-safe and collision-free file name for a unit id."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", unit_id)[:120]
    return f"{safe}.{hashlib.sha1(unit_id.encode()).hexdigest()[:8]}.json"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class RunStore:
    """A run directory holding finished units."""

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.dir = Path(run_dir)
        self.units_dir = self.dir / "units"

    # -- lifecycle ------------------------------------------------------------------------

    def open(self, config: Any, meta: dict[str, Any] | None = None) -> str:
        """Create or reopen the run for ``config``; return its hash."""
        from strdcheck.env import provenance

        h = config_hash(config)
        man = self.dir / "manifest.json"
        now = provenance((meta or {}).get("environment") or {})
        if man.exists():
            old = json.loads(man.read_text("utf-8"))
            if old.get("config_hash") != h:
                raise ConfigMismatch(
                    f"{self.dir} was created with config {old.get('config_hash')}, not {h}; "
                    "use a new run directory"
                )
            first = provenance(old.get("environment") or {})
            if meta is not None and now != first:
                diff = sorted(k for k in now if now[k] != first[k])
                self.log(
                    "WARNING: resuming with different code or packages than the first attempt "
                    f"({', '.join(diff)}); units of this run no longer share one provenance"
                )
        else:
            body = {
                "config_hash": h,
                "config": config,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                **(meta or {}),
            }
            atomic_write(man, json.dumps(body, indent=1, sort_keys=True, default=str))
        self.units_dir.mkdir(parents=True, exist_ok=True)
        if meta is not None:
            line = {
                "opened": datetime.now(UTC).isoformat(timespec="seconds"),
                "units_done": self.count(),
                "provenance": now,
            }
            with (self.dir / "attempts.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(line, sort_keys=True, default=str) + "\n")
        return h

    def manifest(self) -> dict[str, Any]:
        return json.loads((self.dir / "manifest.json").read_text("utf-8"))

    def attempts(self) -> list[dict[str, Any]]:
        """Every recorded open of this run, oldest first (empty for runs made before it)."""
        p = self.dir / "attempts.jsonl"
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x.strip()]

    def mixed_provenance(self) -> bool:
        """True when an attempt that ran units had different code or packages than the first.

        An attempt that found every unit done ran nothing and is ignored.
        """
        att = self.attempts()
        if not att:
            return False
        first = att[0]["provenance"]
        total = self.count()
        return any(a["provenance"] != first and a["units_done"] < total for a in att[1:])

    # -- units ------------------------------------------------------------------------------

    def path(self, unit_id: str) -> Path:
        return self.units_dir / unit_filename(unit_id)

    def is_done(self, unit_id: str) -> bool:
        return self.path(unit_id).exists()

    def write(self, rec: UnitRecord) -> None:
        atomic_write(self.path(rec.unit_id), rec.to_json())
        self.log_progress(rec.unit_id, rec.status, rec.seconds)

    def records(self) -> list[UnitRecord]:
        out = []
        for p in sorted(self.units_dir.glob("*.json")):
            out.append(UnitRecord.from_json(p.read_text("utf-8")))
        return out

    def count(self) -> int:
        return sum(1 for _ in self.units_dir.glob("*.json")) if self.units_dir.exists() else 0

    # -- logs ---------------------------------------------------------------------------------

    def log_progress(self, unit_id: str, status: str, seconds: float) -> None:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        line = f"{stamp}\t{unit_id}\t{status}\t{seconds:.3f}\n"
        with (self.dir / "progress.log").open("a", encoding="utf-8") as fh:
            fh.write(line)

    def log(self, message: str) -> None:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        with (self.dir / "study.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {message}\n")


UnitFn = Callable[[Any], UnitRecord]


def _error_record(unit: Any, exc: BaseException) -> UnitRecord:
    return UnitRecord(
        unit_id=unit.unit_id,
        part=getattr(unit, "part", "?"),
        dataset=getattr(unit, "dataset", "?"),
        adapter=getattr(unit, "adapter", "?"),
        suite=getattr(unit, "suite", "?"),
        start=getattr(unit, "start", None),
        status="error",
        error="".join(traceback.format_exception_only(type(exc), exc)).strip()[:2000],
    )


def execute(
    store: RunStore,
    units: Sequence[Any],
    fn: UnitFn,
    *,
    workers: int = MAX_WORKERS,
    initializer: Callable[..., None] | None = None,
    initargs: Iterable[Any] = (),
    progress_every: int = 50,
    spawn: bool = False,
) -> int:
    """Run every unit not yet in ``store``; return the number of units run now.

    ``fn`` must be a picklable top-level function when ``workers > 1`` or ``spawn`` is set
    (``spawn`` runs even a single worker in a fresh process, so environment settings such as
    thread counts apply before the engines are imported). A unit that raises
    is recorded with ``status = "error"`` (never dropped). A worker process that dies breaks
    the pool; the exception propagates and the retry wrapper resumes the run.
    """
    if workers > MAX_WORKERS:
        raise ValueError(f"at most {MAX_WORKERS} workers")
    todo = [u for u in units if not store.is_done(u.unit_id)]
    ids = [u.unit_id for u in units]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate unit ids")
    total = len(units)
    store.log(f"execute: {total - len(todo)}/{total} done, {len(todo)} to run, workers={workers}")
    t0 = time.perf_counter()
    done_now = 0

    def finished(rec: UnitRecord) -> None:
        nonlocal done_now
        store.write(rec)
        done_now += 1
        if done_now % progress_every == 0 or done_now == len(todo):
            rate = done_now / max(time.perf_counter() - t0, 1e-9)
            remaining = (len(todo) - done_now) / rate if rate > 0 else float("nan")
            store.log(
                f"progress: {total - len(todo) + done_now}/{total} "
                f"({rate:.2f} units/s, about {remaining:.0f} s left)"
            )

    if workers <= 1 and not spawn:
        if initializer is not None:
            initializer(*initargs)
        for u in todo:
            try:
                rec = fn(u)
            except Exception as exc:
                rec = _error_record(u, exc)
            finished(rec)
        return done_now

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(workers, 1),
        mp_context=ctx,
        initializer=initializer,
        initargs=tuple(initargs),
    ) as pool:
        pending: dict[Future[UnitRecord], Any] = {}
        it = iter(todo)

        def submit_next() -> bool:
            u = next(it, None)
            if u is None:
                return False
            pending[pool.submit(fn, u)] = u
            return True

        for _ in range(max(workers, 1) * 4):
            if not submit_next():
                break
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in ready:
                u = pending.pop(fut)
                exc = fut.exception()
                if exc is not None:
                    if type(exc).__name__ == "BrokenProcessPool":
                        raise exc
                    finished(_error_record(u, exc))
                else:
                    finished(fut.result())
                submit_next()
    return done_now


def with_retries(
    run: Callable[[], Any],
    *,
    attempts: int = 3,
    log: Callable[[str], None] = print,
    backoff_s: float = 5.0,
) -> Any:
    """Call ``run`` until it succeeds, at most ``attempts`` times (it must be resumable)."""
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return run()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last = exc
            log(f"attempt {i}/{attempts} failed: {type(exc).__name__}: {exc}")
            if i < attempts:
                time.sleep(backoff_s)
    raise RuntimeError(f"failed after {attempts} attempts") from last
