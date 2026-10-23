from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from strdcheck.checkpoint import (
    ConfigMismatch,
    RunStore,
    config_hash,
    execute,
    unit_filename,
    with_retries,
)
from strdcheck.parse import Certified
from strdcheck.results import EngineResult, QuantityRecord, UnitRecord, score


@dataclass(frozen=True)
class FakeUnit:
    unit_id: str
    part: str = "A"
    dataset: str = "D"
    adapter: str = "a"
    suite: str = "univariate"
    start: str | None = None


def ok_fn(u: FakeUnit) -> UnitRecord:
    if u.unit_id.endswith("boom"):
        raise ValueError("adapter failed")
    return UnitRecord(u.unit_id, u.part, u.dataset, u.adapter, u.suite, "ok", seconds=0.01)


def test_unit_record_roundtrip_with_nonfinite() -> None:
    q = QuantityRecord("sd", float("nan"), "0.1", 15, float("nan"), float("nan"), 8.2, float("nan"))
    r = UnitRecord(
        "u1",
        "A",
        "NumAcc4",
        "x",
        "univariate",
        "ok",
        quantities=[q],
        engine_status={"rank": 3},
        repeat_hashes=["a", "a", "a"],
        seconds=0.5,
        extra={"inf": float("inf")},
    )
    back = UnitRecord.from_json(r.to_json())
    assert back.quantities[0].ceiling == 8.2
    assert math.isnan(back.quantities[0].computed) and math.isnan(back.quantities[0].lre)
    assert back.repeat_identical is True
    assert back.extra["inf"] == "inf"  # extras are kept as stored
    json.loads(r.to_json())  # strict JSON (no NaN tokens)


def test_repeat_identical_flags() -> None:
    r = UnitRecord("u", "A", "d", "a", "s", "ok", repeat_hashes=["a", "b", "a"])
    assert r.repeat_identical is False
    assert UnitRecord("u", "A", "d", "a", "s", "ok").repeat_identical is None


def test_digest_is_bitwise() -> None:
    a = EngineResult({"x": 0.1 + 0.2})
    b = EngineResult({"x": 0.3})
    assert a.digest() != b.digest()
    assert a.digest() == EngineResult({"x": 0.30000000000000004}).digest()


def test_score() -> None:
    cert = {
        "mean": Certified.from_text("mean", "1.00000000000000"),
        "sd": Certified.from_text("sd", "0.000000000000000"),
        "f": Certified("f", "Infinity", float("inf"), 15),
    }
    res = EngineResult({"mean": 1.000001, "sd": 1e-10, "f": 1.0, "extra": 2.0}, derived={"sd"})
    rows = {q.name: q for q in score(cert, res, {"mean": 9.0})}
    assert set(rows) == {"mean", "sd"}  # non-finite certified and unknown names skipped
    assert rows["mean"].lre == pytest.approx(6.0)
    assert rows["mean"].deficit == pytest.approx(3.0)
    assert rows["sd"].lre == pytest.approx(10.0) and rows["sd"].derived
    assert math.isnan(rows["sd"].deficit)
    nan_row = score(cert, EngineResult({"mean": float("nan")}), {"mean": 9.0})[0]
    assert math.isnan(nan_row.lre) and math.isnan(nan_row.deficit)


def test_unit_filename_safe_and_unique() -> None:
    a, b = unit_filename("A|x/y:z"), unit_filename("A|x_y_z")
    assert a != b and "/" not in a and ":" not in a and "|" not in a


def test_config_mismatch_refused(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    h = store.open({"a": 1})
    assert h == config_hash({"a": 1})
    assert RunStore(tmp_path / "run").open({"a": 1}) == h
    with pytest.raises(ConfigMismatch):
        RunStore(tmp_path / "run").open({"a": 2})


def test_execute_inline_records_errors_and_skips(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.open({})
    units = [FakeUnit(f"u{i}") for i in range(5)] + [FakeUnit("u-boom")]
    assert execute(store, units, ok_fn, workers=1) == 6
    recs = {r.unit_id: r for r in store.records()}
    assert recs["u-boom"].status == "error" and "adapter failed" in recs["u-boom"].error
    assert execute(store, units, ok_fn, workers=1) == 0  # rerun skips all
    lines = (tmp_path / "run" / "progress.log").read_text().splitlines()
    assert len(lines) == 6
    with pytest.raises(ValueError, match="duplicate"):
        execute(store, [FakeUnit("x"), FakeUnit("x")], ok_fn, workers=1)
    with pytest.raises(ValueError, match="at most"):
        execute(store, units, ok_fn, workers=4)


def test_execute_process_pool(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run")
    store.open({})
    units = [FakeUnit(f"p{i}") for i in range(7)] + [FakeUnit("p-boom")]
    assert execute(store, units, ok_fn, workers=2) == 8
    assert store.count() == 8
    assert sum(r.status == "error" for r in store.records()) == 1


def test_kill_and_resume(tmp_path: Path) -> None:
    """M2 done-check: a run killed after k units resumes to exactly N files, no duplicates."""
    run = tmp_path / "run"
    script = tmp_path / "crashy.py"
    script.write_text(
        textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from test_checkpoint import FakeUnit, ok_fn
        from strdcheck.checkpoint import RunStore, execute

        def fn(u):
            if u.unit_id == "u4":
                os._exit(3)  # hard crash, no cleanup
            return ok_fn(u)

        store = RunStore({str(run)!r})
        store.open({{"n": 10}})
        execute(store, [FakeUnit(f"u{{i}}") for i in range(10)], fn, workers=1)
    """)
    )
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=120)
    assert proc.returncode == 3
    store = RunStore(run)
    assert store.count() == 4
    store.open({"n": 10})
    assert execute(store, [FakeUnit(f"u{i}") for i in range(10)], ok_fn, workers=1) == 6
    ids = [r.unit_id for r in store.records()]
    assert sorted(ids) == sorted(f"u{i}" for i in range(10))
    assert not list((run / "units").glob("*.part"))


def test_with_retries() -> None:
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("transient")
        return "done"

    logs: list[str] = []
    assert with_retries(flaky, attempts=3, log=logs.append, backoff_s=0) == "done"
    assert len(logs) == 2
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        with_retries(lambda: 1 / 0, attempts=2, log=logs.append, backoff_s=0)
