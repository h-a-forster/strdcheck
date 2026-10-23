"""pytest plugin (entry point ``pytest11: strdcheck``), ``docs/DESIGN.md`` section 7.

- marker ``strd(name)``;
- options ``--strd-data-dir``, ``--strd-min-lre``, ``--strd-fetch``, ``--strd-report PATH``;
- fixtures ``strd_data_dir``, ``strd_assert_lre`` and ``strd_dataset`` (use with
  ``indirect=True`` and :func:`strdcheck.testing.params`).

Every ``strd_assert_lre`` call is recorded; ``--strd-report`` writes them to a CSV and the
terminal summary lists the five lowest LREs.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from strdcheck.fetch import resolve_data_dir
from strdcheck.lre import lre

AssertLRE = Callable[..., float]


@dataclass(frozen=True)
class Record:
    test: str
    label: str
    index: int
    lre: float
    cap: int
    threshold: float
    passed: bool


_RECORDS = pytest.StashKey[list[Record]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("strdcheck", "NIST StRD numerical accuracy checks")
    group.addoption(
        "--strd-data-dir",
        default=None,
        help="directory with the StRD .dat files (default: STRDCHECK_DATA_DIR or user cache)",
    )
    group.addoption(
        "--strd-min-lre",
        type=float,
        default=None,
        help="override the min_lre argument of every strd_assert_lre call",
    )
    group.addoption(
        "--strd-fetch",
        action="store_true",
        default=False,
        help="download (and checksum-verify) StRD files that are not cached",
    )
    group.addoption(
        "--strd-report",
        default=None,
        metavar="PATH",
        help="write every strd_assert_lre comparison to a CSV file",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "strd(name): test that checks a computation against a NIST StRD dataset"
    )
    config.stash[_RECORDS] = []


@pytest.fixture
def strd_data_dir(request: pytest.FixtureRequest) -> Path:
    """The StRD cache directory selected by ``--strd-data-dir`` or the environment."""
    return resolve_data_dir(request.config.getoption("--strd-data-dir"))


@pytest.fixture
def strd_dataset(request: pytest.FixtureRequest, strd_data_dir: Path):
    """Parsed :class:`strdcheck.parse.Dataset` named by the indirect parameter.

    Skips when the file is not cached, unless ``--strd-fetch`` is given.
    """
    from strdcheck.testing import load_dataset

    name = getattr(request, "param", None)
    if name is None:
        raise pytest.UsageError("strd_dataset needs a dataset name via indirect parametrize")
    try:
        return load_dataset(name, strd_data_dir, fetch=request.config.getoption("--strd-fetch"))
    except FileNotFoundError:
        pytest.skip(f"StRD file {name}.dat not cached (run `strdcheck fetch` or use --strd-fetch)")


@pytest.fixture
def strd_assert_lre(request: pytest.FixtureRequest) -> AssertLRE:
    """Assert that every computed value has at least ``min_lre`` correct digits.

    Usage::

        def test_mean(strd_assert_lre):
            strd_assert_lre([computed], [certified], min_lre=10, cap=15)

    Returns the smallest reported LRE. Non-finite computed values fail the assertion.
    """
    override = request.config.getoption("--strd-min-lre")
    records = request.config.stash[_RECORDS]
    nodeid = request.node.nodeid

    def _assert(
        computed: Sequence[float] | np.ndarray,
        certified: Sequence[float] | np.ndarray,
        *,
        min_lre: float,
        cap: int = 15,
        label: str = "",
    ) -> float:
        threshold = override if override is not None else min_lre
        x = np.atleast_1d(np.asarray(computed, dtype=float))
        c = np.atleast_1d(np.asarray(certified, dtype=float))
        if x.shape != c.shape:
            raise ValueError(f"shape mismatch: computed {x.shape} vs certified {c.shape}")
        values = [lre(xi, ci, cap=cap).value for xi, ci in zip(x.ravel(), c.ravel(), strict=True)]
        for i, v in enumerate(values):
            ok = not math.isnan(v) and v >= threshold
            records.append(Record(nodeid, label, i, v, cap, threshold, ok))
        worst = min(values) if values else math.inf
        if any(math.isnan(v) for v in values) or worst < threshold:
            where = f" ({label})" if label else ""
            raise AssertionError(
                f"LRE below {threshold}{where}: "
                + ", ".join("nan" if math.isnan(v) else f"{v:.2f}" for v in values)
            )
        return worst

    return _assert


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    records = config.stash.get(_RECORDS, [])
    path = config.getoption("--strd-report")
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["test", "label", "index", "lre", "cap", "threshold", "passed"])
            for r in records:
                w.writerow([r.test, r.label, r.index, f"{r.lre:.3f}", r.cap, r.threshold, r.passed])
    if not records:
        return
    terminalreporter.section("strdcheck: lowest LREs")
    worst = sorted(records, key=lambda r: -1.0 if math.isnan(r.lre) else r.lre)[:5]
    for r in worst:
        lab = f" [{r.label}]" if r.label else ""
        val = "nan" if math.isnan(r.lre) else f"{r.lre:.2f}"
        terminalreporter.write_line(f"{val:>6} (min {r.threshold:g}) {r.test}{lab}")
    if path:
        terminalreporter.write_line(f"{len(records)} comparisons written to {path}")
