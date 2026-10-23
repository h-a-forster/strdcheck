"""pytest plugin (entry point ``pytest11: strdcheck``).

Available now: the ``strd`` marker, the ``--strd-data-dir`` and ``--strd-min-lre`` options
and the ``strd_assert_lre`` fixture. Dataset parametrisation helpers and the LRE summary
report arrive with milestone M6 (``docs/DESIGN.md`` section 7).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from strdcheck.fetch import resolve_data_dir
from strdcheck.lre import lre

AssertLRE = Callable[..., float]


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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "strd(name): test that checks a computation against a NIST StRD dataset"
    )


@pytest.fixture
def strd_data_dir(request: pytest.FixtureRequest) -> Path:
    """The StRD cache directory selected by ``--strd-data-dir`` or the environment."""
    return resolve_data_dir(request.config.getoption("--strd-data-dir"))


@pytest.fixture
def strd_assert_lre(request: pytest.FixtureRequest) -> AssertLRE:
    """Assert that every computed value has at least ``min_lre`` correct digits.

    Usage::

        def test_mean(strd_assert_lre):
            strd_assert_lre([computed], [certified], min_lre=10, cap=15)

    Returns the smallest reported LRE. Non-finite computed values fail the assertion.
    """
    override = request.config.getoption("--strd-min-lre")

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
        worst = min(values) if values else math.inf
        if any(math.isnan(v) for v in values) or worst < threshold:
            where = f" ({label})" if label else ""
            raise AssertionError(
                f"LRE below {threshold}{where}: "
                + ", ".join("nan" if math.isnan(v) else f"{v:.2f}" for v in values)
            )
        return worst

    return _assert
