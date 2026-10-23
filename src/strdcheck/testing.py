"""Helpers for using StRD accuracy checks in your own test suite (``docs/DESIGN.md`` section 7).

Parametrise a test over datasets::

    import pytest
    from strdcheck.testing import params

    @pytest.mark.parametrize("strd_dataset", params("univariate"), indirect=True)
    def test_my_sd(strd_dataset, strd_assert_lre):
        y = strd_dataset.array("y")
        cert = strd_dataset.certified["sd"]
        strd_assert_lre([my_sd(y)], [cert.value], min_lre=10, cap=cert.cap)

Or run a callable over a whole suite and assert per difficulty tier::

    from strdcheck.testing import check_linear

    def test_my_ols():
        table = check_linear(lambda X, y: my_ols(X, y))
        assert table.min_by_difficulty()["lower"] >= 10
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strdcheck.fetch import fetch_one, is_cached, resolve_data_dir
from strdcheck.lre import lre
from strdcheck.manifest import DatasetInfo, Suite, list_datasets
from strdcheck.parse import Dataset, parse_file


def params(suite: Suite | None = None, difficulty: str | None = None) -> list[Any]:
    """``pytest.param`` objects (id = dataset name, marked ``strd(name)``) for parametrize."""
    import pytest

    return [
        pytest.param(d.name, id=d.name, marks=pytest.mark.strd(d.name))
        for d in list_datasets(suite, difficulty)
    ]


def load_dataset(
    name_or_info: str | DatasetInfo,
    data_dir: str | os.PathLike[str] | None = None,
    *,
    fetch: bool = False,
) -> Dataset:
    """Parse a dataset from the cache, downloading and verifying it first if ``fetch``."""
    from strdcheck.manifest import get_dataset_info

    info = name_or_info if isinstance(name_or_info, DatasetInfo) else get_dataset_info(name_or_info)
    d = resolve_data_dir(data_dir)
    if not is_cached(info, d):
        if not fetch:
            raise FileNotFoundError(f"{info.filename} not cached in {d}")
        fetch_one(info, d)
    return parse_file(d / info.filename, info)


@dataclass(frozen=True)
class Row:
    dataset: str
    difficulty: str
    quantity: str
    computed: float
    certified: float
    lre: float
    cap: int


@dataclass
class LRETable:
    """LREs of a user callable over a suite."""

    rows: list[Row] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def min(self) -> float:
        vals = [r.lre for r in self.rows]
        if not vals:
            return math.nan
        return math.nan if any(math.isnan(v) for v in vals) else min(vals)

    def min_by_difficulty(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.rows:
            cur = out.get(r.difficulty, math.inf)
            out[r.difficulty] = (
                math.nan if math.isnan(r.lre) or math.isnan(cur) else min(cur, r.lre)
            )
        return out

    def by_dataset(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.rows:
            cur = out.get(r.dataset, math.inf)
            out[r.dataset] = math.nan if math.isnan(r.lre) or math.isnan(cur) else min(cur, r.lre)
        return out

    def format(self) -> str:
        lines = [f"{'dataset':<10} {'quantity':<10} {'LRE':>6} {'cap':>4}"]
        for r in self.rows:
            lines.append(f"{r.dataset:<10} {r.quantity:<10} {r.lre:6.2f} {r.cap:4d}")
        return "\n".join(lines)


def _run(
    suite: Suite,
    call: Callable[[Dataset], Mapping[str, float]],
    datasets: Iterable[str] | None,
    data_dir: str | os.PathLike[str] | None,
    fetch: bool,
) -> LRETable:
    table = LRETable()
    wanted = {n.lower() for n in datasets} if datasets is not None else None
    for info in list_datasets(suite):
        if wanted is not None and info.name.lower() not in wanted:
            continue
        try:
            ds = load_dataset(info, data_dir, fetch=fetch)
        except FileNotFoundError:
            table.missing.append(info.name)
            continue
        try:
            values = call(ds)
            if not isinstance(values, Mapping):
                raise TypeError(f"expected a mapping of quantity names, got {type(values)}")
        except Exception as exc:  # recorded, not raised: one dataset must not hide the rest
            table.errors[info.name] = f"{type(exc).__name__}: {exc}"
            for name, c in ds.certified.items():
                if math.isfinite(c.value):
                    table.rows.append(
                        Row(info.name, info.difficulty, name, math.nan, c.value, math.nan, c.cap)
                    )
            continue
        for name, x in values.items():
            c = ds.certified.get(name)
            if c is None or not math.isfinite(c.value):
                continue
            r = lre(float(x), c.value, cap=c.cap)
            table.rows.append(
                Row(info.name, info.difficulty, name, float(x), c.value, r.value, c.cap)
            )
    return table


def check_univariate(
    fn: Callable[[np.ndarray], Mapping[str, float]],
    *,
    datasets: Iterable[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    fetch: bool = False,
) -> LRETable:
    """``fn(y)`` returns any of ``mean``, ``sd`` (n - 1) and ``r1`` for a float64 array."""
    return _run("univariate", lambda ds: fn(ds.array("y")), datasets, data_dir, fetch)


def check_linear(
    fn: Callable[[np.ndarray, np.ndarray], Sequence[float] | Mapping[str, float]],
    *,
    datasets: Iterable[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    fetch: bool = False,
) -> LRETable:
    """``fn(X, y)`` returns the coefficients (in column order) or a mapping of quantity names.

    ``X`` holds the intercept column (unless the model has none) and the raw powers or
    predictors of the NIST model, in float64.
    """
    from strdcheck.engines.linear import design

    def call(ds: Dataset) -> Mapping[str, float]:
        X, y = design(ds)
        out = fn(X, y)
        if isinstance(out, Mapping):
            return out
        return dict(zip(ds.param_names(), np.asarray(out, dtype=float).ravel(), strict=True))

    return _run("linear", call, datasets, data_dir, fetch)


@dataclass(frozen=True)
class NonlinearProblem:
    """What ``check_nonlinear`` passes to the user's fitting function."""

    name: str
    f: Callable[[Sequence[float], tuple[np.ndarray, ...]], np.ndarray]
    jac: Callable[[Sequence[float], tuple[np.ndarray, ...]], list[np.ndarray]]
    x: tuple[np.ndarray, ...]
    y: np.ndarray
    start: tuple[float, ...]


def check_nonlinear(
    fn: Callable[[NonlinearProblem], Sequence[float] | Mapping[str, float]],
    *,
    start: int = 1,
    datasets: Iterable[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    fetch: bool = False,
) -> LRETable:
    """``fn(problem)`` fits ``problem.f(b, x)`` to ``problem.y`` from ``problem.start``
    (NIST Start ``start``) and returns the parameters or a mapping of quantity names."""
    from strdcheck.models import NUMPY, get_model, predictors, response

    def call(ds: Dataset) -> Mapping[str, float]:
        model = get_model(ds.name)
        prob = NonlinearProblem(
            ds.name,
            lambda b, x: model(b, x, NUMPY),
            lambda b, x: model.jac(b, x, NUMPY),
            predictors(ds),
            response(model, ds.array("y")),
            ds.starts[start - 1],
        )
        out = fn(prob)
        if isinstance(out, Mapping):
            return out
        return dict(zip(ds.param_names(), np.asarray(out, dtype=float).ravel(), strict=True))

    return _run("nonlinear", call, datasets, data_dir, fetch)
