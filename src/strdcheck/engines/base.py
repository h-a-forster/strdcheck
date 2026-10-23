"""Adapter interface (``docs/DESIGN.md`` section 5.4).

An adapter is a named engine call path. ``fn(ds, **kw)`` returns the computed values (keys
match :class:`strdcheck.parse.Certified` names) plus optional derived names, status and
warnings; :meth:`Adapter.run` times the call and records warnings. Engines are imported
inside ``fn``, never at module import.
"""

from __future__ import annotations

import importlib.util
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from strdcheck.parse import Dataset
from strdcheck.results import EngineResult

Dtype = Literal["f64", "f32", "dec"]
AdapterFn = Callable[..., Any]


@dataclass(frozen=True)
class Adapter:
    """One engine configuration.

    Attributes:
        id: configuration id, e.g. ``"polars.lazy-streaming.f64"``.
        suite: StRD suite it applies to.
        engine: import name of the engine (availability check, version capture).
        dtype: data representation the engine works on; selects the ceiling (``dec`` means
            the decimal strings are passed, so the ceiling is the certified precision).
        fn: ``fn(ds, **kw) -> EngineResult | dict[str, float]``.
        applies: optional predicate on the dataset (e.g. ``linregress`` only for Norris).
        role: ``"engine"`` or ``"control"`` (positive/negative controls, DESIGN 9 V3).
        note: one-line description for the report.
    """

    id: str
    suite: str
    engine: str
    dtype: Dtype
    fn: AdapterFn
    applies: Callable[[Dataset], bool] | None = None
    role: str = "engine"
    note: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)

    def available(self) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in self.engine.split("+"))

    def applicable(self, ds: Dataset) -> bool:
        return ds.suite == self.suite and (self.applies is None or self.applies(ds))

    def run(self, ds: Dataset, **kw: Any) -> EngineResult:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            out = self.fn(ds, **{**self.kwargs, **kw})
            seconds = time.perf_counter() - t0
        res = out if isinstance(out, EngineResult) else EngineResult(dict(out))
        res.values = {k: float(v) for k, v in res.values.items()}
        res.seconds = seconds
        seen = list(dict.fromkeys(f"{w.category.__name__}: {w.message}" for w in caught))
        res.warnings = list(dict.fromkeys([*res.warnings, *seen]))[:20]
        return res


def duckdb_connect() -> Any:
    """In-memory DuckDB connection limited to one thread (DESIGN 5.4 threading)."""
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.execute("SET threads = 1")
    return con


def decimal_type(strings: tuple[str, ...]) -> str:
    """Smallest DuckDB ``DECIMAL(p, s)`` that holds every decimal string exactly."""
    scale = 0
    int_digits = 1
    for s in strings:
        t = s.lstrip("+-")
        if "e" in t.lower():
            raise ValueError(f"exponent notation not supported for DECIMAL: {s}")
        whole, _, frac = t.partition(".")
        scale = max(scale, len(frac))
        int_digits = max(int_digits, len(whole.lstrip("0")) or 1)
    prec = int_digits + scale
    if prec > 38:
        raise ValueError("needs more than 38 digits")
    return f"DECIMAL({18 if prec <= 18 else 38}, {scale})"


def load_column(
    con: Any,
    table: str,
    column: str,
    strings: tuple[str, ...],
    kind: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Create ``table`` with ``column`` as DOUBLE (``double``/``float``) or exact DECIMAL.

    Data travel through a registered pandas DataFrame: float64 values for DOUBLE, the
    decimal strings (cast by DuckDB's exact string-to-decimal conversion) for DECIMAL.
    ``extra`` adds more columns (e.g. ANOVA group labels) unchanged.
    """
    import numpy as np
    import pandas as pd

    if kind == "decimal":
        frame = pd.DataFrame({"v": list(strings), **(extra or {})})
        select = f"CAST(v AS {decimal_type(strings)}) AS {column}"
    else:
        frame = pd.DataFrame({"v": np.array([float(s) for s in strings]), **(extra or {})})
        select = f"CAST(v AS DOUBLE) AS {column}"
    others = "".join(f", {k}" for k in (extra or {}))
    con.register("_src", frame)
    con.execute(f"CREATE TABLE {table} AS SELECT {select}{others} FROM _src")
    con.unregister("_src")
