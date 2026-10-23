"""Univariate adapters: mean, sample SD (n - 1) and lag-1 autocorrelation (DESIGN 4.2).

Engine conventions respected here: numpy ``std`` needs ``ddof=1``; pandas, polars and torch
default to ``n - 1``; DuckDB ``stddev_samp``; statsmodels ``DescrStatsW`` needs ``ddof=1``.
r(1) is reported only where the engine computes NIST's definition (statsmodels ``acf`` with
``adjusted=False``) or by the numpy reference formula; pandas ``Series.autocorr`` is a
different statistic (Pearson correlation of the lagged series) and is not used.
"""

from __future__ import annotations

import math

import numpy as np

from strdcheck.engines.base import Adapter, duckdb_connect, load_column
from strdcheck.parse import Dataset

SUITE = "univariate"


def _y(ds: Dataset, dtype=np.float64) -> np.ndarray:
    return ds.array("y", dtype)


def _r1_twopass(y: np.ndarray) -> float:
    d = y - y.mean()
    return float(np.dot(d[1:], d[:-1]) / np.dot(d, d))


def numpy_stats(ds: Dataset, dtype: str = "f64") -> dict[str, float]:
    y = _y(ds, np.float32 if dtype == "f32" else np.float64)
    return {"mean": y.mean(), "sd": y.std(ddof=1), "r1": _r1_twopass(y)}


def pandas_stats(ds: Dataset, dtype: str = "f64") -> dict[str, float]:
    import pandas as pd

    s = pd.Series(_y(ds, np.float32 if dtype == "f32" else np.float64))
    return {"mean": s.mean(), "sd": s.std()}


def polars_stats(ds: Dataset, dtype: str = "f64", streaming: bool = False) -> dict[str, float]:
    import polars as pl

    pdt = pl.Float32 if dtype == "f32" else pl.Float64
    if streaming:
        lf = pl.LazyFrame({"y": _y(ds)}, schema={"y": pl.Float64}).select(
            pl.col("y").cast(pdt).mean().alias("m"), pl.col("y").cast(pdt).std().alias("s")
        )
        m, s = lf.collect(engine="streaming").row(0)
    else:
        ser = pl.Series("y", _y(ds)).cast(pdt)
        m, s = ser.mean(), ser.std()
    return {"mean": m, "sd": s}


def duckdb_stats(ds: Dataset, kind: str = "double") -> dict[str, float]:
    con = duckdb_connect()
    try:
        load_column(con, "t", "y", ds.raw["y"], kind)
        col = "CAST(y AS FLOAT)" if kind == "float" else "y"
        m, s = con.execute(f"SELECT avg({col}), stddev_samp({col}) FROM t").fetchone()
    finally:
        con.close()
    return {"mean": float(m), "sd": float(s)}


def scipy_stats(ds: Dataset) -> dict[str, float]:
    from scipy import stats

    d = stats.describe(_y(ds))  # variance uses ddof=1
    return {"mean": d.mean, "sd": math.sqrt(d.variance)}


def statsmodels_stats(ds: Dataset) -> dict[str, float]:
    from statsmodels.stats.weightstats import DescrStatsW
    from statsmodels.tsa.stattools import acf

    y = _y(ds)
    d = DescrStatsW(y, ddof=1)
    return {"mean": d.mean, "sd": d.std, "r1": acf(y, nlags=1, adjusted=False, fft=False)[1]}


def torch_stats(ds: Dataset, dtype: str = "f64") -> dict[str, float]:
    import torch

    t = torch.tensor(_y(ds), dtype=torch.float32 if dtype == "f32" else torch.float64)
    return {"mean": t.mean().item(), "sd": t.std().item()}


def stdlib_stats(ds: Dataset) -> dict[str, float]:
    import statistics

    y = [float(v) for v in _y(ds)]
    return {"mean": statistics.mean(y), "sd": statistics.stdev(y)}


def onepass_stats(ds: Dataset) -> dict[str, float]:
    """Negative control: textbook one-pass formula ``(sum y^2 - n ybar^2) / (n - 1)``."""
    y = _y(ds)
    n = y.size
    m = y.sum() / n
    var = (np.sum(y * y) - n * m * m) / (n - 1)
    return {"mean": m, "sd": math.sqrt(var) if var >= 0 else math.nan}


def exact_stats(ds: Dataset) -> dict[str, float]:
    """Positive control: exact rational arithmetic on the decimal strings."""
    from strdcheck import exact

    return {k: float(v) for k, v in exact.univariate(exact.to_fractions(ds.raw["y"])).items()}


ADAPTERS = [
    Adapter("numpy.f64", SUITE, "numpy", "f64", numpy_stats, note="mean, std(ddof=1), r1 two-pass"),
    Adapter("numpy.f32", SUITE, "numpy", "f32", numpy_stats, kwargs={"dtype": "f32"}),
    Adapter("pandas.f64", SUITE, "pandas", "f64", pandas_stats),
    Adapter("pandas.f32", SUITE, "pandas", "f32", pandas_stats, kwargs={"dtype": "f32"}),
    Adapter("polars.eager.f64", SUITE, "polars", "f64", polars_stats),
    Adapter("polars.eager.f32", SUITE, "polars", "f32", polars_stats, kwargs={"dtype": "f32"}),
    Adapter(
        "polars.lazy-streaming.f64",
        SUITE,
        "polars",
        "f64",
        polars_stats,
        kwargs={"streaming": True},
    ),
    Adapter("duckdb.double", SUITE, "duckdb+pandas", "f64", duckdb_stats),
    Adapter("duckdb.float", SUITE, "duckdb+pandas", "f32", duckdb_stats, kwargs={"kind": "float"}),
    Adapter(
        "duckdb.decimal",
        SUITE,
        "duckdb+pandas",
        "dec",
        duckdb_stats,
        kwargs={"kind": "decimal"},
        note="decimal strings loaded as DECIMAL(p, s)",
    ),
    Adapter("scipy.describe.f64", SUITE, "scipy", "f64", scipy_stats),
    Adapter(
        "statsmodels.f64",
        SUITE,
        "statsmodels",
        "f64",
        statsmodels_stats,
        note="DescrStatsW(ddof=1), acf(adjusted=False, fft=False)",
    ),
    Adapter("torch.f64", SUITE, "torch", "f64", torch_stats),
    Adapter("torch.f32", SUITE, "torch", "f32", torch_stats, kwargs={"dtype": "f32"}),
    Adapter(
        "stdlib.statistics",
        SUITE,
        "statistics",
        "f64",
        stdlib_stats,
        note="statistics.mean/stdev on the float64 values",
    ),
    Adapter(
        "control.onepass.f64",
        SUITE,
        "numpy",
        "f64",
        onepass_stats,
        role="control",
        note="negative control: one-pass sum of squares",
    ),
    Adapter(
        "control.exact",
        SUITE,
        "fractions",
        "dec",
        exact_stats,
        role="control",
        note="positive control: exact rationals",
    ),
]
