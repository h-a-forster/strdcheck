"""Part C: error growth with n on StRD-derived tiled data (DESIGN 4.7 and 6.4).

``Y`` is ``k = round(n / n0)`` concatenated copies of a univariate dataset's float64 values,
optionally permuted with a seeded permutation. References (mean, SD) are exact for the
float64 data (:func:`strdcheck.exact.tiled_reference`); they are *derived references*, not
NIST-certified values.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from strdcheck.manifest import list_datasets


def tiled_array(y0: np.ndarray, n: int, order: str, seed: int, dataset: str) -> np.ndarray:
    k = max(1, round(n / y0.size))
    y = np.tile(np.asarray(y0, dtype=np.float64), k)
    if order == "permuted":
        idx = [d.name for d in list_datasets("univariate")].index(dataset)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3, idx, int(n)]))
        y = rng.permutation(y)
    elif order != "tiled":
        raise ValueError(f"unknown order {order!r}")
    return y


def k_copies(n0: int, n: int) -> int:
    return max(1, round(n / n0))


def _numpy(y: np.ndarray) -> dict[str, float]:
    return {"mean": float(y.mean()), "sd": float(y.std(ddof=1))}


def _pandas(y: np.ndarray) -> dict[str, float]:
    import pandas as pd

    s = pd.Series(y)
    return {"mean": float(s.mean()), "sd": float(s.std())}


def _polars_eager(y: np.ndarray) -> dict[str, float]:
    import polars as pl

    s = pl.Series("y", y)
    return {"mean": float(s.mean()), "sd": float(s.std())}


def _polars_streaming(y: np.ndarray) -> dict[str, float]:
    import polars as pl

    m, s = (
        pl.LazyFrame({"y": y})
        .select(pl.col("y").mean(), pl.col("y").std().alias("s"))
        .collect(engine="streaming")
        .row(0)
    )
    return {"mean": float(m), "sd": float(s)}


def _duckdb(y: np.ndarray, threads: int) -> dict[str, float]:
    import duckdb
    import pandas as pd

    con = duckdb.connect(database=":memory:")
    try:
        con.execute(f"SET threads = {threads}")
        con.register("t", pd.DataFrame({"y": y}))
        m, s = con.execute("SELECT avg(y), stddev_samp(y) FROM t").fetchone()
    finally:
        con.close()
    return {"mean": float(m), "sd": float(s)}


def _torch(y: np.ndarray) -> dict[str, float]:
    import torch

    t = torch.from_numpy(y)
    return {"mean": float(t.mean()), "sd": float(t.std())}


TILED_ADAPTERS: dict[str, tuple[str, Callable[[np.ndarray], dict[str, float]]]] = {
    "numpy.f64": ("numpy", _numpy),
    "pandas.f64": ("pandas", _pandas),
    "polars.eager.f64": ("polars", _polars_eager),
    "polars.lazy-streaming.f64": ("polars", _polars_streaming),
    "duckdb.double.threads-1": ("duckdb", lambda y: _duckdb(y, 1)),
    "duckdb.double.threads-3": ("duckdb", lambda y: _duckdb(y, 3)),
    "torch.f64": ("torch", _torch),
}
"""Part C configurations: id -> (import name, function). polars uses the worker's thread
pool (``POLARS_MAX_THREADS = 3`` in the Part C worker); torch runs with one thread."""
