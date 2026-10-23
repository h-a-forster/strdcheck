"""One-way ANOVA adapters (DESIGN 4.3).

Group-by adapters (pandas, polars, DuckDB) take per-group ``count``, ``mean`` and ``var``
(ddof 1) and the grand mean from the engine and combine them with :func:`combine`, the same
float64 Python arithmetic for every engine, so differences come from the engine aggregates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from strdcheck.engines.base import Adapter, duckdb_connect, load_column
from strdcheck.parse import Dataset

SUITE = "anova"


def _gy(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    g, v = ds.columns
    return np.array([int(s) for s in ds.raw[g]]), ds.array(v)


def combine(
    counts: Sequence[float], means: Sequence[float], variances: Sequence[float], grand: float
) -> dict[str, float]:
    """ANOVA quantities from per-group count, mean, sample variance and the grand mean."""
    k = len(counts)
    n_tot = float(sum(counts))
    ss_b = math.fsum(n * (m - grand) ** 2 for n, m in zip(counts, means, strict=True))
    ss_w = math.fsum((n - 1) * v for n, v in zip(counts, variances, strict=True))
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (n_tot - k)
    return {
        "ss_between": ss_b,
        "ms_between": ms_b,
        "f": ms_b / ms_w if ms_w else math.nan,
        "ss_within": ss_w,
        "ms_within": ms_w,
        "r2": ss_b / (ss_b + ss_w) if ss_b + ss_w else math.nan,
        # A negative within-group variance (possible for the one-pass control) is a numerical
        # failure of the inputs, reported as nan rather than raised.
        "resid_sd": math.sqrt(ms_w) if ms_w >= 0 else math.nan,
    }


def numpy_twopass(ds: Dataset) -> dict[str, float]:
    """Reference implementation of DESIGN 4.3 in numpy float64 (two-pass per group)."""
    g, y = _gy(ds)
    levels = np.unique(g)
    counts, means, variances = [], [], []
    for lv in levels:
        yy = y[g == lv]
        counts.append(yy.size)
        means.append(yy.mean())
        variances.append(yy.var(ddof=1))
    return combine(counts, means, variances, y.mean())


def onepass(ds: Dataset) -> dict[str, float]:
    """Negative control: one-pass sums of squares per group."""
    g, y = _gy(ds)
    counts, means, variances = [], [], []
    for lv in np.unique(g):
        yy = y[g == lv]
        n = yy.size
        m = yy.sum() / n
        counts.append(n)
        means.append(m)
        variances.append((np.sum(yy * yy) - n * m * m) / (n - 1))
    return combine(counts, means, variances, y.sum() / y.size)


def pandas_groupby(ds: Dataset) -> dict[str, float]:
    import pandas as pd

    g, y = _gy(ds)
    df = pd.DataFrame({"g": g, "y": y})
    agg = df.groupby("g")["y"].agg(["count", "mean", "var"])
    return combine(agg["count"].tolist(), agg["mean"].tolist(), agg["var"].tolist(), df["y"].mean())


def polars_groupby(ds: Dataset) -> dict[str, float]:
    import polars as pl

    g, y = _gy(ds)
    df = pl.DataFrame({"g": g, "y": y})
    agg = (
        df.group_by("g")
        .agg(pl.len().alias("n"), pl.col("y").mean().alias("m"), pl.col("y").var().alias("v"))
        .sort("g")
    )
    return combine(agg["n"].to_list(), agg["m"].to_list(), agg["v"].to_list(), df["y"].mean())


def duckdb_groupby(ds: Dataset, kind: str = "double") -> dict[str, float]:
    gcol, vcol = ds.columns
    con = duckdb_connect()
    try:
        load_column(con, "t", "y", ds.raw[vcol], kind, extra={"g": [int(s) for s in ds.raw[gcol]]})
        rows = con.execute(
            "SELECT g, count(*), avg(y), var_samp(y) FROM t GROUP BY g ORDER BY g"
        ).fetchall()
        (grand,) = con.execute("SELECT avg(y) FROM t").fetchone()
    finally:
        con.close()
    return combine(
        [float(r[1]) for r in rows],
        [float(r[2]) for r in rows],
        [float(r[3]) for r in rows],
        float(grand),
    )


def statsmodels_anova(ds: Dataset) -> dict[str, float]:
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    g, y = _gy(ds)
    fit = smf.ols("y ~ C(g)", data=pd.DataFrame({"g": g, "y": y})).fit()
    tab = anova_lm(fit, typ=1)
    b, w = tab.loc["C(g)"], tab.loc["Residual"]
    return {
        "ss_between": b["sum_sq"],
        "ms_between": b["mean_sq"],
        "f": b["F"],
        "ss_within": w["sum_sq"],
        "ms_within": w["mean_sq"],
        "r2": fit.rsquared,
        "resid_sd": math.sqrt(fit.mse_resid),
    }


def scipy_f_oneway(ds: Dataset) -> dict[str, float]:
    from scipy import stats

    g, y = _gy(ds)
    return {"f": stats.f_oneway(*[y[g == lv] for lv in np.unique(g)]).statistic}


def exact_anova(ds: Dataset) -> dict[str, float]:
    from strdcheck import exact

    return {k: float(v) for k, v in exact.reference(ds, "dec").items()}


ADAPTERS = [
    Adapter(
        "statsmodels.anova_lm",
        SUITE,
        "statsmodels+pandas",
        "f64",
        statsmodels_anova,
        note='ols("y ~ C(g)") + anova_lm(typ=1)',
    ),
    Adapter("scipy.f_oneway", SUITE, "scipy", "f64", scipy_f_oneway, note="F only"),
    Adapter("pandas.groupby", SUITE, "pandas", "f64", pandas_groupby),
    Adapter("polars.groupby", SUITE, "polars", "f64", polars_groupby),
    Adapter("duckdb.groupby.double", SUITE, "duckdb+pandas", "f64", duckdb_groupby),
    Adapter(
        "duckdb.groupby.decimal",
        SUITE,
        "duckdb+pandas",
        "dec",
        duckdb_groupby,
        kwargs={"kind": "decimal"},
    ),
    Adapter(
        "numpy.twopass",
        SUITE,
        "numpy",
        "f64",
        numpy_twopass,
        note="reference implementation of DESIGN 4.3",
    ),
    Adapter("control.onepass.f64", SUITE, "numpy", "f64", onepass, role="control"),
    Adapter("control.exact", SUITE, "fractions", "dec", exact_anova, role="control"),
]
