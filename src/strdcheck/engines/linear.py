"""Linear least squares adapters (DESIGN 4.4).

The design matrix follows the NIST model line: an intercept column unless the model has none
and raw powers ``x^j`` for polynomial models, computed in the engine's dtype. Engines that
return only coefficients (numpy/scipy/torch ``lstsq``, the normal-equations control,
``Polynomial.fit``) get SEs, residual SD and R² from :func:`derived_stats`, one shared
float64 routine; those names are marked ``derived``. All engine defaults are kept (no
``rcond`` tuning): a reported rank below ``p`` is recorded in ``status["rank"]`` and is the
documented truncation by the default ``rcond``, not a defect.
"""

from __future__ import annotations

import math

import numpy as np

from strdcheck.engines.base import Adapter, duckdb_connect, load_column
from strdcheck.parse import Dataset
from strdcheck.results import EngineResult

SUITE = "linear"


def design(ds: Dataset, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """``(X, y)`` in ``dtype`` (powers computed in that dtype)."""
    spec = ds.linear_spec()
    y = ds.array("y", dtype)
    cols: list[np.ndarray] = []
    if spec.intercept:
        cols.append(np.ones(ds.n, dtype=dtype))
    if spec.degree is not None:
        x = ds.array("x", dtype)
        p = np.ones(ds.n, dtype=dtype)
        for _ in range(spec.degree):
            p = p * x
            cols.append(p)
    else:
        cols += [ds.array(c, dtype) for c in spec.predictors]
    return np.column_stack(cols), y


def derived_stats(ds: Dataset, b: np.ndarray) -> dict[str, float]:
    """SEs, residual SD and R² for coefficients ``b`` (shared float64 routine, QR of X)."""
    spec = ds.linear_spec()
    X, y = design(ds)
    b = np.asarray(b, dtype=np.float64)
    n, p = X.shape
    r = y - X @ b
    rss = float(r @ r)
    s2 = rss / (n - p)
    R = np.linalg.qr(X, mode="r")
    try:
        rinv = np.linalg.solve(R, np.eye(p))
        diag = np.sum(rinv * rinv, axis=1)
    except np.linalg.LinAlgError:
        diag = np.full(p, np.nan)
    tss = float(np.sum((y - y.mean()) ** 2)) if spec.intercept else float(y @ y)
    out = {
        f"se_{nm}": math.sqrt(s2 * d) if d >= 0 else math.nan
        for nm, d in zip(spec.param_names(), diag, strict=True)
    }
    out["resid_sd"] = math.sqrt(s2)
    out["r2"] = 1 - rss / tss
    return out


def _coef_result(ds: Dataset, b: np.ndarray, status: dict | None = None) -> EngineResult:
    names = ds.linear_spec().param_names()
    vals = {nm: float(v) for nm, v in zip(names, np.asarray(b, dtype=np.float64), strict=True)}
    d = derived_stats(ds, np.asarray(b, dtype=np.float64))
    vals.update(d)
    return EngineResult(vals, derived=set(d), status=status or {})


def _full_result(ds: Dataset, b, se, resid_sd, r2, status=None) -> EngineResult:
    names = ds.linear_spec().param_names()
    vals = {nm: float(v) for nm, v in zip(names, b, strict=True)}
    vals.update({f"se_{nm}": float(v) for nm, v in zip(names, se, strict=True)})
    vals["resid_sd"] = float(resid_sd)
    vals["r2"] = float(r2)
    return EngineResult(vals, status=status or {})


# --- numpy -------------------------------------------------------------------------------------


def numpy_lstsq(ds: Dataset, dtype: str = "f64") -> EngineResult:
    X, y = design(ds, np.float32 if dtype == "f32" else np.float64)
    b, _, rank, _ = np.linalg.lstsq(X, y)
    return _coef_result(ds, b, {"rank": int(rank), "p": X.shape[1]})


def numpy_polyfit(ds: Dataset) -> EngineResult:
    spec = ds.linear_spec()
    x, y = ds.array("x"), ds.array("y")
    c, cov = np.polyfit(x, y, spec.degree, cov=True)  # cov scaled by RSS / (n - p)
    b = c[::-1]
    se = np.sqrt(np.diag(cov))[::-1]
    res = _coef_result(ds, b)
    for nm, v in zip(spec.param_names(), se, strict=True):
        res.values[f"se_{nm}"] = float(v)
        res.derived.discard(f"se_{nm}")
    return res


def numpy_polynomial_fit(ds: Dataset) -> EngineResult:
    from numpy.polynomial import Polynomial

    spec = ds.linear_spec()
    p = Polynomial.fit(ds.array("x"), ds.array("y"), spec.degree).convert()
    coef = np.zeros(spec.degree + 1)
    coef[: p.coef.size] = p.coef  # convert() may drop trailing zero coefficients
    return _coef_result(ds, coef)


def normal_equations(ds: Dataset) -> EngineResult:
    """Negative control: solve ``X'X b = X'y`` in float64."""
    X, y = design(ds)
    return _coef_result(ds, np.linalg.solve(X.T @ X, X.T @ y))


# --- scipy ---------------------------------------------------------------------------------------


def scipy_lstsq(ds: Dataset, driver: str = "gelsd") -> EngineResult:
    from scipy import linalg

    X, y = design(ds)
    b, _, rank, _ = linalg.lstsq(X, y, lapack_driver=driver)
    return _coef_result(ds, b, {"rank": int(rank), "p": X.shape[1]})


def scipy_linregress(ds: Dataset) -> EngineResult:
    from scipy import stats

    x, y = ds.array("x"), ds.array("y")
    r = stats.linregress(x, y)
    resid = y - (r.intercept + r.slope * x)
    return _full_result(
        ds,
        [r.intercept, r.slope],
        [r.intercept_stderr, r.stderr],
        math.sqrt(float(resid @ resid) / (x.size - 2)),
        r.rvalue**2,
        {"derived_note": "resid_sd from linregress coefficients"},
    )


# --- statsmodels -------------------------------------------------------------------------------


def statsmodels_ols(ds: Dataset, method: str = "pinv") -> EngineResult:
    import statsmodels.api as sm

    X, y = design(ds)
    fit = sm.OLS(y, X, hasconst=ds.linear_spec().intercept).fit(method=method)
    status = {"rank": int(fit.model.rank), "p": X.shape[1]}
    return _full_result(ds, fit.params, fit.bse, math.sqrt(fit.scale), fit.rsquared, status)


# --- torch ---------------------------------------------------------------------------------------


def torch_lstsq(ds: Dataset, driver: str = "gelsy", dtype: str = "f64") -> EngineResult:
    import torch

    tdt = torch.float32 if dtype == "f32" else torch.float64
    X, y = design(ds, np.float32 if dtype == "f32" else np.float64)
    out = torch.linalg.lstsq(
        torch.from_numpy(X).to(tdt), torch.from_numpy(y).to(tdt)[:, None], driver=driver
    )
    b = out.solution[:, 0].double().numpy()
    rank = int(out.rank) if out.rank.numel() else None
    return _coef_result(ds, b, {"rank": rank, "p": X.shape[1]})


# --- simple regression in columnar engines (Norris) ---------------------------------------------


def duckdb_regr(ds: Dataset) -> EngineResult:
    con = duckdb_connect()
    try:
        load_column(con, "t", "y", ds.raw["y"], "double", extra={"x": ds.array("x")})
        slope, icpt, r2 = con.execute(
            "SELECT regr_slope(y, x), regr_intercept(y, x), regr_r2(y, x) FROM t"
        ).fetchone()
    finally:
        con.close()
    res = _coef_result(ds, np.array([icpt, slope], dtype=np.float64))
    res.values["r2"] = float(r2)
    res.derived.discard("r2")
    return res


def _cov_slope(ds: Dataset, cxy: float, vx: float, my: float, mx: float) -> EngineResult:
    slope = cxy / vx
    return _coef_result(ds, np.array([my - slope * mx, slope]))


def polars_cov(ds: Dataset) -> EngineResult:
    import polars as pl

    df = pl.DataFrame({"x": ds.array("x"), "y": ds.array("y")})
    row = df.select(
        pl.cov("x", "y").alias("c"),
        pl.col("x").var().alias("v"),
        pl.col("y").mean().alias("my"),
        pl.col("x").mean().alias("mx"),
    ).row(0)
    return _cov_slope(ds, *row)


def pandas_cov(ds: Dataset) -> EngineResult:
    import pandas as pd

    df = pd.DataFrame({"x": ds.array("x"), "y": ds.array("y")})
    return _cov_slope(ds, df["x"].cov(df["y"]), df["x"].var(), df["y"].mean(), df["x"].mean())


def _poly(ds: Dataset) -> bool:
    spec = ds.linear_spec()
    return spec.degree is not None and spec.intercept


def _simple(ds: Dataset) -> bool:
    spec = ds.linear_spec()
    return spec.degree == 1 and spec.intercept


ADAPTERS = [
    Adapter("numpy.lstsq", SUITE, "numpy", "f64", numpy_lstsq, note="default rcond"),
    Adapter("numpy.lstsq.f32", SUITE, "numpy", "f32", numpy_lstsq, kwargs={"dtype": "f32"}),
    Adapter(
        "numpy.polyfit",
        SUITE,
        "numpy",
        "f64",
        numpy_polyfit,
        applies=_poly,
        note="cov=True (scaled by RSS/(n-p))",
    ),
    Adapter(
        "numpy.Polynomial.fit",
        SUITE,
        "numpy",
        "f64",
        numpy_polynomial_fit,
        applies=_poly,
        note="scaled domain, convert() to the power basis",
    ),
    Adapter("scipy.lstsq.gelsd", SUITE, "scipy", "f64", scipy_lstsq),
    Adapter("scipy.lstsq.gelsy", SUITE, "scipy", "f64", scipy_lstsq, kwargs={"driver": "gelsy"}),
    Adapter("scipy.lstsq.gelss", SUITE, "scipy", "f64", scipy_lstsq, kwargs={"driver": "gelss"}),
    Adapter("scipy.linregress", SUITE, "scipy", "f64", scipy_linregress, applies=_simple),
    Adapter("statsmodels.ols.pinv", SUITE, "statsmodels", "f64", statsmodels_ols),
    Adapter(
        "statsmodels.ols.qr", SUITE, "statsmodels", "f64", statsmodels_ols, kwargs={"method": "qr"}
    ),
    Adapter("torch.lstsq.gelsy.f64", SUITE, "torch", "f64", torch_lstsq),
    Adapter(
        "torch.lstsq.gelsd.f64", SUITE, "torch", "f64", torch_lstsq, kwargs={"driver": "gelsd"}
    ),
    Adapter(
        "torch.lstsq.gelss.f64", SUITE, "torch", "f64", torch_lstsq, kwargs={"driver": "gelss"}
    ),
    Adapter("torch.lstsq.gelsy.f32", SUITE, "torch", "f32", torch_lstsq, kwargs={"dtype": "f32"}),
    Adapter(
        "duckdb.regr",
        SUITE,
        "duckdb+pandas",
        "f64",
        duckdb_regr,
        applies=_simple,
        note="regr_slope, regr_intercept, regr_r2",
    ),
    Adapter(
        "polars.cov",
        SUITE,
        "polars",
        "f64",
        polars_cov,
        applies=_simple,
        note="slope = cov(x, y) / var(x)",
    ),
    Adapter(
        "pandas.cov",
        SUITE,
        "pandas",
        "f64",
        pandas_cov,
        applies=_simple,
        note="slope = cov(x, y) / var(x)",
    ),
    Adapter(
        "control.normal-equations.f64",
        SUITE,
        "numpy",
        "f64",
        normal_equations,
        role="control",
        note="negative control: solve(X'X, X'y)",
    ),
]
