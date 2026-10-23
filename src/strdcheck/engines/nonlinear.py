"""Nonlinear least squares adapters (DESIGN 4.5).

Configurations: ``scipy.optimize.least_squares`` with ``method`` in {trf, dogbox, lm},
analytic or 2-point Jacobian, default or tight tolerances (``ftol = xtol = gtol = 1e-15``),
and ``scipy.optimize.curve_fit`` with its default or the analytic Jacobian. No bounds, default
``max_nfev``. Each call takes ``start=`` (the starting vector; NIST Start 1 by default).

Returned values: ``b1..bp``, ``se_b*``, ``rss`` and ``resid_sd``. ``status["success"]`` is
the solver's own verdict (``least_squares``: ``status > 0``; ``curve_fit``: ``ier`` in 1-4;
``RuntimeError`` from ``curve_fit`` is a reported failure). The outcome class (``ok``,
``silent``, ``flagged_ok``, ``flagged``, ``error``) is assigned from that verdict and the
minimum parameter LRE by :func:`classify`, so the threshold ``tau`` can be varied in the
analysis without re-running fits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from strdcheck.engines.base import Adapter
from strdcheck.models import NUMPY, get_model, predictors, response
from strdcheck.parse import Dataset
from strdcheck.results import EngineResult

SUITE = "nonlinear"
TIGHT = 1e-15
OUTCOMES = ("ok", "silent", "flagged_ok", "flagged", "error")


def classify(success: bool | None, min_lre: float, *, tau: float = 4.0, error: bool = False) -> str:
    """Outcome class of one fit (DESIGN 4.5)."""
    if error or success is None:
        return "error"
    good = math.isfinite(min_lre) and min_lre >= tau
    if success:
        return "ok" if good else "silent"
    return "flagged_ok" if good else "flagged"


def _problem(ds: Dataset):
    model = get_model(ds.name)
    X = predictors(ds)
    y = response(model, ds.array("y"))
    return model, X, y


def _jac_matrix(model, b, X, n: int) -> np.ndarray:
    cols = [np.broadcast_to(np.asarray(c, dtype=np.float64), (n,)) for c in model.jac(b, X, NUMPY)]
    return np.column_stack(cols)


def _summary(ds: Dataset, b: np.ndarray, rss: float, jac: np.ndarray | None) -> dict[str, float]:
    n, p = ds.n, b.size
    s2 = rss / (n - p)
    vals = {f"b{j + 1}": float(b[j]) for j in range(p)}
    vals["rss"] = float(rss)
    vals["resid_sd"] = math.sqrt(s2) if s2 >= 0 else math.nan
    se = np.full(p, np.nan)
    if jac is not None and np.all(np.isfinite(jac)):
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
        thresh = np.finfo(float).eps * max(n, p) * (s[0] if s.size else 0.0)
        keep = s > thresh
        if keep.all():
            cov = (vt.T / s**2) @ vt * s2
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
    vals.update({f"se_b{j + 1}": float(se[j]) for j in range(p)})
    return vals


def _nan_values(p: int) -> dict[str, float]:
    vals = {f"b{j + 1}": math.nan for j in range(p)}
    vals.update({f"se_b{j + 1}": math.nan for j in range(p)})
    vals.update({"rss": math.nan, "resid_sd": math.nan})
    return vals


def scipy_least_squares(
    ds: Dataset,
    method: str = "trf",
    jac: str = "analytic",
    tol: str = "default",
    start: Sequence[float] | None = None,
) -> EngineResult:
    from scipy.optimize import least_squares

    model, X, y = _problem(ds)
    n = ds.n
    x0 = np.asarray(start if start is not None else ds.starts[0], dtype=np.float64)

    def fun(b):
        return model(b, X) - y

    def jfun(b):
        return _jac_matrix(model, b, X, n)

    kw = {"ftol": TIGHT, "xtol": TIGHT, "gtol": TIGHT} if tol == "tight" else {}
    with np.errstate(all="ignore"):
        res = least_squares(
            fun, x0, jac=jfun if jac == "analytic" else "2-point", method=method, **kw
        )
        rss = float(np.dot(res.fun, res.fun))
        vals = _summary(ds, res.x, rss, res.jac)
    status = {
        "success": bool(res.status > 0),
        "status": int(res.status),
        "message": str(res.message)[:200],
        "nfev": int(res.nfev),
        "njev": None if res.njev is None else int(res.njev),
    }
    return EngineResult(vals, status=status)


def scipy_curve_fit(
    ds: Dataset, jac: str = "default", start: Sequence[float] | None = None
) -> EngineResult:
    from scipy.optimize import curve_fit

    model, X, y = _problem(ds)
    n, p = ds.n, ds.n_params
    xdata = np.vstack(X) if len(X) > 1 else X[0]
    x0 = np.asarray(start if start is not None else ds.starts[0], dtype=np.float64)

    def cols(xd):
        return tuple(xd) if np.ndim(xd) == 2 else (xd,)

    def f(xd, *b):
        return model(b, cols(xd))

    def jf(xd, *b):
        return _jac_matrix(model, b, cols(xd), n)

    with np.errstate(all="ignore"):
        try:
            popt, _pcov, info, mesg, ier = curve_fit(
                f, xdata, y, p0=x0, jac=jf if jac == "analytic" else None, full_output=True
            )
        except RuntimeError as exc:  # "Optimal parameters not found": a reported failure
            return EngineResult(
                _nan_values(p), status={"success": False, "message": str(exc)[:200]}
            )
        r = f(xdata, *popt) - y
        rss = float(np.dot(r, r))
        J = _jac_matrix(model, popt, X, n)
        vals = _summary(ds, np.asarray(popt), rss, J)
        # SEs as curve_fit reports them (pcov scaled by RSS / (n - p)).
        se = np.sqrt(np.diag(_pcov)) if np.all(np.isfinite(_pcov)) else np.full(p, np.nan)
        vals.update({f"se_b{j + 1}": float(se[j]) for j in range(p)})
    status = {
        "success": ier in (1, 2, 3, 4),
        "status": int(ier),
        "message": str(mesg)[:200],
        "nfev": int(info.get("nfev", -1)),
    }
    return EngineResult(vals, status=status)


ADAPTERS: list[Adapter] = []
for _m in ("trf", "dogbox", "lm"):
    for _j in ("analytic", "2point"):
        for _t in ("default", "tight"):
            ADAPTERS.append(
                Adapter(
                    f"scipy.least_squares.{_m}.jac-{_j}.tol-{_t}",
                    SUITE,
                    "scipy",
                    "f64",
                    scipy_least_squares,
                    kwargs={
                        "method": _m,
                        "jac": "analytic" if _j == "analytic" else "2-point",
                        "tol": _t,
                    },
                )
            )
ADAPTERS += [
    Adapter("scipy.curve_fit.default", SUITE, "scipy", "f64", scipy_curve_fit),
    Adapter(
        "scipy.curve_fit.jac-analytic",
        SUITE,
        "scipy",
        "f64",
        scipy_curve_fit,
        kwargs={"jac": "analytic"},
    ),
]
