"""Exact references and representation ceilings (``docs/DESIGN.md`` sections 4.6, 4.7, 5.3).

Univariate, ANOVA and linear statistics are computed in exact rational arithmetic
(:class:`fractions.Fraction`) from one of three versions of the data:

- ``"dec"``: the decimal strings as printed by NIST (must reproduce the certified values),
- ``"f64"``: each value rounded to float64 (``Fraction(float(s))`` is exact),
- ``"f32"``: each value rounded to float64 and then to float32 (the path engines take).

Square roots are taken in :mod:`decimal` at 60 significant digits; all results are returned
as :class:`decimal.Decimal`. The *representation ceiling* of a quantity for a dtype is the
LRE of the exact statistic of the rounded data against the certified value: no engine
working on that dtype can be expected to beat it, so engine results are judged on their
*deficit* below the ceiling.

Nonlinear references use mpmath (imported lazily): Gauss-Newton refinement from the
certified parameters at 40 digits, on the decimal or float64-rounded data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Literal

import numpy as np

from strdcheck.lre import clip_lre
from strdcheck.parse import Certified, Dataset

Kind = Literal["dec", "f64", "f32"]
KINDS: tuple[Kind, ...] = ("dec", "f64", "f32")
PREC = 60


# --- conversions ---------------------------------------------------------------------------


def to_fractions(strings: Sequence[str], kind: Kind = "dec") -> list[Fraction]:
    """Exact rationals of decimal strings as given, or after rounding to float64/float32."""
    if kind == "dec":
        return [Fraction(s) for s in strings]
    if kind == "f64":
        return [Fraction(float(s)) for s in strings]
    if kind == "f32":
        a = np.array([float(s) for s in strings], dtype=np.float64).astype(np.float32)
        return [Fraction(float(v)) for v in a]
    raise ValueError(f"unknown kind {kind!r}")


def to_decimal(q: Fraction | int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = PREC
        q = Fraction(q)
        return Decimal(q.numerator) / Decimal(q.denominator)


def sqrt_fraction(q: Fraction) -> Decimal:
    """Square root of a non-negative rational to ``PREC`` significant digits."""
    if q < 0:
        raise ValueError("negative argument")
    with localcontext() as ctx:
        ctx.prec = PREC + 5
        r = (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
        ctx.prec = PREC
        return +r


def certified_decimal(c: Certified) -> Decimal:
    return Decimal(c.text)


def lre_decimal(value: Decimal, certified: Certified) -> tuple[float, float]:
    """``(raw, reported)`` LRE of a high-precision value, computed in decimal arithmetic."""
    c = certified_decimal(certified)
    with localcontext() as ctx:
        ctx.prec = PREC
        err = abs(value - c)
        if err == 0:
            raw = math.inf
        elif c == 0:
            raw = -float(err.log10())
        else:
            raw = -float((err / abs(c)).log10())
    return raw, clip_lre(raw, certified.cap)


# --- univariate (DESIGN 4.2) ------------------------------------------------------------------


def univariate(values: Sequence[Fraction]) -> dict[str, Decimal]:
    """Exact mean, sample SD (n - 1) and lag-1 autocorrelation r(1)."""
    n = len(values)
    if n < 2:
        raise ValueError("need at least two values")
    mean = sum(values, Fraction(0)) / n
    d = [v - mean for v in values]
    ss = sum((x * x for x in d), Fraction(0))
    out = {"mean": to_decimal(mean), "sd": sqrt_fraction(ss / (n - 1))}
    if ss != 0:
        num = sum((d[i] * d[i - 1] for i in range(1, n)), Fraction(0))
        out["r1"] = to_decimal(num / ss)
    return out


# --- one-way ANOVA (DESIGN 4.3) ---------------------------------------------------------------


def anova(groups: Sequence[str], values: Sequence[Fraction]) -> dict[str, Decimal]:
    """Exact one-way ANOVA quantities; groups are labels, in any order."""
    by: dict[str, list[Fraction]] = {}
    for g, v in zip(groups, values, strict=True):
        by.setdefault(g, []).append(v)
    k = len(by)
    n_tot = len(values)
    grand = sum(values, Fraction(0)) / n_tot
    ss_b = Fraction(0)
    ss_w = Fraction(0)
    for vals in by.values():
        m = sum(vals, Fraction(0)) / len(vals)
        ss_b += len(vals) * (m - grand) ** 2
        ss_w += sum(((v - m) ** 2 for v in vals), Fraction(0))
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (n_tot - k)
    out = {
        "ss_between": to_decimal(ss_b),
        "ms_between": to_decimal(ms_b),
        "ss_within": to_decimal(ss_w),
        "ms_within": to_decimal(ms_w),
        "resid_sd": sqrt_fraction(ms_w),
    }
    if ss_b + ss_w != 0:  # all values equal (possible after float32 rounding)
        out["r2"] = to_decimal(ss_b / (ss_b + ss_w))
    if ms_w != 0:
        out["f"] = to_decimal(ms_b / ms_w)
    return out


# --- linear least squares (DESIGN 4.4) ----------------------------------------------------------


def _invert(a: list[list[Fraction]]) -> list[list[Fraction]]:
    """Gauss-Jordan inverse of a non-singular rational matrix."""
    p = len(a)
    m = [row[:] + [Fraction(int(i == j)) for j in range(p)] for i, row in enumerate(a)]
    for col in range(p):
        piv = next((r for r in range(col, p) if m[r][col] != 0), None)
        if piv is None:
            raise ZeroDivisionError("singular matrix")
        m[col], m[piv] = m[piv], m[col]
        inv = 1 / m[col][col]
        m[col] = [v * inv for v in m[col]]
        for r in range(p):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [vr - f * vc for vr, vc in zip(m[r], m[col], strict=True)]
    return [row[p:] for row in m]


def design_columns(ds: Dataset, kind: Kind = "dec") -> list[list[Fraction]]:
    """Exact columns of X for a linear dataset (powers taken exactly of the rounded x)."""
    spec = ds.linear_spec()
    n = ds.n
    cols: list[list[Fraction]] = []
    if spec.intercept:
        cols.append([Fraction(1)] * n)
    if spec.degree is not None:
        x = to_fractions(ds.raw["x"], kind)
        for j in range(1, spec.degree + 1):
            cols.append([v**j for v in x])
    else:
        for c in spec.predictors:
            cols.append(to_fractions(ds.raw[c], kind))
    return cols


def linear(
    cols: Sequence[Sequence[Fraction]],
    y: Sequence[Fraction],
    names: Sequence[str],
    *,
    intercept: bool,
) -> dict[str, Decimal]:
    """Exact OLS coefficients, SEs, residual SD and R² (uncentred without intercept)."""
    p = len(cols)
    n = len(y)
    a = [
        [
            sum((ci * cj for ci, cj in zip(cols[i], cols[j], strict=True)), Fraction(0))
            for j in range(p)
        ]
        for i in range(p)
    ]
    rhs = [sum((ci * yi for ci, yi in zip(cols[i], y, strict=True)), Fraction(0)) for i in range(p)]
    inv = _invert(a)
    b = [sum((inv[i][j] * rhs[j] for j in range(p)), Fraction(0)) for i in range(p)]
    rss = Fraction(0)
    for r in range(n):
        fit = sum((b[j] * cols[j][r] for j in range(p)), Fraction(0))
        rss += (y[r] - fit) ** 2
    s2 = rss / (n - p)
    if intercept:
        ybar = sum(y, Fraction(0)) / n
        tss = sum(((v - ybar) ** 2 for v in y), Fraction(0))
    else:
        tss = sum((v * v for v in y), Fraction(0))
    out: dict[str, Decimal] = {}
    for j, name in enumerate(names):
        out[name] = to_decimal(b[j])
        out[f"se_{name}"] = sqrt_fraction(s2 * inv[j][j])
    out["resid_sd"] = sqrt_fraction(s2)
    out["r2"] = to_decimal(1 - rss / tss)
    return out


# --- dataset-level references and ceilings ---------------------------------------------------


def reference(ds: Dataset, kind: Kind = "dec") -> dict[str, Decimal]:
    """Exact statistics of ``ds`` (univariate, ANOVA, linear) for one data version."""
    if ds.suite == "univariate":
        return univariate(to_fractions(ds.raw["y"], kind))
    if ds.suite == "anova":
        g, v = ds.columns
        return anova(ds.raw[g], to_fractions(ds.raw[v], kind))
    if ds.suite == "linear":
        spec = ds.linear_spec()
        return linear(
            design_columns(ds, kind),
            to_fractions(ds.raw["y"], kind),
            spec.param_names(),
            intercept=spec.intercept,
        )
    raise ValueError(f"no exact rational reference for suite {ds.suite!r}; see nonlinear_refine")


@dataclass(frozen=True, slots=True)
class Ceiling:
    """Representation ceiling of one certified quantity."""

    name: str
    dec: float  # LRE of the exact statistic of the decimal data (V1; should be >= cap - 1)
    f64: float  # ceiling for float64 engines (reported LRE scale)
    f32: float  # ceiling for float32 engines


def ceilings(ds: Dataset) -> dict[str, Ceiling]:
    """Ceilings for every finite certified quantity of a univariate/ANOVA/linear dataset."""
    refs = {k: reference(ds, k) for k in KINDS}
    out: dict[str, Ceiling] = {}
    for name, cert in ds.certified.items():
        if not math.isfinite(cert.value):
            continue
        vals = []
        for k in KINDS:
            v = refs[k].get(name)
            vals.append(math.nan if v is None else lre_decimal(v, cert)[1])
        out[name] = Ceiling(name, *vals)
    return out


# --- tiled data (DESIGN 4.7) --------------------------------------------------------------------


def tiled_reference(y: np.ndarray, k: int) -> dict[str, Decimal]:
    """Exact mean and SD of ``k`` concatenated copies of the float64 values ``y``."""
    if k < 1:
        raise ValueError("k must be >= 1")
    vals = [Fraction(float(v)) for v in np.asarray(y, dtype=np.float64)]
    n0 = len(vals)
    mean = sum(vals, Fraction(0)) / n0
    ss = sum(((v - mean) ** 2 for v in vals), Fraction(0))
    return {"mean": to_decimal(mean), "sd": sqrt_fraction(k * ss / (k * n0 - 1))}


# --- nonlinear references (mpmath) ------------------------------------------------------------


@dataclass(frozen=True)
class NonlinearReference:
    values: dict[str, Decimal]  # b*, se_b*, rss, resid_sd
    converged: bool
    iterations: int


def _mp_data(ds: Dataset, kind: Kind):
    import mpmath

    from strdcheck.models import get_model

    model = get_model(ds.name)
    if kind == "dec":
        y = [mpmath.mpf(s) for s in ds.raw["y"]]
        if model.log_response:
            y = [mpmath.log(v) for v in y]
        xs = [[mpmath.mpf(s) for s in ds.raw[c]] for c in ds.columns[1:]]
    elif kind == "f64":
        y64 = ds.array("y")
        if model.log_response:
            y64 = np.log(y64)  # what a float64 engine sees
        y = [mpmath.mpf(float(v)) for v in y64]
        xs = [[mpmath.mpf(float(v)) for v in ds.array(c)] for c in ds.columns[1:]]
    else:
        raise ValueError("nonlinear references support 'dec' and 'f64'")
    return model, y, xs


def _mp_stats(model, b, y, xs, m):
    import mpmath

    n, p = len(y), len(b)
    rows = [tuple(c[i] for c in xs) for i in range(n)]
    r = [y[i] - model.f(b, rows[i], m) for i in range(n)]
    jac = [model.jac(b, rows[i], m) for i in range(n)]
    jtj = mpmath.matrix(p, p)
    jtr = mpmath.matrix(p, 1)
    for i in range(n):
        ji = jac[i]
        for a in range(p):
            jtr[a] += ji[a] * r[i]
            for c in range(a, p):
                jtj[a, c] += ji[a] * ji[c]
    for a in range(p):
        for c in range(a):
            jtj[a, c] = jtj[c, a]
    rss = mpmath.fsum(v * v for v in r)
    return rss, jtj, jtr


def _mp_to_decimal(v) -> Decimal:
    import mpmath

    return Decimal(mpmath.nstr(v, 45, strip_zeros=False, min_fixed=1, max_fixed=0))


def nonlinear_at(ds: Dataset, b: Sequence[str | float], kind: Kind = "dec", dps: int = 40):
    """RSS, residual SD and SEs (DESIGN 4.5) at the given parameters, in mpmath."""
    import mpmath

    from strdcheck.models import mp_backend

    with mpmath.workdps(dps):
        model, y, xs = _mp_data(ds, kind)
        m = mp_backend()
        bb = [mpmath.mpf(v) for v in b]
        rss, jtj, _ = _mp_stats(model, bb, y, xs, m)
        return _summary(ds, bb, rss, jtj)


def _summary(ds: Dataset, b, rss, jtj) -> dict[str, Decimal]:
    import mpmath

    n, p = ds.n, len(b)
    s2 = rss / (n - p)
    inv = mpmath.inverse(jtj)
    out: dict[str, Decimal] = {}
    for j in range(p):
        out[f"b{j + 1}"] = _mp_to_decimal(b[j])
        out[f"se_b{j + 1}"] = _mp_to_decimal(mpmath.sqrt(s2 * inv[j, j]))
    out["rss"] = _mp_to_decimal(rss)
    out["resid_sd"] = _mp_to_decimal(mpmath.sqrt(s2))
    return out


def nonlinear_refine(
    ds: Dataset, kind: Kind = "dec", *, dps: int = 40, max_iter: int = 100, tol: float = 1e-20
) -> NonlinearReference:
    """Gauss-Newton refinement from the certified parameters, in mpmath at ``dps`` digits.

    Gauss-Newton converges only linearly on problems with large residuals (MGH09 gains
    about 0.2 digits per iteration), hence the iteration limit of 100 and a step tolerance
    (relative, per parameter) of ``1e-20``, which is far below the 11-15 digits compared.
    """
    import mpmath

    from strdcheck.models import mp_backend

    names = ds.param_names()
    with mpmath.workdps(dps):
        model, y, xs = _mp_data(ds, kind)
        m = mp_backend()
        b = [mpmath.mpf(ds.certified[nm].text) for nm in names]
        tol_mp = mpmath.mpf(tol)
        converged = False
        it = 0
        for it in range(1, max_iter + 1):  # noqa: B007 - reported below
            _, jtj, jtr = _mp_stats(model, b, y, xs, m)
            step = mpmath.lu_solve(jtj, jtr)
            b = [b[j] + step[j] for j in range(len(b))]
            rel = max(abs(step[j]) / max(abs(b[j]), tol_mp) for j in range(len(b)))
            if rel < tol_mp:
                converged = True
                break
        rss, jtj, _ = _mp_stats(model, b, y, xs, m)
        return NonlinearReference(_summary(ds, b, rss, jtj), converged, it)


def nonlinear_ceilings(ds: Dataset, *, dps: int = 40) -> dict[str, Ceiling]:
    """Float64 ceilings for a nonlinear problem (``f32`` is not computed: ``nan``)."""
    dec = nonlinear_refine(ds, "dec", dps=dps)
    f64 = nonlinear_refine(ds, "f64", dps=dps)
    out: dict[str, Ceiling] = {}
    for name, cert in ds.certified.items():
        d = lre_decimal(dec.values[name], cert)[1] if dec.converged else math.nan
        f = lre_decimal(f64.values[name], cert)[1] if f64.converged else math.nan
        out[name] = Ceiling(name, d, f, math.nan)
    return out
