"""The 27 NIST StRD nonlinear regression models with analytic Jacobians.

Each model is written once against a small math backend so the same code runs on numpy
arrays (float64, used by the engine adapters) and on mpmath scalars (used for the
multiple-precision references in :mod:`strdcheck.exact`). Model formulas are transcribed
from the model lines of the NIST files; ``tests/test_models.py`` checks each formula against
NIST's certified residual sum of squares and each Jacobian against mpmath numerical
derivatives.

Nelson's model is ``log(y) = b1 - b2*x1*exp(-b3*x2)``: :func:`response` returns ``log(y)``
for it, as in NIST's certification; every other model fits ``y`` directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

Num = Any  # numpy array or mpmath mpf
Cols = Sequence[Num]  # predictor columns (x,) or (x1, x2)

NUMPY = SimpleNamespace(exp=np.exp, log=np.log, sin=np.sin, cos=np.cos, atan=np.arctan, pi=np.pi)


def mp_backend() -> SimpleNamespace:
    """mpmath backend (imported lazily; pi follows the current mpmath precision)."""
    import mpmath

    return SimpleNamespace(
        exp=mpmath.exp,
        log=mpmath.log,
        sin=mpmath.sin,
        cos=mpmath.cos,
        atan=mpmath.atan,
        pi=mpmath.pi,
    )


ModelF = Callable[[Sequence[Num], Cols, SimpleNamespace], Num]
ModelJ = Callable[[Sequence[Num], Cols, SimpleNamespace], list[Num]]


@dataclass(frozen=True)
class Model:
    """A nonlinear model ``f(b, x)`` with its Jacobian ``[df/db_j]``."""

    name: str
    formula: str
    n_params: int
    f: ModelF
    jac: ModelJ
    log_response: bool = False

    def __call__(self, b: Sequence[Num], x: Cols, m: SimpleNamespace = NUMPY) -> Num:
        return self.f(b, x, m)


MODELS: dict[str, Model] = {}


def _reg(name: str, formula: str, p: int, *, log_response: bool = False):
    def deco(fj: tuple[ModelF, ModelJ]) -> tuple[ModelF, ModelJ]:
        MODELS[name] = Model(name, formula, p, fj[0], fj[1], log_response)
        return fj

    return deco


def _ones(x: Num) -> Num:
    return x * 0 + 1


# --- exponential saturation: Misra1a, BoxBOD -------------------------------------------


def _sat_f(b, X, m):
    (x,) = X
    return b[0] * (1 - m.exp(-b[1] * x))


def _sat_j(b, X, m):
    (x,) = X
    e = m.exp(-b[1] * x)
    return [1 - e, b[0] * x * e]


_reg("Misra1a", "b1*(1-exp[-b2*x])", 2)((_sat_f, _sat_j))
_reg("BoxBOD", "b1*(1-exp[-b2*x])", 2)((_sat_f, _sat_j))


# --- Chwirut1, Chwirut2 ----------------------------------------------------------------


def _chw_f(b, X, m):
    (x,) = X
    return m.exp(-b[0] * x) / (b[1] + b[2] * x)


def _chw_j(b, X, m):
    (x,) = X
    e = m.exp(-b[0] * x)
    d = b[1] + b[2] * x
    return [-x * e / d, -e / d**2, -x * e / d**2]


_reg("Chwirut1", "exp[-b1*x]/(b2+b3*x)", 3)((_chw_f, _chw_j))
_reg("Chwirut2", "exp(-b1*x)/(b2+b3*x)", 3)((_chw_f, _chw_j))


# --- Lanczos1-3: three exponentials ------------------------------------------------------


def _lan_f(b, X, m):
    (x,) = X
    return b[0] * m.exp(-b[1] * x) + b[2] * m.exp(-b[3] * x) + b[4] * m.exp(-b[5] * x)


def _lan_j(b, X, m):
    (x,) = X
    out = []
    for k in (0, 2, 4):
        e = m.exp(-b[k + 1] * x)
        out += [e, -b[k] * x * e]
    return out


for _n in ("Lanczos1", "Lanczos2", "Lanczos3"):
    _reg(_n, "b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)", 6)((_lan_f, _lan_j))


# --- Gauss1-3: exponential baseline plus two Gaussian peaks ------------------------------


def _gau_f(b, X, m):
    (x,) = X
    return (
        b[0] * m.exp(-b[1] * x)
        + b[2] * m.exp(-((x - b[3]) ** 2) / b[4] ** 2)
        + b[5] * m.exp(-((x - b[6]) ** 2) / b[7] ** 2)
    )


def _gau_j(b, X, m):
    (x,) = X
    e0 = m.exp(-b[1] * x)
    out = [e0, -b[0] * x * e0]
    for a, c, w in ((2, 3, 4), (5, 6, 7)):
        u = x - b[c]
        e = m.exp(-(u**2) / b[w] ** 2)
        out += [e, b[a] * e * 2 * u / b[w] ** 2, b[a] * e * 2 * u**2 / b[w] ** 3]
    return out


for _n in ("Gauss1", "Gauss2", "Gauss3"):
    _reg(
        _n,
        "b1*exp(-b2*x) + b3*exp(-(x-b4)**2/b5**2) + b6*exp(-(x-b7)**2/b8**2)",
        8,
    )((_gau_f, _gau_j))


# --- DanWood ----------------------------------------------------------------------------


def _dan_f(b, X, m):
    (x,) = X
    return b[0] * x ** b[1]


def _dan_j(b, X, m):
    (x,) = X
    p = x ** b[1]
    return [p, b[0] * p * m.log(x)]


_reg("DanWood", "b1*x**b2", 2)((_dan_f, _dan_j))


# --- Misra1b, Misra1c, Misra1d ------------------------------------------------------------


def _m1b_f(b, X, m):
    (x,) = X
    return b[0] * (1 - (1 + b[1] * x / 2) ** (-2))


def _m1b_j(b, X, m):
    (x,) = X
    u = 1 + b[1] * x / 2
    return [1 - u ** (-2), b[0] * x * u ** (-3)]


def _m1c_f(b, X, m):
    (x,) = X
    return b[0] * (1 - (1 + 2 * b[1] * x) ** (-0.5))


def _m1c_j(b, X, m):
    (x,) = X
    u = 1 + 2 * b[1] * x
    return [1 - u ** (-0.5), b[0] * x * u ** (-1.5)]


def _m1d_f(b, X, m):
    (x,) = X
    return b[0] * b[1] * x * ((1 + b[1] * x) ** (-1))


def _m1d_j(b, X, m):
    (x,) = X
    u = 1 + b[1] * x
    return [b[1] * x / u, b[0] * x / u**2]


_reg("Misra1b", "b1*(1-(1+b2*x/2)**(-2))", 2)((_m1b_f, _m1b_j))
_reg("Misra1c", "b1*(1-(1+2*b2*x)**(-.5))", 2)((_m1c_f, _m1c_j))
_reg("Misra1d", "b1*b2*x*((1+b2*x)**(-1))", 2)((_m1d_f, _m1d_j))


# --- rational models: Kirby2 (quadratic/quadratic), Hahn1 and Thurber (cubic/cubic) ------


def _rational(num: int, den: int) -> tuple[ModelF, ModelJ]:
    def f(b, X, m):
        (x,) = X
        n = sum(b[k] * x**k for k in range(num + 1))
        d = 1 + sum(b[num + k] * x**k for k in range(1, den + 1))
        return n / d

    def j(b, X, m):
        (x,) = X
        n = sum(b[k] * x**k for k in range(num + 1))
        d = 1 + sum(b[num + k] * x**k for k in range(1, den + 1))
        out = [_ones(x) / d] + [x**k / d for k in range(1, num + 1)]
        out += [-n * x**k / d**2 for k in range(1, den + 1)]
        return out

    return f, j


_reg("Kirby2", "(b1 + b2*x + b3*x**2) / (1 + b4*x + b5*x**2)", 5)(_rational(2, 2))
_reg("Hahn1", "(b1+b2*x+b3*x**2+b4*x**3) / (1+b5*x+b6*x**2+b7*x**3)", 7)(_rational(3, 3))
_reg("Thurber", "(b1 + b2*x + b3*x**2 + b4*x**3) / (1 + b5*x + b6*x**2 + b7*x**3)", 7)(
    _rational(3, 3)
)


# --- Nelson (log response) ------------------------------------------------------------------


def _nel_f(b, X, m):
    x1, x2 = X
    return b[0] - b[1] * x1 * m.exp(-b[2] * x2)


def _nel_j(b, X, m):
    x1, x2 = X
    e = m.exp(-b[2] * x2)
    return [_ones(x1), -x1 * e, b[1] * x1 * x2 * e]


_reg("Nelson", "log[y] = b1 - b2*x1 * exp[-b3*x2]", 3, log_response=True)((_nel_f, _nel_j))


# --- MGH17 --------------------------------------------------------------------------------


def _m17_f(b, X, m):
    (x,) = X
    return b[0] + b[1] * m.exp(-x * b[3]) + b[2] * m.exp(-x * b[4])


def _m17_j(b, X, m):
    (x,) = X
    e4 = m.exp(-x * b[3])
    e5 = m.exp(-x * b[4])
    return [_ones(x), e4, e5, -b[1] * x * e4, -b[2] * x * e5]


_reg("MGH17", "b1 + b2*exp[-x*b4] + b3*exp[-x*b5]", 5)((_m17_f, _m17_j))


# --- Roszman1 -------------------------------------------------------------------------------


def _ros_f(b, X, m):
    (x,) = X
    return b[0] - b[1] * x - m.atan(b[2] / (x - b[3])) / m.pi


def _ros_j(b, X, m):
    (x,) = X
    u = x - b[3]
    den = m.pi * (u**2 + b[2] ** 2)
    return [_ones(x), -x, -u / den, -b[2] / den]


_reg("Roszman1", "b1 - b2*x - arctan[b3/(x-b4)]/pi", 4)((_ros_f, _ros_j))


# --- ENSO -------------------------------------------------------------------------------------


def _ens_f(b, X, m):
    (x,) = X
    t = 2 * m.pi * x
    return (
        b[0]
        + b[1] * m.cos(t / 12)
        + b[2] * m.sin(t / 12)
        + b[4] * m.cos(t / b[3])
        + b[5] * m.sin(t / b[3])
        + b[7] * m.cos(t / b[6])
        + b[8] * m.sin(t / b[6])
    )


def _ens_j(b, X, m):
    (x,) = X
    t = 2 * m.pi * x
    out = [_ones(x), m.cos(t / 12), m.sin(t / 12)]
    for p, c, s in ((3, 4, 5), (6, 7, 8)):
        a = t / b[p]
        dper = (b[c] * m.sin(a) - b[s] * m.cos(a)) * t / b[p] ** 2
        out += [dper, m.cos(a), m.sin(a)]
    return out


_reg(
    "ENSO",
    "b1 + b2*cos(2*pi*x/12) + b3*sin(2*pi*x/12) + b5*cos(2*pi*x/b4) + b6*sin(2*pi*x/b4)"
    " + b8*cos(2*pi*x/b7) + b9*sin(2*pi*x/b7)",
    9,
)((_ens_f, _ens_j))


# --- MGH09 ---------------------------------------------------------------------------------


def _m09_f(b, X, m):
    (x,) = X
    return b[0] * (x**2 + x * b[1]) / (x**2 + x * b[2] + b[3])


def _m09_j(b, X, m):
    (x,) = X
    n = x**2 + x * b[1]
    d = x**2 + x * b[2] + b[3]
    return [n / d, b[0] * x / d, -b[0] * n * x / d**2, -b[0] * n / d**2]


_reg("MGH09", "b1*(x**2+x*b2) / (x**2+x*b3+b4)", 4)((_m09_f, _m09_j))


# --- Rat42, Rat43 -----------------------------------------------------------------------------


def _r42_f(b, X, m):
    (x,) = X
    return b[0] / (1 + m.exp(b[1] - b[2] * x))


def _r42_j(b, X, m):
    (x,) = X
    e = m.exp(b[1] - b[2] * x)
    d = 1 + e
    return [1 / d, -b[0] * e / d**2, b[0] * x * e / d**2]


def _r43_f(b, X, m):
    (x,) = X
    return b[0] / ((1 + m.exp(b[1] - b[2] * x)) ** (1 / b[3]))


def _r43_j(b, X, m):
    (x,) = X
    e = m.exp(b[1] - b[2] * x)
    d = 1 + e
    p = d ** (-1 / b[3])
    return [
        p,
        -b[0] * p * e / (b[3] * d),
        b[0] * p * x * e / (b[3] * d),
        b[0] * p * m.log(d) / b[3] ** 2,
    ]


_reg("Rat42", "b1 / (1+exp[b2-b3*x])", 3)((_r42_f, _r42_j))
_reg("Rat43", "b1 / ((1+exp[b2-b3*x])**(1/b4))", 4)((_r43_f, _r43_j))


# --- MGH10 ---------------------------------------------------------------------------------


def _m10_f(b, X, m):
    (x,) = X
    return b[0] * m.exp(b[1] / (x + b[2]))


def _m10_j(b, X, m):
    (x,) = X
    u = x + b[2]
    e = m.exp(b[1] / u)
    return [e, b[0] * e / u, -b[0] * e * b[1] / u**2]


_reg("MGH10", "b1 * exp[b2/(x+b3)]", 3)((_m10_f, _m10_j))


# --- Eckerle4 ---------------------------------------------------------------------------------


def _eck_f(b, X, m):
    (x,) = X
    return (b[0] / b[1]) * m.exp(-0.5 * ((x - b[2]) / b[1]) ** 2)


def _eck_j(b, X, m):
    (x,) = X
    u = (x - b[2]) / b[1]
    e = m.exp(-0.5 * u**2)
    g = (b[0] / b[1]) * e
    return [e / b[1], g * (u**2 - 1) / b[1], g * u / b[1]]


_reg("Eckerle4", "(b1/b2) * exp[-0.5*((x-b3)/b2)**2]", 3)((_eck_f, _eck_j))


# --- Bennett5 ----------------------------------------------------------------------------------


def _ben_f(b, X, m):
    (x,) = X
    return b[0] * (b[1] + x) ** (-1 / b[2])


def _ben_j(b, X, m):
    (x,) = X
    u = b[1] + x
    p = u ** (-1 / b[2])
    return [p, -b[0] * p / (b[2] * u), b[0] * p * m.log(u) / b[2] ** 2]


_reg("Bennett5", "b1 * (b2+x)**(-1/b3)", 3)((_ben_f, _ben_j))


def get_model(name: str) -> Model:
    """Model by StRD dataset name."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"no nonlinear model for {name!r}") from None


def response(model: Model, y: np.ndarray) -> np.ndarray:
    """The fitted response: ``log(y)`` for Nelson, ``y`` otherwise (float64)."""
    return np.log(y) if model.log_response else y


def predictors(ds: Any, dtype: Any = np.float64) -> tuple[np.ndarray, ...]:
    """Predictor columns of a nonlinear dataset (all columns after ``y``)."""
    return tuple(ds.array(c, dtype) for c in ds.columns[1:])
