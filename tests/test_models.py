from __future__ import annotations

import mpmath
import numpy as np
import pytest

from strdcheck.manifest import list_datasets
from strdcheck.models import MODELS, NUMPY, get_model, mp_backend, response
from strdcheck.parse import load


def test_all_27_models_registered() -> None:
    names = {d.name for d in list_datasets("nonlinear")}
    assert set(MODELS) == names
    with pytest.raises(KeyError):
        get_model("Norris")


def _point(name: str, rng: np.random.Generator):
    model = MODELS[name]
    b = list(rng.uniform(0.5, 1.5, size=model.n_params))
    x = (0.3,) if name != "Nelson" else (0.3, 1.7)
    if name == "Roszman1":
        x = (3.0,)
    return model, b, x


@pytest.mark.parametrize("name", sorted(MODELS))
def test_jacobian_matches_mpmath_derivative(name: str) -> None:
    """Analytic Jacobians against mpmath numerical derivatives at a generic point."""
    rng = np.random.default_rng(len(name) * 7919)
    model, b, x = _point(name, rng)
    m = mp_backend()
    with mpmath.workdps(30):
        bb = [mpmath.mpf(v) for v in b]
        xx = tuple(mpmath.mpf(v) for v in x)
        jac = model.jac(bb, xx, m)
        for j in range(model.n_params):

            def fj(t, j=j):
                bt = list(bb)
                bt[j] = t
                return model.f(bt, xx, m)

            num = mpmath.diff(fj, bb[j])
            scale = max(abs(num), mpmath.mpf(1e-12))
            assert abs(jac[j] - num) / scale < 1e-8, (name, j)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_numpy_and_mpmath_agree(name: str) -> None:
    rng = np.random.default_rng(len(name))
    model, b, x = _point(name, rng)
    xs = tuple(np.array([v, v * 1.1]) for v in x)
    f_np = model(b, xs, NUMPY)
    with mpmath.workdps(30):
        f_mp = model.f([mpmath.mpf(v) for v in b], tuple(mpmath.mpf(v) for v in x), mp_backend())
    assert f_np[0] == pytest.approx(float(f_mp), rel=1e-13)
    j_np = model.jac(b, xs, NUMPY)
    assert len(j_np) == model.n_params
    for col in j_np:
        assert np.shape(col) == (2,)  # constant columns broadcast to the data length


def test_response_log_for_nelson() -> None:
    y = np.array([1.0, np.e])
    np.testing.assert_allclose(response(MODELS["Nelson"], y), [0.0, 1.0])
    np.testing.assert_array_equal(response(MODELS["BoxBOD"], y), y)


def test_jacobian_at_certified_real_data(strd_dir) -> None:
    """M5 done-check: analytic Jacobians agree with mpmath derivatives at b* (rel 1e-8)."""
    m = mp_backend()
    for info in list_datasets("nonlinear"):
        ds = load(info.name, strd_dir)
        model = get_model(info.name)
        with mpmath.workdps(30):
            b = [mpmath.mpf(ds.certified[p].text) for p in ds.param_names()]
            for i in (0, ds.n // 2, ds.n - 1):
                x = tuple(mpmath.mpf(ds.raw[c][i]) for c in ds.columns[1:])
                jac = model.jac(b, x, m)
                for j in range(len(b)):

                    def fj(t, j=j, x=x, model=model, b=b):
                        bt = list(b)
                        bt[j] = t
                        return model.f(bt, x, m)

                    num = mpmath.diff(fj, b[j])
                    scale = max(abs(num), mpmath.mpf(10) ** -20)
                    assert abs(jac[j] - num) / scale < 1e-8, (info.name, i, j)
