"""Registry of every adapter configuration (each id appears exactly once)."""

from __future__ import annotations

from functools import cache

from strdcheck.engines.base import Adapter


@cache
def all_adapters() -> tuple[Adapter, ...]:
    from strdcheck.engines import anova, linear, nonlinear, univariate

    out = (*univariate.ADAPTERS, *anova.ADAPTERS, *linear.ADAPTERS, *nonlinear.ADAPTERS)
    keys = [(a.suite, a.id) for a in out]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise RuntimeError(f"duplicate adapter ids: {sorted(dup)}")
    return out


def adapters(suite: str | None = None, role: str | None = None) -> tuple[Adapter, ...]:
    return tuple(
        a
        for a in all_adapters()
        if (suite is None or a.suite == suite) and (role is None or a.role == role)
    )


def get_adapter(suite: str, adapter_id: str) -> Adapter:
    for a in all_adapters():
        if a.suite == suite and a.id == adapter_id:
            return a
    raise KeyError(f"no adapter {adapter_id!r} for suite {suite!r}")
