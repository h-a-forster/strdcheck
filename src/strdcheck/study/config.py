"""Study configuration (``configs/*.toml``), DESIGN 6."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CANONICAL_SIGMAS = (0.25, 1.0, 2.0)
"""Spreads of the random starts; a start's seed uses the index in this tuple, so the quick
study's starts are the first starts of the full study."""


@dataclass(frozen=True)
class PartA:
    enabled: bool = True
    suites: tuple[str, ...] = ("univariate", "anova", "linear", "nonlinear")


@dataclass(frozen=True)
class PartB:
    enabled: bool = True
    seed: int = 20261023
    sigmas: tuple[float, ...] = (1.0,)
    starts_per_problem: int = 20
    taus: tuple[float, ...] = (2.0, 4.0, 6.0)
    bootstrap: int = 10000
    permutations: int = 20000
    problems: tuple[str, ...] = ()  # empty: all 27 (a subset is for tests only)


@dataclass(frozen=True)
class PartC:
    enabled: bool = True
    seed: int = 20261023
    datasets: tuple[str, ...] = ("NumAcc3", "NumAcc4", "PiDigits")
    sizes: tuple[int, ...] = (10_000, 100_000, 1_000_000)
    orders: tuple[str, ...] = ("tiled", "permuted")


@dataclass(frozen=True)
class StudyConfig:
    name: str = "quick"
    workers: int = 3
    repeats: int = 3
    part_a: PartA = field(default_factory=PartA)
    part_b: PartB = field(default_factory=PartB)
    part_c: PartC = field(default_factory=PartC)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuples(d: dict[str, Any]) -> dict[str, Any]:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def load_config(path: str | Path) -> StudyConfig:
    raw = tomllib.loads(Path(path).read_text("utf-8"))
    study = raw.get("study", {})
    cfg = StudyConfig(
        name=study.get("name", Path(path).stem),
        workers=int(study.get("workers", 3)),
        repeats=int(study.get("repeats", 3)),
        part_a=PartA(**_tuples(raw.get("part_a", {}))),
        part_b=PartB(**_tuples(raw.get("part_b", {}))),
        part_c=PartC(**_tuples(raw.get("part_c", {}))),
    )
    if not 1 <= cfg.workers <= 3:
        raise ValueError("workers must be between 1 and 3")
    for s in cfg.part_b.sigmas:
        if s not in CANONICAL_SIGMAS:
            raise ValueError(f"sigma {s} not in {CANONICAL_SIGMAS}")
    for o in cfg.part_c.orders:
        if o not in ("tiled", "permuted"):
            raise ValueError(f"unknown order {o!r}")
    return cfg
