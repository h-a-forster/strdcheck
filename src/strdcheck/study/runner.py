"""Study runner (DESIGN 5.5, 6.1-6.5): builds units, runs them resumably, records results.

Phases, each a set of checkpointed units in the same run directory:

- ``P``: ceilings per dataset (exact rationals; mpmath refinement for nonlinear problems);
- ``A``: every applicable adapter on every certified dataset (nonlinear: NIST Starts 1 and
  2), repeated ``repeats`` times in-process;
- ``B``: seeded random starts for the 27 nonlinear problems, all 14 configurations per start
  (one unit = one start);
- ``C``: tiled data, one unit = (dataset, n, order, configuration), in a single worker.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from strdcheck.checkpoint import RunStore, execute
from strdcheck.results import EngineResult, UnitRecord, score
from strdcheck.study.config import CANONICAL_SIGMAS, StudyConfig

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class Unit:
    unit_id: str
    part: str
    suite: str
    dataset: str
    adapter: str
    run_dir: str
    data_dir: str
    start: str | None = None
    payload: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)


# --- worker-side helpers ---------------------------------------------------------------------


def worker_init(torch_threads: int = 1) -> None:
    """Runs in each spawned worker before any unit (threads were set via the environment)."""
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import torch

        torch.set_num_threads(torch_threads)
    except ImportError:
        pass


@cache
def _dataset(name: str, data_dir: str):
    from strdcheck.parse import load

    return load(name, data_dir)


@cache
def _ceilings(run_dir: str, dataset: str) -> dict[str, dict[str, float]]:
    store = RunStore(run_dir)
    p = store.path(f"P|{dataset}")
    if not p.exists():
        return {}
    rec = UnitRecord.from_json(p.read_text("utf-8"))
    return {
        k: {kk: float(vv) for kk, vv in v.items()} for k, v in rec.extra.get("ceilings", {}).items()
    }


def _ceil_for(run_dir: str, dataset: str, dtype: str) -> dict[str, float]:
    return {k: v.get(dtype, math.nan) for k, v in _ceilings(run_dir, dataset).items()}


def _version(module: str) -> str:
    from importlib import metadata

    dist = {"sklearn": "scikit-learn"}.get(module, module)
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return ""


# --- unit functions ----------


def _run_p(u: Unit) -> UnitRecord:
    from strdcheck import exact

    t0 = time.perf_counter()
    ds = _dataset(u.dataset, u.data_dir)
    c = exact.nonlinear_ceilings(ds) if ds.suite == "nonlinear" else exact.ceilings(ds)
    ceil = {k: {"dec": v.dec, "f64": v.f64, "f32": v.f32} for k, v in c.items()}
    return UnitRecord(
        u.unit_id,
        "P",
        u.dataset,
        "-",
        ds.suite,
        "ok",
        seconds=time.perf_counter() - t0,
        extra={"ceilings": ceil},
    )


def _start_vector(ds, start: str | None) -> tuple[float, ...] | None:
    if start is None:
        return None
    return ds.starts[int(start[1:]) - 1]


def _run_a(u: Unit) -> UnitRecord:
    from strdcheck.engines.nonlinear import classify
    from strdcheck.engines.registry import get_adapter

    ds = _dataset(u.dataset, u.data_dir)
    a = get_adapter(u.suite, u.adapter)
    kw = {}
    if u.suite == "nonlinear":
        kw["start"] = _start_vector(ds, u.start)
    repeats = int(u.payload.get("repeats", 3))
    results: list[EngineResult] = [a.run(ds, **kw) for _ in range(repeats)]
    res = results[0]
    rows = score(ds.certified, res, _ceil_for(u.run_dir, u.dataset, a.dtype))
    extra: dict[str, Any] = {
        "role": a.role,
        "dtype": a.dtype,
        "engine": a.engine,
        "difficulty": ds.info.difficulty,
        "version": _version(a.engine.split("+")[0]),
    }
    if u.suite == "nonlinear":
        b = [q.lre for q in rows if q.name in ds.param_names()]
        mb = min(b) if b and not any(math.isnan(v) for v in b) else math.nan
        extra["min_param_lre"] = mb
        extra["outcome"] = classify(res.status.get("success"), mb)
    return UnitRecord(
        u.unit_id,
        "A",
        u.dataset,
        u.adapter,
        u.suite,
        "ok",
        start=u.start,
        quantities=rows,
        engine_status=res.status,
        warnings=res.warnings,
        repeat_hashes=[r.digest() for r in results],
        seconds=sum(r.seconds for r in results),
        extra=extra,
    )


def random_start(bstar: np.ndarray, seed: int, p_index: int, s_index: int, sigma: float):
    """``b0_j = b*_j * exp(sigma * z_j)``, z seeded per (problem, start, sigma); DESIGN 6.3."""
    sigma_index = CANONICAL_SIGMAS.index(sigma)
    rng = np.random.default_rng(np.random.SeedSequence([seed, p_index, s_index, sigma_index]))
    z = rng.standard_normal(bstar.size)
    return bstar * np.exp(sigma * z)


def _run_b(u: Unit) -> UnitRecord:
    from strdcheck.engines.nonlinear import ADAPTERS
    from strdcheck.lre import lre

    ds = _dataset(u.dataset, u.data_dir)
    names = ds.param_names()
    bstar = np.array([ds.certified[n].value for n in names])
    pl = u.payload
    x0 = random_start(bstar, pl["seed"], pl["p_index"], pl["s_index"], pl["sigma"])
    fits = []
    t0 = time.perf_counter()
    for a in ADAPTERS:
        try:
            res = a.run(ds, start=tuple(float(v) for v in x0))
        except Exception as exc:
            fits.append({"adapter": a.id, "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        lres = [
            lre(res.values[n], ds.certified[n].value, cap=ds.certified[n].cap).value for n in names
        ]
        mb = math.nan if any(math.isnan(v) for v in lres) else min(lres)
        fits.append(
            {
                "adapter": a.id,
                "success": bool(res.status.get("success")),
                "status": res.status.get("status"),
                "nfev": res.status.get("nfev"),
                "min_param_lre": mb,
                "rss_lre": lre(
                    res.values["rss"], ds.certified["rss"].value, cap=ds.certified["rss"].cap
                ).value,
            }
        )
    return UnitRecord(
        u.unit_id,
        "B",
        u.dataset,
        "all",
        "nonlinear",
        "ok",
        start=u.start,
        seconds=time.perf_counter() - t0,
        extra={
            "difficulty": ds.info.difficulty,
            "sigma": pl["sigma"],
            "s_index": pl["s_index"],
            "x0": [float(v) for v in x0],
            "fits": fits,
        },
    )


def _run_c(u: Unit) -> UnitRecord:
    from strdcheck import exact
    from strdcheck.lre import lre
    from strdcheck.study.tiled import TILED_ADAPTERS, k_copies, tiled_array

    ds = _dataset(u.dataset, u.data_dir)
    y0 = ds.array("y")
    n, order = int(u.payload["n"]), u.payload["order"]
    k = k_copies(y0.size, n)
    ref = exact.tiled_reference(y0, k)
    y = tiled_array(y0, n, order, int(u.payload["seed"]), u.dataset)
    module, fn = TILED_ADAPTERS[u.adapter]
    repeats = int(u.payload.get("repeats", 3))
    outs, secs = [], 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        outs.append(fn(y))
        secs += time.perf_counter() - t0
    vals = outs[0]
    rows = []
    from strdcheck.results import QuantityRecord

    for name in ("mean", "sd"):
        r = lre(vals[name], float(ref[name]), cap=15)
        rows.append(
            QuantityRecord(
                name,
                vals[name],
                str(ref[name])[:40],
                15,
                r.raw,
                r.value,
                15.0,
                max(15.0 - r.value, 0.0) if not math.isnan(r.value) else math.nan,
            )
        )
    hashes = [EngineResult(o).digest() for o in outs]
    return UnitRecord(
        u.unit_id,
        "C",
        u.dataset,
        u.adapter,
        "univariate",
        "ok",
        quantities=rows,
        repeat_hashes=hashes,
        seconds=secs,
        extra={
            "n": int(y.size),
            "k": k,
            "order": order,
            "version": _version(module),
            "reference": "derived (exact for the float64 tiled data)",
        },
    )


def run_unit(u: Unit) -> UnitRecord:
    fn = {"P": _run_p, "A": _run_a, "B": _run_b, "C": _run_c}[u.part]
    return fn(u)


# --- unit construction ----------


def build_units(cfg: StudyConfig, run_dir: str, data_dir: str) -> dict[str, list[Unit]]:
    from strdcheck.engines.registry import adapters
    from strdcheck.manifest import list_datasets
    from strdcheck.study.tiled import TILED_ADAPTERS

    rd, dd = str(run_dir), str(data_dir)
    out: dict[str, list[Unit]] = {"P": [], "A": [], "B": [], "C": []}
    suites = cfg.part_a.suites if cfg.part_a.enabled else ()
    need_p = set(suites)  # Part B and C do not use ceilings
    for suite in ("univariate", "anova", "linear", "nonlinear"):
        if suite in need_p:
            for info in list_datasets(suite):  # type: ignore[arg-type]
                out["P"].append(Unit(f"P|{info.name}", "P", suite, info.name, "-", rd, dd))
    for suite in suites:
        avail = [a for a in adapters(suite) if a.available()]
        for info in list_datasets(suite):  # type: ignore[arg-type]
            ds = _dataset(info.name, dd)
            for a in avail:
                if not a.applicable(ds):
                    continue
                starts = ("S1", "S2") if suite == "nonlinear" else (None,)
                for s in starts:
                    uid = f"A|{suite}|{info.name}|{a.id}" + (f"|{s}" if s else "")
                    out["A"].append(
                        Unit(
                            uid,
                            "A",
                            suite,
                            info.name,
                            a.id,
                            rd,
                            dd,
                            start=s,
                            payload={"repeats": cfg.repeats},
                        )
                    )
    if cfg.part_b.enabled:
        for p_index, info in enumerate(list_datasets("nonlinear")):
            if cfg.part_b.problems and info.name not in cfg.part_b.problems:
                continue
            for sigma in cfg.part_b.sigmas:
                for s in range(cfg.part_b.starts_per_problem):
                    uid = f"B|{info.name}|sigma={sigma}|s={s}"
                    out["B"].append(
                        Unit(
                            uid,
                            "B",
                            "nonlinear",
                            info.name,
                            "all",
                            rd,
                            dd,
                            start=f"R{s}",
                            payload={
                                "seed": cfg.part_b.seed,
                                "p_index": p_index,
                                "s_index": s,
                                "sigma": sigma,
                            },
                        )
                    )
    if cfg.part_c.enabled:
        for name in cfg.part_c.datasets:
            for n in cfg.part_c.sizes:
                for order in cfg.part_c.orders:
                    for aid, (module, _) in TILED_ADAPTERS.items():
                        import importlib.util

                        if importlib.util.find_spec(module) is None:
                            continue
                        uid = f"C|{name}|n={n}|{order}|{aid}"
                        out["C"].append(
                            Unit(
                                uid,
                                "C",
                                "univariate",
                                name,
                                aid,
                                rd,
                                dd,
                                payload={
                                    "n": n,
                                    "order": order,
                                    "seed": cfg.part_c.seed,
                                    "repeats": cfg.repeats,
                                },
                            )
                        )
    return out


def _set_threads(n: int, polars: int | None = None) -> dict[str, str | None]:
    old = {k: os.environ.get(k) for k in THREAD_VARS}
    for k in THREAD_VARS:
        os.environ[k] = str(n)
    if polars is not None:
        os.environ["POLARS_MAX_THREADS"] = str(polars)
    return old


def _restore(old: dict[str, str | None]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_study(cfg: StudyConfig, run_dir: str | Path, data_dir: str | Path | None = None) -> dict:
    """Run (or resume) a study; returns counts per phase."""
    from strdcheck.env import environment_record
    from strdcheck.fetch import resolve_data_dir

    run_dir = Path(run_dir)
    dd = resolve_data_dir(data_dir)
    store = RunStore(run_dir)
    store.open(cfg.to_dict(), meta={"environment": environment_record()})
    units = build_units(cfg, str(run_dir), str(dd))
    total = sum(len(v) for v in units.values())
    store.log(
        f"study {cfg.name}: {total} units " + ", ".join(f"{k}={len(v)}" for k, v in units.items())
    )
    counts = {}
    t0 = time.perf_counter()
    for phase in ("P", "A", "B"):
        if not units[phase]:
            continue
        old = _set_threads(1)
        try:
            ts = time.perf_counter()
            counts[phase] = execute(
                store, units[phase], run_unit, workers=cfg.workers, initializer=worker_init
            )
            store.log(f"phase {phase}: {counts[phase]} run in {time.perf_counter() - ts:.0f} s")
        finally:
            _restore(old)
    if units["C"]:
        old = _set_threads(1, polars=3)
        try:
            ts = time.perf_counter()
            counts["C"] = execute(
                store, units["C"], run_unit, workers=1, spawn=True, initializer=worker_init
            )
            store.log(f"phase C: {counts['C']} run in {time.perf_counter() - ts:.0f} s")
        finally:
            _restore(old)
    store.log(
        f"study {cfg.name} complete: {store.count()}/{total} units, "
        f"{time.perf_counter() - t0:.0f} s this attempt"
    )
    return {"total": total, "done": store.count(), "ran": counts}
