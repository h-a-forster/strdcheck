"""Study config, runner units, analysis statistics, and a small end-to-end run."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from strdcheck.checkpoint import RunStore
from strdcheck.study import analysis
from strdcheck.study.config import StudyConfig, load_config
from strdcheck.study.runner import build_units, random_start, run_study, run_unit
from strdcheck.study.tiled import k_copies, tiled_array

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_load() -> None:
    q = load_config(ROOT / "configs" / "quick.toml")
    f = load_config(ROOT / "configs" / "full.toml")
    assert q.name == "quick" and q.workers == 3 and q.part_b.sigmas == (1.0,)
    assert f.part_b.starts_per_problem == 200 and f.part_b.sigmas == (0.25, 1.0, 2.0)
    assert max(f.part_c.sizes) == 20_000_000


def test_config_validation(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    p.write_text("[study]\nworkers = 5\n")
    with pytest.raises(ValueError, match="workers"):
        load_config(p)
    p.write_text("[part_b]\nsigmas = [0.5]\n")
    with pytest.raises(ValueError, match="sigma"):
        load_config(p)
    p.write_text("[part_c]\norders = ['sorted']\n")
    with pytest.raises(ValueError, match="order"):
        load_config(p)


def test_random_start_seeding() -> None:
    b = np.array([2.0, -3.0, 0.5])
    a1 = random_start(b, 20261023, 4, 7, 1.0)
    a2 = random_start(b, 20261023, 4, 7, 1.0)
    np.testing.assert_array_equal(a1, a2)
    assert np.all(np.sign(a1) == np.sign(b))  # multiplicative: signs kept (DESIGN R5)
    assert not np.array_equal(a1, random_start(b, 20261023, 4, 8, 1.0))
    assert not np.array_equal(a1, random_start(b, 20261023, 4, 7, 2.0))
    with pytest.raises(ValueError):
        random_start(b, 1, 0, 0, 0.5)


def test_tiled_array() -> None:
    y0 = np.arange(5, dtype=float)
    t = tiled_array(y0, 20, "tiled", 1, "NumAcc1")
    assert t.size == 20 and k_copies(5, 20) == 4
    p1 = tiled_array(y0, 20, "permuted", 1, "NumAcc1")
    p2 = tiled_array(y0, 20, "permuted", 1, "NumAcc1")
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(np.sort(p1), np.sort(t))
    with pytest.raises(ValueError):
        tiled_array(y0, 20, "reversed", 1, "NumAcc1")


def test_wilson_holm_bootstrap_permutation() -> None:
    lo, hi = analysis.wilson(0, 20)
    assert lo == 0 and 0.15 < hi < 0.18
    lo, hi = analysis.wilson(10, 20)
    assert lo == pytest.approx(0.299, abs=1e-3) and hi == pytest.approx(0.701, abs=1e-3)
    assert all(math.isnan(v) for v in analysis.wilson(0, 0))
    assert analysis.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert analysis.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    lo, hi = analysis.cluster_bootstrap_mean(np.ones(10), 500, 1)
    assert lo == hi == 1.0
    # all 8 differences positive: exact two-sided p = 2/256; Monte Carlo close to it
    p = analysis.sign_flip_test(np.ones(8), 20000, 3)
    assert abs(p - 2 / 256) < 0.004
    assert analysis.sign_flip_test(np.array([1.0, -1.0, 1.0, -1.0]), 2000, 3) > 0.5


def test_comparisons_detect_a_shift() -> None:
    rng = np.random.default_rng(0)
    probs = [f"p{i}" for i in range(27)]
    pp = {}
    a, b = analysis.COMPARISONS[0][1], analysis.COMPARISONS[0][2]
    pp[(1.0, 4.0, a, "silent")] = {p: 0.5 + 0.1 * rng.random() for p in probs}
    pp[(1.0, 4.0, b, "silent")] = {p: 0.1 * rng.random() for p in probs}
    rows = analysis.comparisons(pp, 1.0, 1000, 2000, 1)
    assert len(rows) == 1 and rows[0]["mean_difference"] > 0.4
    assert rows[0]["p_raw"] < 0.01 and rows[0]["reject_at_0.05"]


def test_build_units_counts(strd_dir) -> None:
    cfg = load_config(ROOT / "configs" / "quick.toml")
    u = build_units(cfg, "run", str(strd_dir))
    assert len(u["P"]) == 58
    assert len(u["B"]) == 27 * 20
    assert {x.start for x in u["A"] if x.suite == "nonlinear"} == {"S1", "S2"}
    assert len([x for x in u["A"] if x.suite == "nonlinear"]) == 27 * 2 * 14
    ids = [x.unit_id for v in u.values() for x in v]
    assert len(ids) == len(set(ids))
    # Part C keeps only engine paths whose module is installed (torch is an optional extra).
    import importlib.util

    from strdcheck.study.tiled import TILED_ADAPTERS

    paths = sum(importlib.util.find_spec(m) is not None for m, _ in TILED_ADAPTERS.values())
    assert len(TILED_ADAPTERS) == 7
    assert len(u["C"]) == 3 * 3 * 2 * paths
    assert {x.adapter for x in u["C"]} == {
        a for a, (m, _) in TILED_ADAPTERS.items() if importlib.util.find_spec(m) is not None
    }


@pytest.mark.slow
def test_end_to_end_small_run(strd_dir, tmp_path: Path) -> None:
    """Inline (one worker) run of P, A, B and C on a tiny configuration, then the tables."""
    from strdcheck.study.config import PartA, PartB, PartC

    cfg = StudyConfig(
        name="tiny",
        workers=1,
        repeats=2,
        part_a=PartA(suites=("univariate",)),
        part_b=PartB(
            starts_per_problem=2, problems=("Misra1a", "BoxBOD"), bootstrap=200, permutations=200
        ),
        part_c=PartC(datasets=("NumAcc4",), sizes=(2002,), orders=("tiled",)),
    )
    run = tmp_path / "run"
    out = run_study(cfg, run, strd_dir)
    assert out["done"] == out["total"]
    assert run_study(cfg, run, strd_dir)["ran"] == {"P": 0, "A": 0, "B": 0, "C": 0}
    store = RunStore(run)
    recs = store.records()
    assert all(r.status == "ok" for r in recs), [r.error for r in recs if r.status != "ok"]
    a = [r for r in recs if r.part == "A" and r.adapter == "control.exact"]
    assert a and all(q.lre >= q.cap - 1 for r in a for q in r.quantities)
    # Inline runs use this process's thread pools, not the 1-thread worker setup; polars
    # streaming with several threads is not bitwise repeatable (REPORT section 5).
    unstable = [
        r.unit_id
        for r in recs
        if r.part == "A" and not r.repeat_identical and "lazy-streaming" not in r.adapter
    ]
    assert not unstable, unstable
    b = [r for r in recs if r.part == "B"]
    assert len(b) == 4 and all(len(r.extra["fits"]) == 14 for r in b)
    res = analysis.write_results(run, tmp_path / "results")
    names = {p.name for p in res}
    assert {
        "lre_table.csv",
        "lre_table.md",
        "nonlinear_rates.csv",
        "comparisons.csv",
        "tiled.csv",
        "environment.json",
        "summary.json",
    } <= names
    rows = list(csv.DictReader((tmp_path / "results" / "comparisons.csv").open()))
    assert len(rows) == 5
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summary["parts"]["A"] == len([r for r in recs if r.part == "A"])
    env = (tmp_path / "results" / "environment.json").read_text()
    assert str(Path.home()) not in env


def test_run_unit_records_adapter_exception(strd_dir, tmp_path, monkeypatch) -> None:
    from strdcheck.checkpoint import execute
    from strdcheck.study.runner import Unit

    store = RunStore(tmp_path / "r")
    store.open({})
    u = Unit(
        "A|univariate|NumAcc1|nope",
        "A",
        "univariate",
        "NumAcc1",
        "nope",
        str(tmp_path / "r"),
        str(strd_dir),
    )
    execute(store, [u], run_unit, workers=1)
    rec = store.records()[0]
    assert rec.status == "error" and "nope" in rec.error


def test_m8_k1_reproduces_exact_f64(strd_dir) -> None:
    """M8 done-check: the tiled reference with k = 1 equals the float64 exact reference."""
    from strdcheck import exact
    from strdcheck.parse import load

    for name in ("NumAcc3", "NumAcc4", "PiDigits"):
        ds = load(name, strd_dir)
        t = exact.tiled_reference(ds.array("y"), 1)
        r = exact.reference(ds, "f64")
        assert t["mean"] == r["mean"] and t["sd"] == r["sd"]


def test_silent_split_separates_equivalent_minima() -> None:
    from strdcheck.results import UnitRecord

    def rec(prob: str, fits: list[dict]) -> UnitRecord:
        return UnitRecord(
            f"B|{prob}", "B", prob, "all", "nonlinear", "ok", extra={"sigma": 1.0, "fits": fits}
        )

    ok = {"adapter": "a", "success": True, "min_param_lre": 8.0, "rss_lre": 9.0}
    at_min = {"adapter": "a", "success": True, "min_param_lre": 0.5, "rss_lre": 7.0}
    wrong = {"adapter": "a", "success": True, "min_param_lre": 0.5, "rss_lre": 1.0}
    flagged = {"adapter": "a", "success": False, "min_param_lre": 0.5, "rss_lre": 1.0}
    err = {"adapter": "a", "error": "boom"}
    recs = [rec("P1", [ok, at_min, wrong, wrong]), rec("P2", [flagged, err, ok, at_min])]
    (row,) = analysis.silent_split(recs, 4.0, 200, 1)
    assert row["problems"] == 2 and row["problems_with_wrong_rss"] == 1
    assert row["silent_at_min"] == pytest.approx((1 / 4 + 1 / 4) / 2)
    assert row["silent_wrong"] == pytest.approx((2 / 4 + 0) / 2)


def test_silent_split_ceiling_aware_rss() -> None:
    """A silent fit at the float64 RSS ceiling (Lanczos1: 3.06 digits) counts as at_min."""
    from strdcheck.results import UnitRecord

    fits = [{"adapter": "a", "success": True, "min_param_lre": 0.5, "rss_lre": 3.0}] * 2 + [
        {"adapter": "a", "success": True, "min_param_lre": 0.5, "rss_lre": 0.3}
    ]
    recs = [
        UnitRecord("B|L", "B", "L", "all", "nonlinear", "ok", extra={"sigma": 1.0, "fits": fits})
    ]
    (plain,) = analysis.silent_split(recs, 4.0, 200, 1)
    assert plain["silent_at_min"] == 0 and plain["silent_wrong"] == pytest.approx(1.0)
    (aware,) = analysis.silent_split(recs, 4.0, 200, 1, rss_ceilings={"L": 3.0639})
    assert aware["silent_at_min"] == pytest.approx(2 / 3)
    assert aware["silent_wrong"] == pytest.approx(1 / 3)
    assert analysis.rss_threshold(4.0, 3.0639) == pytest.approx(2.0639)
    assert analysis.rss_threshold(4.0, 10.8) == 4.0
    assert analysis.rss_threshold(4.0, math.nan) == 4.0


def test_write_results_tolerates_error_records(tmp_path: Path) -> None:
    """One failed unit per part (empty extra and quantities) must not break the tables."""
    from strdcheck.checkpoint import _error_record
    from strdcheck.results import UnitRecord
    from strdcheck.study.runner import Unit

    cfg = StudyConfig(name="err").to_dict()
    store = RunStore(tmp_path / "run")
    store.open(cfg)
    for part, suite in (
        ("P", "linear"),
        ("A", "univariate"),
        ("B", "nonlinear"),
        ("C", "univariate"),
    ):
        u = Unit(f"{part}|x", part, suite, "Norris", "ad", "run", "data")
        store.write(_error_record(u, RuntimeError("boom")))
    fit = {"adapter": "a", "success": True, "min_param_lre": 8.0, "rss_lre": 9.0}
    store.write(
        UnitRecord(
            "B|ok",
            "B",
            "Misra1a",
            "all",
            "nonlinear",
            "ok",
            extra={"sigma": 1.0, "s_index": 0, "difficulty": "lower", "fits": [fit]},
        )
    )
    out = tmp_path / "res"
    analysis.write_results(tmp_path / "run", out)
    summary = json.loads((out / "summary.json").read_text("utf-8"))
    assert summary["status"] == {"error": 4, "ok": 1}
    assert len(summary["error_units"]) == 4
