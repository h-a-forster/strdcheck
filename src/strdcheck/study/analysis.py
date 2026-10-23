"""Analysis of a run directory into ``results/`` (DESIGN 6.2, 6.3, 6.8).

Outputs (all small, committed):

- ``lre_table.csv``: every Part A quantity (LRE, ceiling, deficit, status, reproducibility);
- ``lre_table.md``: per-suite pivot of the minimum LRE (headline group) with ceilings;
- ``notable.csv``: Part A cells meeting the DESIGN 6.2 criteria (input to triage);
- ``nonlinear_nist_starts.csv``: outcome counts per configuration and difficulty tier;
- ``nonlinear_rates.csv``: Part B outcome rates with 95% cluster-bootstrap intervals;
- ``nonlinear_problem_rates.csv``: per-problem silent-failure rates with Wilson intervals;
- ``comparisons.csv``: the five pre-specified comparisons (permutation p, Holm);
- ``tiled.csv``: Part C LREs against derived references;
- ``ceilings.csv``: representation ceilings; ``conditioning.csv``: cond(X) of the linear
  designs; ``environment.json``; ``summary.json``.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from strdcheck.checkpoint import RunStore
from strdcheck.engines.nonlinear import OUTCOMES, classify
from strdcheck.results import UnitRecord

PRIMARY_TAU = 4.0
NOTABLE_DEFICIT = 2.0
NOTABLE_MIN_CEILING = 6.0

COMPARISONS: tuple[tuple[str, str, str], ...] = (
    (
        "C1 lm vs trf (analytic, default tol)",
        "scipy.least_squares.lm.jac-analytic.tol-default",
        "scipy.least_squares.trf.jac-analytic.tol-default",
    ),
    (
        "C2 dogbox vs trf (analytic, default tol)",
        "scipy.least_squares.dogbox.jac-analytic.tol-default",
        "scipy.least_squares.trf.jac-analytic.tol-default",
    ),
    (
        "C3 analytic vs 2-point Jacobian (trf, default tol)",
        "scipy.least_squares.trf.jac-analytic.tol-default",
        "scipy.least_squares.trf.jac-2point.tol-default",
    ),
    (
        "C4 tight vs default tolerances (trf, analytic)",
        "scipy.least_squares.trf.jac-analytic.tol-tight",
        "scipy.least_squares.trf.jac-analytic.tol-default",
    ),
    (
        "C5 curve_fit default vs least_squares trf analytic tight",
        "scipy.curve_fit.default",
        "scipy.least_squares.trf.jac-analytic.tol-tight",
    ),
)
"""Pre-specified in DESIGN 6.3 before any run: difference (first minus second) in the
silent-failure rate at tau = 4."""


# --- small statistics helpers ----------


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_mean(values: np.ndarray, B: int, seed: int) -> tuple[float, float]:
    """95% percentile interval of the mean over clusters (rows of ``values``)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(B, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def sign_flip_test(diffs: np.ndarray, B: int, seed: int) -> float:
    """Two-sided sign-flip permutation p-value for mean(diffs) = 0, ``(1 + count) / (B + 1)``."""
    d = np.asarray(diffs, dtype=float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    count = 0
    chunk = 2000
    done = 0
    while done < B:
        m = min(chunk, B - done)
        signs = rng.choice((-1.0, 1.0), size=(m, d.size))
        stats = np.abs((signs * d).mean(axis=1))
        count += int(np.sum(stats >= obs - 1e-12))
        done += m
    return (1 + count) / (B + 1)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


# --- writers ----------


def _fmt(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf"
        return round(x, 4)
    return x


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})
    return path


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# --- Part A ----------


def _headline(rec: UnitRecord, param_names: set[str]) -> list:
    """Quantities summarised in the pivot: coefficients for linear/nonlinear, else all
    non-derived quantities."""
    if rec.suite in ("linear", "nonlinear"):
        qs = [q for q in rec.quantities if q.name in param_names]
    else:
        qs = [q for q in rec.quantities if not q.derived]
    return qs


def part_a_rows(records: list[UnitRecord]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        if r.part != "A":
            continue
        for q in r.quantities:
            rows.append(
                {
                    "suite": r.suite,
                    "dataset": r.dataset,
                    "difficulty": r.extra.get("difficulty", ""),
                    "adapter": r.adapter,
                    "role": r.extra.get("role", ""),
                    "dtype": r.extra.get("dtype", ""),
                    "start": r.start or "",
                    "quantity": q.name,
                    "derived": q.derived,
                    "computed": repr(q.computed),
                    "certified": q.certified,
                    "cap": q.cap,
                    "lre": q.lre,
                    "ceiling": q.ceiling,
                    "deficit": q.deficit,
                    "success": r.engine_status.get("success", ""),
                    "rank": r.engine_status.get("rank", ""),
                    "p": r.engine_status.get("p", ""),
                    "repeat_identical": r.repeat_identical,
                    "status": r.status,
                    "version": r.extra.get("version", ""),
                }
            )
    return rows


def notable_cells(records: list[UnitRecord]) -> list[dict[str, Any]]:
    """DESIGN 6.2 criteria, one row per (dataset, adapter, start, criterion).

    Criterion (a) rows name the worst quantity and give, for context, the best LRE that any
    float64 engine adapter reached on that quantity: when the best engine is also far below
    the ceiling, the loss is the problem's conditioning; when it is not, the adapter's
    algorithm (or a documented default such as ``rcond``) is the explanation to check.
    """
    best: dict[tuple[str, str, str, str], float] = {}
    for r in records:
        if r.part == "A" and r.extra.get("dtype") == "f64" and r.extra.get("role") == "engine":
            for q in r.quantities:
                k = (r.suite, r.dataset, r.start or "", q.name)
                if not math.isnan(q.lre):
                    best[k] = max(best.get(k, -1.0), q.lre)
    out = []
    for r in records:
        if r.part != "A":
            continue
        base = {
            "suite": r.suite,
            "dataset": r.dataset,
            "adapter": r.adapter,
            "start": r.start or "",
            "role": r.extra.get("role", ""),
        }
        if r.repeat_identical is False:
            out.append({**base, "criterion": "c: repeats not bitwise identical"})
        if r.suite == "nonlinear":
            if r.extra.get("outcome") == "silent":
                out.append(
                    {
                        **base,
                        "criterion": "b: success reported, < 4 digits",
                        "quantity": "min param",
                        "lre": r.extra.get("min_param_lre"),
                    }
                )
            continue
        if r.extra.get("dtype") != "f64" or r.extra.get("role") != "engine":
            continue
        bad = [
            q
            for q in r.quantities
            if q.ceiling >= NOTABLE_MIN_CEILING
            and (math.isnan(q.lre) or q.ceiling - q.lre > NOTABLE_DEFICIT)
        ]
        if not bad:
            continue
        worst = max(bad, key=lambda q: math.inf if math.isnan(q.lre) else q.ceiling - q.lre)
        out.append(
            {
                **base,
                "criterion": "a: deficit > 2 (float64, ceiling >= 6)",
                "quantities": len(bad),
                "derived_only": all(q.derived for q in bad),
                "quantity": worst.name + (" (derived)" if worst.derived else ""),
                "lre": worst.lre,
                "ceiling": worst.ceiling,
                "best_engine_lre": best.get(
                    (r.suite, r.dataset, r.start or "", worst.name), math.nan
                ),
                "rank": r.engine_status.get("rank", ""),
                "p": r.engine_status.get("p", ""),
            }
        )
    return out


def _pivot_md(records: list[UnitRecord], suite: str, names: dict[str, set[str]]) -> str:
    from strdcheck.manifest import list_datasets

    recs = [r for r in records if r.part == "A" and r.suite == suite]
    if not recs:
        return ""
    datasets = [d.name for d in list_datasets(suite)]  # type: ignore[arg-type]
    adapters = list(dict.fromkeys(r.adapter for r in recs))
    starts = sorted({r.start or "" for r in recs})
    lines = []
    for s in starts:
        cell: dict[tuple[str, str], str] = {}
        ceil: dict[str, float] = {}
        for r in recs:
            if (r.start or "") != s:
                continue
            qs = _headline(r, names.get(r.dataset, set()))
            if not qs:
                continue
            vals = [q.lre for q in qs]
            v = math.nan if any(math.isnan(x) for x in vals) else min(vals)
            txt = "nan" if math.isnan(v) else f"{v:.1f}"
            deficits = [q.deficit for q in qs if not math.isnan(q.deficit)]
            if r.extra.get("role") == "engine" and (
                math.isnan(v) or (deficits and max(deficits) > NOTABLE_DEFICIT)
            ):
                txt += "*"
            if r.suite == "nonlinear" and r.extra.get("outcome") in ("flagged", "flagged_ok"):
                txt += "f"
            cell[(r.dataset, r.adapter)] = txt
            if r.extra.get("dtype") == "f64":
                cs = [q.ceiling for q in qs]
                ceil[r.dataset] = min(cs) if cs else math.nan
        title = f"### {suite}" + (f", NIST {s.replace('S', 'Start ')}" if s else "")
        lines += [title, ""]
        head = "| adapter | " + " | ".join(datasets) + " |"
        lines += [head, "|" + "---|" * (len(datasets) + 1)]
        lines.append(
            "| *float64 ceiling* | "
            + " | ".join(
                "" if math.isnan(ceil.get(d, math.nan)) else f"*{ceil[d]:.1f}*" for d in datasets
            )
            + " |"
        )
        for a in adapters:
            lines.append(f"| {a} | " + " | ".join(cell.get((d, a), "") for d in datasets) + " |")
        lines.append("")
    return "\n".join(lines)


# --- Part B ----------


def _fits(records: list[UnitRecord]):
    """Yield (problem, difficulty, sigma, s_index, adapter, success, min_lre, error)."""
    for r in records:
        if r.part != "B":
            continue
        for f in r.extra.get("fits", []):
            err = "error" in f
            yield (
                r.dataset,
                r.extra.get("difficulty", ""),
                float(r.extra["sigma"]),
                int(r.extra["s_index"]),
                f["adapter"],
                f.get("success"),
                _f(f.get("min_param_lre")),
                err,
            )


def outcome_rates(records: list[UnitRecord], taus: Sequence[float], B: int, seed: int):
    """Per (sigma, tau, adapter, tier) rates of each outcome, problems weighted equally."""
    counts: dict[tuple, Counter] = defaultdict(Counter)
    totals: dict[tuple, int] = Counter()
    tiers: dict[str, str] = {}
    for prob, diff, sigma, _s, ad, succ, mlre, err in _fits(records):
        tiers[prob] = diff
        for tau in taus:
            o = classify(succ, mlre, tau=tau, error=err)
            counts[(sigma, tau, ad, prob)][o] += 1
            totals[(sigma, tau, ad, prob)] += 1
    keys = sorted({(k[0], k[1], k[2]) for k in counts})
    rows = []
    per_problem: dict[tuple, dict[str, np.ndarray]] = {}
    for sigma, tau, ad in keys:
        probs = sorted(p for (s, t, a, p) in counts if (s, t, a) == (sigma, tau, ad))
        for tier in ("all", "lower", "average", "higher"):
            sel = [p for p in probs if tier == "all" or tiers[p] == tier]
            if not sel:
                continue
            row: dict[str, Any] = {
                "sigma": sigma,
                "tau": tau,
                "adapter": ad,
                "tier": tier,
                "problems": len(sel),
                "fits": sum(totals[(sigma, tau, ad, p)] for p in sel),
            }
            for o in OUTCOMES:
                rates = np.array(
                    [counts[(sigma, tau, ad, p)][o] / totals[(sigma, tau, ad, p)] for p in sel]
                )
                row[o] = float(rates.mean())
                lo, hi = cluster_bootstrap_mean(rates, B, seed)
                row[f"{o}_lo"], row[f"{o}_hi"] = lo, hi
                if tier == "all":
                    per_problem[(sigma, tau, ad, o)] = dict(zip(sel, rates, strict=True))
            rows.append(row)
    return rows, per_problem, counts, totals


def silent_split(records: list[UnitRecord], tau: float, B: int, seed: int) -> list[dict[str, Any]]:
    """Secondary analysis: split silent failures by the residual sum of squares.

    A silent fit whose RSS still agrees with the certified minimum to ``tau`` digits reached
    an equivalent minimum (exchangeable model components, such as the terms of a sum of
    exponentials, or a flat direction of the objective) and differs from the certified
    point only in parameter labelling or ill-determined parameters. A silent fit with a wrong
    RSS stopped at a different stationary point or short of the minimum. Rates are per
    problem and then averaged, as in :func:`outcome_rates`.
    """
    cnt: dict[tuple, Counter] = defaultdict(Counter)
    for r in records:
        if r.part != "B":
            continue
        sigma = float(r.extra["sigma"])
        for f in r.extra.get("fits", []):
            key = (sigma, f["adapter"], r.dataset)
            cnt[key]["fits"] += 1
            if "error" in f:
                continue
            o = classify(f.get("success"), _f(f.get("min_param_lre")), tau=tau)
            if o != "silent":
                continue
            rss = _f(f.get("rss_lre"))
            cnt[key]["at_min" if math.isfinite(rss) and rss >= tau else "wrong"] += 1
    rows = []
    for sigma, ad in sorted({(k[0], k[1]) for k in cnt}):
        probs = sorted(k[2] for k in cnt if (k[0], k[1]) == (sigma, ad))
        row: dict[str, Any] = {"sigma": sigma, "tau": tau, "adapter": ad, "problems": len(probs)}
        for part in ("at_min", "wrong"):
            rates = np.array(
                [cnt[(sigma, ad, p)][part] / cnt[(sigma, ad, p)]["fits"] for p in probs]
            )
            lo, hi = cluster_bootstrap_mean(rates, B, seed)
            row[f"silent_{part}"], row[f"silent_{part}_lo"], row[f"silent_{part}_hi"] = (
                float(rates.mean()),
                lo,
                hi,
            )
        row["problems_with_wrong_rss"] = sum(cnt[(sigma, ad, p)]["wrong"] > 0 for p in probs)
        rows.append(row)
    return rows


SPLIT_FIELDS = [
    "sigma",
    "tau",
    "adapter",
    "problems",
    "silent_at_min",
    "silent_at_min_lo",
    "silent_at_min_hi",
    "silent_wrong",
    "silent_wrong_lo",
    "silent_wrong_hi",
    "problems_with_wrong_rss",
]


def comparisons(per_problem, sigma: float, B_boot: int, B_perm: int, seed: int):
    rows = []
    for i, (label, a, b) in enumerate(COMPARISONS):
        pa = per_problem.get((sigma, PRIMARY_TAU, a, "silent"))
        pb = per_problem.get((sigma, PRIMARY_TAU, b, "silent"))
        if pa is None or pb is None:
            continue
        probs = sorted(set(pa) & set(pb))
        d = np.array([pa[p] - pb[p] for p in probs])
        lo, hi = cluster_bootstrap_mean(d, B_boot, seed + 101 + i)
        rows.append(
            {
                "comparison": label,
                "first": a,
                "second": b,
                "sigma": sigma,
                "tau": PRIMARY_TAU,
                "problems": len(probs),
                "rate_first": float(np.mean([pa[p] for p in probs])),
                "rate_second": float(np.mean([pb[p] for p in probs])),
                "mean_difference": float(d.mean()),
                "ci_lo": lo,
                "ci_hi": hi,
                "problems_first_worse": int(np.sum(d > 0)),
                "problems_first_better": int(np.sum(d < 0)),
                "p_raw": sign_flip_test(d, B_perm, seed + 202 + i),
            }
        )
    adj = holm([r["p_raw"] for r in rows])
    for r, p in zip(rows, adj, strict=True):
        r["p_holm"] = p
        r["reject_at_0.05"] = p < 0.05
    return rows


# --- Part C ----------


def tiled_rows(records: list[UnitRecord]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        if r.part != "C":
            continue
        q = {x.name: x for x in r.quantities}
        rows.append(
            {
                "dataset": r.dataset,
                "n": r.extra.get("n"),
                "k": r.extra.get("k"),
                "order": r.extra.get("order"),
                "adapter": r.adapter,
                "mean_lre": q["mean"].lre,
                "sd_lre": q["sd"].lre,
                "repeat_identical": r.repeat_identical,
                "version": r.extra.get("version", ""),
            }
        )
    rows.sort(key=lambda x: (x["dataset"], x["order"], x["adapter"], x["n"]))
    return rows


# --- entry point ----------


def write_results(run_dir: str | Path, out_dir: str | Path) -> list[Path]:
    from strdcheck.parse import LINEAR_SPECS

    store = RunStore(run_dir)
    man = store.manifest()
    cfg = man["config"]
    records = store.records()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    names: dict[str, set[str]] = {k: set(v.param_names()) for k, v in LINEAR_SPECS.items()}
    from strdcheck.models import MODELS

    for k, m in MODELS.items():
        names[k] = {f"b{j}" for j in range(1, m.n_params + 1)}

    # ceilings
    crow = []
    for r in records:
        if r.part == "P":
            for q, c in r.extra.get("ceilings", {}).items():
                crow.append(
                    {
                        "dataset": r.dataset,
                        "suite": r.suite,
                        "quantity": q,
                        "dec": _f(c.get("dec")),
                        "f64": _f(c.get("f64")),
                        "f32": _f(c.get("f32")),
                    }
                )
    written.append(
        _write_csv(
            out / "ceilings.csv", crow, ["dataset", "suite", "quantity", "dec", "f64", "f32"]
        )
    )
    cond = [
        {"dataset": r.dataset, "cond": r.extra["cond"], "log10_cond": math.log10(r.extra["cond"])}
        for r in records
        if r.part == "P" and "cond" in r.extra
    ]
    if cond:
        written.append(
            _write_csv(out / "conditioning.csv", cond, ["dataset", "cond", "log10_cond"])
        )

    # Part A
    arows = part_a_rows(records)
    afields = [
        "suite",
        "dataset",
        "difficulty",
        "adapter",
        "role",
        "dtype",
        "start",
        "quantity",
        "derived",
        "computed",
        "certified",
        "cap",
        "lre",
        "ceiling",
        "deficit",
        "success",
        "rank",
        "p",
        "repeat_identical",
        "status",
        "version",
    ]
    written.append(_write_csv(out / "lre_table.csv", arows, afields))
    notable = notable_cells(records)
    written.append(
        _write_csv(
            out / "notable.csv",
            notable,
            [
                "suite",
                "dataset",
                "adapter",
                "start",
                "role",
                "criterion",
                "quantities",
                "derived_only",
                "quantity",
                "lre",
                "ceiling",
                "best_engine_lre",
                "rank",
                "p",
            ],
        )
    )
    md = [
        "# LRE table",
        "",
        "Minimum LRE (correct significant digits) per dataset and adapter: coefficients for",
        "linear and nonlinear regression, all engine-computed quantities otherwise.",
        "`*` marks an engine cell whose deficit below the representation ceiling exceeds",
        "2 digits on some quantity (or a non-finite result); `f` marks a nonlinear fit the",
        "solver reported as failed. The ceiling row is the float64 ceiling of the same",
        "quantities. Generated by `strdcheck table`; full detail in `lre_table.csv`.",
        "",
    ]
    for suite in ("univariate", "anova", "linear", "nonlinear"):
        md.append(_pivot_md(records, suite, names))
    (out / "lre_table.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    written.append(out / "lre_table.md")

    # nonlinear NIST starts
    nist = Counter()
    for r in records:
        if r.part == "A" and r.suite == "nonlinear":
            for tier in ("all", r.extra.get("difficulty", "")):
                nist[(r.adapter, tier, r.extra.get("outcome", "error"))] += 1
    nrows = []
    for ad in dict.fromkeys(a for a, _, _ in nist):
        for tier in ("all", "lower", "average", "higher"):
            row = {"adapter": ad, "tier": tier}
            row.update({o: nist[(ad, tier, o)] for o in OUTCOMES})
            row["fits"] = sum(row[o] for o in OUTCOMES)
            nrows.append(row)
    written.append(
        _write_csv(out / "nonlinear_nist_starts.csv", nrows, ["adapter", "tier", "fits", *OUTCOMES])
    )

    # Part B
    pb = cfg.get("part_b", {})
    summary: dict[str, Any] = {"config": cfg, "units": len(records)}
    if any(r.part == "B" for r in records):
        seed = int(pb.get("seed", 20261023))
        rates, per_problem, counts, totals = outcome_rates(
            records, pb.get("taus", [2.0, 4.0, 6.0]), int(pb.get("bootstrap", 10000)), seed
        )
        rfields = ["sigma", "tau", "adapter", "tier", "problems", "fits"]
        for o in OUTCOMES:
            rfields += [o, f"{o}_lo", f"{o}_hi"]
        written.append(_write_csv(out / "nonlinear_rates.csv", rates, rfields))
        split = silent_split(records, PRIMARY_TAU, int(pb.get("bootstrap", 10000)), seed)
        written.append(_write_csv(out / "nonlinear_silent_split.csv", split, SPLIT_FIELDS))
        prow = []
        for (sigma, tau, ad, prob), c in sorted(counts.items()):
            if tau != PRIMARY_TAU:
                continue
            n = totals[(sigma, tau, ad, prob)]
            lo, hi = wilson(c["silent"], n)
            prow.append(
                {
                    "sigma": sigma,
                    "adapter": ad,
                    "problem": prob,
                    "starts": n,
                    "silent": c["silent"],
                    "ok": c["ok"],
                    "flagged": c["flagged"],
                    "flagged_ok": c["flagged_ok"],
                    "error": c["error"],
                    "silent_rate": c["silent"] / n,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                }
            )
        written.append(
            _write_csv(
                out / "nonlinear_problem_rates.csv",
                prow,
                [
                    "sigma",
                    "adapter",
                    "problem",
                    "starts",
                    "silent",
                    "ok",
                    "flagged",
                    "flagged_ok",
                    "error",
                    "silent_rate",
                    "wilson_lo",
                    "wilson_hi",
                ],
            )
        )
        sigmas = sorted({k[0] for k in counts})
        primary_sigma = 1.0 if 1.0 in sigmas else sigmas[0]
        comp = comparisons(
            per_problem,
            primary_sigma,
            int(pb.get("bootstrap", 10000)),
            int(pb.get("permutations", 20000)),
            seed,
        )
        written.append(
            _write_csv(
                out / "comparisons.csv",
                comp,
                [
                    "comparison",
                    "first",
                    "second",
                    "sigma",
                    "tau",
                    "problems",
                    "rate_first",
                    "rate_second",
                    "mean_difference",
                    "ci_lo",
                    "ci_hi",
                    "problems_first_worse",
                    "problems_first_better",
                    "p_raw",
                    "p_holm",
                    "reject_at_0.05",
                ],
            )
        )
        summary["part_b"] = {
            "fits": sum(totals.values()) // max(len(pb.get("taus", [1])), 1),
            "primary_sigma": primary_sigma,
        }

    # Part C
    trows = tiled_rows(records)
    if trows:
        written.append(
            _write_csv(
                out / "tiled.csv",
                trows,
                [
                    "dataset",
                    "n",
                    "k",
                    "order",
                    "adapter",
                    "mean_lre",
                    "sd_lre",
                    "repeat_identical",
                    "version",
                ],
            )
        )

    env = man.get("environment", {})
    (out / "environment.json").write_text(
        json.dumps(env, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    written.append(out / "environment.json")
    status = Counter(r.status for r in records)
    summary.update(
        {
            "status": dict(status),
            "parts": dict(Counter(r.part for r in records)),
            "notable": len(notable),
            "part_a_quantities": len(arows),
            "created": man.get("created"),
            "config_hash": man.get("config_hash"),
            "seconds": {
                p: round(sum(r.seconds for r in records if r.part == p), 1) for p in "PABC"
            },
        }
    )
    (out / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    written.append(out / "summary.json")
    return written
