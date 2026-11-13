"""Render every results table REPORT.md publishes from the committed CSVs under results/.

Writes results/report_tables.md. tests/test_report_tables.py checks that every numeric table row
in REPORT.md is a line of that file, and CI reruns this script and fails on any diff, so a table
in the report cannot drift from the committed results. The CSVs themselves come from the study
runs (`strdcheck table`), which need the NIST files and are not rerun in CI.

Run from the repository root: uv run python scripts/report_tables.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from strdcheck.study import analysis

RESULTS = Path("results")
OUT = RESULTS / "report_tables.md"

UNIVARIATE = ["PiDigits", "Mavro", "Michelso", "NumAcc3", "NumAcc4"]
UNIVARIATE_ROWS = [
    (
        "numpy, pandas, polars, scipy, statsmodels, statistics (f64)",
        ["numpy.f64", "pandas.f64", "polars.eager.f64", "polars.lazy-streaming.f64",
         "scipy.describe.f64", "statsmodels.f64", "stdlib.statistics"],
    ),
    ("torch.f64", ["torch.f64"]),
    ("duckdb.double", ["duckdb.double"]),
    ("float32 paths (numpy, pandas, polars, torch)",
     ["numpy.f32", "pandas.f32", "polars.eager.f32", "torch.f32"]),
    ("duckdb.float (`FLOAT` column, double accumulation)", ["duckdb.float"]),
]  # fmt: skip
ANOVA = ["SmLs03", "SmLs06", "SmLs09", "AtmWtAg"]
ANOVA_ROWS = [
    ("scipy.f_oneway (F only)", ["scipy.f_oneway"]),
    ("numpy two-pass, pandas, polars group-by",
     ["numpy.twopass", "pandas.groupby", "polars.groupby"]),
    ("duckdb.groupby.decimal", ["duckdb.groupby.decimal"]),
    ("duckdb.groupby.double", ["duckdb.groupby.double"]),
    ("statsmodels.anova_lm", ["statsmodels.anova_lm"]),
]  # fmt: skip
LINEAR = ["Norris", "Pontius", "Filip", "Longley", "Wampler4", "Wampler5"]
# Each row joins its adapters' cells with " / "; a dataset an adapter does not fit is empty.
LINEAR_ROWS = [
    ("numpy.linalg.lstsq", ["numpy.lstsq"]),
    ("scipy.linalg.lstsq gelsd / gelss / gelsy",
     ["scipy.lstsq.gelsd", "scipy.lstsq.gelss", "scipy.lstsq.gelsy"]),
    ("statsmodels OLS pinv / qr", ["statsmodels.ols.pinv", "statsmodels.ols.qr"]),
    ("torch.linalg.lstsq gelsd / gelsy (f64)", ["torch.lstsq.gelsd.f64", "torch.lstsq.gelsy.f64"]),
    ("numpy.polyfit / Polynomial.fit", ["numpy.polyfit", "numpy.Polynomial.fit"]),
    ("normal equations (negative control)", ["control.normal-equations.f64"]),
]  # fmt: skip

LS = "scipy.least_squares.{}.jac-{}.tol-{}"
CURVE_FIT = "scipy.curve_fit.default"
NIST_STARTS_ROWS = [
    ("least_squares trf, analytic Jacobian, tight tolerances",
     [LS.format("trf", "analytic", "tight")]),
    ("least_squares lm, analytic, tight", [LS.format("lm", "analytic", "tight")]),
    ("least_squares dogbox, analytic, tight", [LS.format("dogbox", "analytic", "tight")]),
    ("least_squares trf, analytic, default tolerances", [LS.format("trf", "analytic", "default")]),
    ("least_squares lm, analytic, default", [LS.format("lm", "analytic", "default")]),
    ("curve_fit, defaults (2-point Jacobian, lm)", [CURVE_FIT]),
    ("least_squares trf, 2-point, default", [LS.format("trf", "2point", "default")]),
    ("least_squares lm or dogbox, 2-point, default",
     [LS.format("lm", "2point", "default"), LS.format("dogbox", "2point", "default")]),
]  # fmt: skip
COMPARISON_LABELS = {
    "C1": "lm vs trf (analytic, default tol)",
    "C2": "dogbox vs trf (analytic, default tol)",
    "C3": "analytic vs 2-point Jacobian (trf, default tol)",
    "C4": "tight vs default tolerances (trf, analytic)",
    "C5": "curve_fit defaults vs trf (analytic, tight)",
}
FULL_ROWS = [
    LS.format("trf", "analytic", "tight"),
    LS.format("lm", "analytic", "tight"),
    LS.format("dogbox", "analytic", "tight"),
    LS.format("trf", "analytic", "default"),
    LS.format("trf", "2point", "default"),
    CURVE_FIT,
]
TILED_ROWS = [
    ("numpy (pairwise summation)", "numpy.f64"),
    ("pandas", "pandas.f64"),
    ("torch", "torch.f64"),
    ("DuckDB, 1 thread", "duckdb.double.threads-1"),
    ("DuckDB, 3 threads", "duckdb.double.threads-3"),
    ("polars eager", "polars.eager.f64"),
    ("polars streaming (lazy, 3 threads)", "polars.lazy-streaming.f64"),
]
PRIMARY_SIGMA, PRIMARY_TAU = "1.0", "4.0"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row(cells: list[str]) -> str:
    return ("| " + " | ".join(cells) + " |").replace("|  |", "| |")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [row(header), "|" + "---|" * len(header), *(row(r) for r in rows), ""]


def lre(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.1f}"


def span(values: list[float]) -> str:
    lo, hi = lre(min(values)), lre(max(values))
    return lo if lo == hi else f"{lo}-{hi}"


def label(adapter: str) -> str:
    if adapter == CURVE_FIT:
        return "curve_fit, defaults"
    if adapter == "scipy.curve_fit.jac-analytic":
        return "curve_fit, analytic Jacobian"
    _, _, method, jac, tol = adapter.split(".")
    jac = {"jac-analytic": "analytic", "jac-2point": "2-point"}[jac]
    return f"least_squares {method}, {jac}, {tol.removeprefix('tol-')}"


def minima(rows: list[dict[str, str]], suite: str, quantity) -> tuple[dict, dict]:
    """Minimum LRE per (dataset, adapter) over the quantities `quantity` accepts, and the
    minimum float64 ceiling per dataset over the same quantities (as in lre_table.md)."""
    low: dict[tuple[str, str], float] = {}
    ceiling: dict[str, float] = {}
    for r in rows:
        if r["suite"] != suite or not quantity(r) or r["status"] != "ok":
            continue
        key = (r["dataset"], r["adapter"])
        value = float(r["lre"]) if r["lre"] not in ("", "nan") else float("nan")
        low[key] = min(low.get(key, math.inf), value) if not math.isnan(value) else value
        if r["dtype"] == "f64":
            ceiling[r["dataset"]] = min(ceiling.get(r["dataset"], math.inf), float(r["ceiling"]))
    return low, ceiling


def part_a(out: list[str]) -> None:
    rows = read(RESULTS / "lre_table.csv")
    engine = [r for r in rows if r["role"] == "engine" or r["adapter"].startswith("control.n")]

    low, ceiling = minima(engine, "univariate", lambda r: True)
    body = [["float64 ceiling", *(lre(ceiling[d]) for d in UNIVARIATE)]]
    for name, adapters in UNIVARIATE_ROWS:
        body.append([name, *(span([low[d, a] for a in adapters]) for d in UNIVARIATE)])
    out += ["## 3.1 Univariate summary statistics", ""]
    out += table(["Minimum LRE (mean, SD, lag-1 autocorrelation)", *UNIVARIATE], body)

    low, ceiling = minima(engine, "anova", lambda r: r["derived"] == "False")
    body = [["float64 ceiling", *(lre(ceiling[d]) for d in ANOVA)]]
    for name, adapters in ANOVA_ROWS:
        body.append([name, *(span([low[d, a] for a in adapters]) for d in ANOVA)])
    out += ["## 3.2 One-way ANOVA", ""]
    out += table(["Minimum LRE", *ANOVA], body)

    coefficient = lambda r: r["quantity"].startswith("b")  # noqa: E731
    low, ceiling = minima(engine, "linear", coefficient)
    conditioning = {r["dataset"]: float(r["cond"]) for r in read(RESULTS / "conditioning.csv")}
    best = {
        d: max(v for (ds, a), v in low.items() if ds == d and _f64_engine(a, engine))
        for d in LINEAR
    }
    body = [
        ["cond(X), 2-norm", *(_sci(conditioning[d]) for d in LINEAR)],
        ["float64 ceiling", *(lre(ceiling[d]) for d in LINEAR)],
        ["best float64 engine", *(lre(best[d]) for d in LINEAR)],
    ]
    for name, adapters in LINEAR_ROWS:
        cells = []
        for d in LINEAR:
            have = [lre(low[d, a]) for a in adapters if (d, a) in low]
            if have and len(have) != len(adapters):
                raise SystemExit(f"{name}: some but not all adapters fit {d}")
            cells.append(" / ".join(have))
        body.append([name, *cells])
    out += ["## 3.3 Linear least squares", ""]
    out += table(["Minimum LRE of coefficients", *LINEAR], body)


def _f64_engine(adapter: str, rows: list[dict[str, str]]) -> bool:
    kinds = {(r["role"], r["dtype"]) for r in rows if r["adapter"] == adapter}
    return kinds == {("engine", "f64")}


def _sci(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def per_problem(folder: Path) -> tuple[dict, dict]:
    """Per-problem outcome rates at the primary tau, keyed as `analysis.outcome_rates` keys them,
    rebuilt from the committed counts, and the run's Part B settings."""
    part_b = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["config"]["part_b"]
    rates: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in read(folder / "nonlinear_problem_rates.csv"):
        starts = int(r["starts"])
        for outcome in ("ok", "silent", "flagged"):
            key = (float(r["sigma"]), analysis.PRIMARY_TAU, r["adapter"], outcome)
            rates[key][r["problem"]] = int(r[outcome]) / starts
    return rates, part_b


def interval(rates: dict[str, float], part_b: dict) -> tuple[float, float, float]:
    """Mean over problems and its cluster-bootstrap interval, as the study computes them."""
    values = np.array([rates[p] for p in sorted(rates)])
    lo, hi = analysis.cluster_bootstrap_mean(values, int(part_b["bootstrap"]), int(part_b["seed"]))
    return float(values.mean()), lo, hi


def check(name: str, recomputed: float, committed: str) -> float:
    """The committed CSVs round to 4 decimals; the recomputed value must agree with them."""
    if abs(recomputed - float(committed)) > 0.00005 + 1e-12:
        raise SystemExit(f"{name}: recomputed {recomputed} against committed {committed}")
    return recomputed


def part_b(out: list[str]) -> None:
    starts = {
        r["adapter"]: r for r in read(RESULTS / "nonlinear_nist_starts.csv") if r["tier"] == "all"
    }
    body = []
    for name, adapters in NIST_STARTS_ROWS:
        counts = {tuple(starts[a][k] for k in ("ok", "silent", "flagged")) for a in adapters}
        if len(counts) != 1:
            raise SystemExit(f"{name}: the joined configurations differ")
        body.append([name, *counts.pop()])
    out += ["## 3.5 Nonlinear problems from the NIST starts", ""]
    out += table(["Configuration", "ok", "silent", "flagged"], body)

    rates, part_b = per_problem(RESULTS)
    sigma = float(PRIMARY_SIGMA)
    committed = {
        r["adapter"]: r
        for r in read(RESULTS / "nonlinear_rates.csv")
        if (r["sigma"], r["tau"], r["tier"]) == (PRIMARY_SIGMA, PRIMARY_TAU, "all")
    }
    shown = []
    for adapter, c in committed.items():
        cells = {}
        for outcome in ("ok", "silent", "flagged"):
            mean, lo, hi = interval(rates[sigma, analysis.PRIMARY_TAU, adapter, outcome], part_b)
            cells[outcome] = check(f"{adapter} {outcome}", mean, c[outcome])
            if outcome == "silent":
                cells["lo"] = check(f"{adapter} silent_lo", lo, c["silent_lo"])
                cells["hi"] = check(f"{adapter} silent_hi", hi, c["silent_hi"])
        shown.append((adapter, cells))
    body = [
        [label(a), f"{c['ok']:.3f}", f"{c['silent']:.3f} ({c['lo']:.3f}-{c['hi']:.3f})",
         f"{c['flagged']:.3f}"]
        for a, c in sorted(shown, key=lambda x: x[1]["silent"])
    ]  # fmt: skip
    out += ["## 4.1 Rates at tau = 4, sigma = 1", ""]
    out += table(["Configuration", "ok", "silent (95% CI)", "flagged"], body)

    fresh = analysis.comparisons(
        rates, sigma, int(part_b["bootstrap"]), int(part_b["permutations"]), int(part_b["seed"])
    )
    body = []
    for r, c in zip(fresh, read(RESULTS / "comparisons.csv"), strict=True):
        cid = r["comparison"].split()[0]
        for field in ("mean_difference", "ci_lo", "ci_hi", "p_raw", "p_holm"):
            check(f"{cid} {field}", r[field], c[field])
        body.append([
            cid, COMPARISON_LABELS[cid],
            f"{r['rate_first']:.3f} vs {r['rate_second']:.3f}",
            f"{r['mean_difference']:.3f} ({r['ci_lo']:.3f}, {r['ci_hi']:.3f})",
            f"{r['problems_first_worse']} / {r['problems_first_better']}",
            f"{r['p_raw']:.3f}",
            f"**{r['p_holm']:.3f}**" if r["reject_at_0.05"] else f"{r['p_holm']:.3f}",
        ])  # fmt: skip
    out += ["## 4.3 Pre-specified comparisons", ""]
    out += table(
        ["", "Comparison", "Rates", "Difference (95% CI)", "Problems worse / better", "p",
         "Holm p"], body)  # fmt: skip

    rates, part_b = per_problem(RESULTS / "full")
    sigmas = [float(s) for s in part_b["sigmas"]]
    body = []
    for a in FULL_ROWS:
        cells = []
        for s in sigmas:
            mean, lo, hi = interval(rates[s, analysis.PRIMARY_TAU, a, "silent"], part_b)
            cells.append(f"{mean:.3f} ({lo:.3f}-{hi:.3f})")
        body.append([label(a), *cells])
    out += ["## 4.4 Full-size run: silent rate by spread", ""]
    out += table(["Silent rate at tau = 4 (95% CI)", *(f"sigma = {s}" for s in sigmas)], body)


def part_c(out: list[str]) -> None:
    by = defaultdict(list)
    for r in read(RESULTS / "tiled.csv"):
        by[r["adapter"]].append(r)
    body = []
    for name, adapter in TILED_ROWS:
        cells = by[adapter]
        sd = [float(r["sd_lre"]) for r in cells]
        moving = sum(r["repeat_identical"] != "True" for r in cells)
        body.append([name, lre(min(sd)), lre(statistics.median(sd)), f"{moving} / {len(cells)}"])
    if {len(v) for v in by.values()} != {18}:
        raise SystemExit("tiled.csv no longer has 3 datasets x 3 sizes x 2 orders per engine")
    out += ["## 5. Part C: large n", ""]
    out += table(
        ["SD LRE over 3 datasets x 3 sizes x 2 orders", "Worst cell", "Median",
         "Not bitwise repeatable"], body)  # fmt: skip


def readme_fragments() -> list[str]:
    """The README's key-result numbers, as the phrases it states them in."""
    tight, default = LS.format("trf", "analytic", "tight"), LS.format("trf", "analytic", "default")
    out = []
    for folder in (RESULTS, RESULTS / "full"):
        rates, part_b = per_problem(folder)
        key = (float(PRIMARY_SIGMA), analysis.PRIMARY_TAU)

        def mean(adapter: str, outcome: str, rates=rates, key=key, part_b=part_b) -> float:
            return interval(rates[(*key, adapter, outcome)], part_b)[0]

        silent, lo, hi = interval(rates[(*key, tight, "silent")], part_b)
        drop = mean(default, "silent") - silent
        better = mean(tight, "ok") - mean(default, "ok")
        reported = (mean(tight, "flagged") - mean(default, "flagged")) / drop
        significant = [r for r in read(folder / "comparisons.csv") if r["reject_at_0.05"] == "True"]
        c4 = significant[0]
        if len(significant) != 1 or not c4["comparison"].startswith("C4"):
            raise SystemExit(f"{folder}: C4 is no longer the only significant comparison")
        p = float(c4["p_holm"])
        if folder == RESULTS:
            out += [
                f"(for example {silent:.2f}, 95% CI {lo:.2f}-{hi:.2f}, for",
                f"trf by {drop:.3f} (Holm p = {p:.3f}), but only {better:.3f} of that is more "
                f"correct fits; {reported:.0%} of the drop is wrong fits",
            ]
        else:
            out += [
                f"agrees: {silent:.2f} ({lo:.2f}-{hi:.2f}) at the same spread",
                f"(-{drop:.3f}, Holm p = {p:.4f}; {better:.3f} more correct fits, {reported:.0%} "
                "of the drop now reported failures)",
            ]
    tiled = defaultdict(list)
    for r in read(RESULTS / "tiled.csv"):
        tiled[r["adapter"]].append(float(r["sd_lre"]))
    exact = [min(tiled[a]) for a in ("numpy.f64", "pandas.f64")]
    others = [min(v) for a, v in tiled.items() if a not in ("numpy.f64", "pandas.f64")]
    if min(exact) < 15.0:
        raise SystemExit("numpy or pandas no longer keep 15 digits on tiled data")
    out.append(f"keep {lre(min(others))}-{lre(max(others))} in their worst cell")
    return out


def render() -> str:
    out = [
        "# Report tables",
        "",
        "Generated by `scripts/report_tables.py` from the CSVs in this folder; REPORT.md copies",
        "these rows. Do not edit by hand.",
        "",
    ]
    part_a(out)
    part_b(out)
    part_c(out)
    out += ["## README: key results", "", *(f"- {f}" for f in readme_fragments()), ""]
    return "\n".join(out).rstrip("\n") + "\n"


def main() -> None:
    OUT.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
