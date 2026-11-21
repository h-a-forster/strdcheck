"""Platform spread: the same study configuration run on more than one machine, cell by cell.

Compares the committed table folders of each platform:

- the reference platform, the published run: ``results/`` (quick) and ``results/full/``;
- every other platform: ``results/platforms/<name>/`` (quick) and, when present,
  ``results/platforms/<name>/full/``, written there by ``strdcheck table <run> --out ...``.

A Part A cell is one (suite, dataset, adapter, start) unit. Its LRE on a platform is the
minimum over the unit's non-derived quantities (``nan``, a failure, counts as 0 here); its
spread is the largest absolute LRE difference between platforms over all of its quantities;
``bitwise`` says whether every computed value is the same float on every platform. Its verdict
class is, for nonlinear fits, the outcome class of DESIGN 4.5 (ok, silent, flagged,
flagged_ok, error); for the other suites ``error`` when the unit failed, ``silent`` when a
float64 engine adapter returns a quantity more than 2 digits below a ceiling of at least 6
(notable criterion (a), DESIGN 6.2) and ``pass`` otherwise. A Part B cell is one (sigma,
adapter, problem) row of ``nonlinear_problem_rates.csv``; its class is its outcome counts.

Writes results/platforms/spread.csv and results/platforms/spread.md. REPORT.md copies rows and
sentences of spread.md; tests/test_platform_spread.py checks both files against a rerun of this
script and the copies against spread.md.

Run from the repository root: uv run python scripts/platform_spread.py
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results")
PLATFORMS = RESULTS / "platforms"
REFERENCE = "laptop-win"
PARAM = re.compile(r"b\d+$")
NOTABLE_MIN_CEILING, NOTABLE_DEFICIT, TAU = 6.0, 2.0, 4.0
OUTCOMES = ("ok", "silent", "flagged", "flagged_ok", "error")
THRESHOLDS = (0.5, 2.0)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def platforms() -> dict[str, Path]:
    """Name -> quick-run table folder, the reference first, then the others by name."""
    found = {REFERENCE: RESULTS}
    for d in sorted(PLATFORMS.iterdir()) if PLATFORMS.is_dir() else []:
        if (d / "lre_table.csv").is_file():
            found[d.name] = d
    return found


def _lre(x: str) -> float:
    v = float(x)
    return 0.0 if math.isnan(v) else v


def _cell_class(rows: list[dict[str, str]]) -> str:
    first = rows[0]
    if any(r["status"] != "ok" for r in rows):
        return "error"
    if first["suite"] == "nonlinear":
        if first["success"] not in ("True", "False"):
            return "error"
        params = [float(r["lre"]) for r in rows if PARAM.fullmatch(r["quantity"])]
        good = bool(params) and all(math.isfinite(v) for v in params) and min(params) >= TAU
        if first["success"] == "True":
            return "ok" if good else "silent"
        return "flagged_ok" if good else "flagged"
    if first["dtype"] == "f64" and first["role"] == "engine":
        for r in rows:
            ceiling, lre = float(r["ceiling"]), float(r["lre"])
            short = math.isnan(lre) or ceiling - lre > NOTABLE_DEFICIT
            if ceiling >= NOTABLE_MIN_CEILING and short:
                return "silent"
    return "pass"


def part_a_cells(folder: Path) -> dict[tuple[str, ...], list[dict[str, str]]]:
    cells: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for r in read(folder / "lre_table.csv"):
        cells[(r["suite"], r["dataset"], r["adapter"], r["start"])].append(r)
    return cells


def part_b_cells(folder: Path) -> dict[tuple[str, ...], str]:
    path = folder / "nonlinear_problem_rates.csv"
    if not path.is_file():
        return {}
    return {
        ("nonlinear", r["problem"], r["adapter"], f"sigma {r['sigma']}"): " ".join(
            f"{o}={r[o]}" for o in OUTCOMES
        )
        for r in read(path)
    }


def compare(run: str, folders: dict[str, Path]) -> list[dict[str, str]]:
    """One output row per cell present on every platform; raises if the cell sets differ."""
    names = list(folders)
    out: list[dict[str, str]] = []
    a = {n: part_a_cells(p) for n, p in folders.items()}
    keys = set(a[names[0]])
    for n in names[1:]:
        if set(a[n]) != keys:
            raise ValueError(f"{run}: Part A cells of {n} differ from {names[0]}")
    for key in sorted(keys):
        per = {n: {r["quantity"]: r for r in a[n][key]} for n in names}
        quantities = list(per[names[0]])
        if any(list(per[n]) != quantities for n in names):
            raise ValueError(f"{run}: quantities of {key} differ between platforms")
        diff = max(
            max(_lre(per[n][q]["lre"]) for n in names) - min(_lre(per[n][q]["lre"]) for n in names)
            for q in quantities
        )
        bitwise = all(
            per[n][q]["computed"] == per[names[0]][q]["computed"] for n in names for q in quantities
        )
        row = {"run": run, "part": "A", "suite": key[0], "dataset": key[1], "adapter": key[2],
               "start": key[3]}  # fmt: skip
        classes = []
        for n in names:
            own = [r for r in per[n].values() if r["derived"] != "True"] or list(per[n].values())
            row[f"lre_{n}"] = f"{min(_lre(r['lre']) for r in own):.4f}"
            classes.append(_cell_class(list(per[n].values())))
            row[f"class_{n}"] = classes[-1]
        row.update(max_abs_diff=f"{diff:.4f}", bitwise=str(bitwise),
                   class_changes=str(len(set(classes)) > 1))  # fmt: skip
        out.append(row)
    b = {n: part_b_cells(p) for n, p in folders.items()}
    if all(b.values()):
        keys = set(b[names[0]])
        for n in names[1:]:
            if set(b[n]) != keys:
                raise ValueError(f"{run}: Part B cells of {n} differ from {names[0]}")
        for key in sorted(keys):
            row = {"run": run, "part": "B", "suite": key[0], "dataset": key[1], "adapter": key[2],
                   "start": key[3]}  # fmt: skip
            for n in names:
                row[f"lre_{n}"] = ""
                row[f"class_{n}"] = b[n][key]
            same = len({b[n][key] for n in names}) == 1
            row.update(max_abs_diff="", bitwise="", class_changes=str(not same))
            out.append(row)
    return out


def rows_all() -> tuple[list[str], list[dict[str, str]]]:
    quick = platforms()
    rows = compare("quick", quick)
    full = {n: (RESULTS / "full" if n == REFERENCE else p / "full") for n, p in quick.items()}
    full = {n: p for n, p in full.items() if (p / "lre_table.csv").is_file()}
    if len(full) > 1:
        rows += compare("full", full)
    return list(quick), rows


def fields(names: list[str]) -> list[str]:
    per = [f"{k}_{n}" for n in names for k in ("lre", "class")]
    head = ["run", "part", "suite", "dataset", "adapter", "start"]
    return head + per + ["max_abs_diff", "bitwise", "class_changes"]


def md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _env(folder: Path) -> list[str]:
    e = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
    blas = e["numpy_build"]["blas"]
    kernel = ", ".join(sorted({t.get("architecture", "") for t in e.get("threadpools", [])}))
    return [
        e["cpu"],
        e["os"],
        e["python"],
        f"{blas['name']} {blas['version']}",
        kernel or "n/a",
        "MKL" if e.get("torch", {}).get("mkl") else "no MKL",
        e.get("git_commit", ""),
    ]


def summary(names: list[str], rows: list[dict[str, str]]) -> list[str]:
    """Counts per run, and the sentences REPORT.md and README.md copy."""
    out: list[str] = []
    for run in ("quick", "full"):
        a = [r for r in rows if r["run"] == run and r["part"] == "A"]
        b = [r for r in rows if r["run"] == run and r["part"] == "B"]
        if not a:
            continue
        diffs = [float(r["max_abs_diff"]) for r in a]
        changed = [r for r in a + b if r["class_changes"] == "True"]
        same = sum(r["bitwise"] == "True" for r in a)
        out += [
            f"### {run} run: {', '.join(names)}",
            "",
            md_row(
                [
                    "Part A cells",
                    "bitwise identical",
                    "differ at all",
                    "> 0.5 digits",
                    "> 2 digits",
                    "largest difference",
                    "class changes",
                ]
            ),  # fmt: skip
            "|" + "---|" * 7,
            md_row(
                [
                    f"{len(a):,}",
                    f"{same:,}",
                    str(sum(d > 0 for d in diffs)),
                    *(str(sum(d > t for d in diffs)) for t in THRESHOLDS),
                    f"{max(diffs):.1f}",
                    str(sum(r["class_changes"] == "True" for r in a)),
                ]
            ),  # fmt: skip
            "",
        ]
        if b:
            diff_b = sum(r["class_changes"] == "True" for r in b)
            out += [f"Part B: {len(b)} (sigma, adapter, problem) cells, {diff_b} with different "
                    "outcome counts.", ""]  # fmt: skip
        big = [r for r in a if float(r["max_abs_diff"]) > THRESHOLDS[0]]
        if big:
            out += [
                md_row(
                    [
                        "Cell (suite, dataset, adapter, start)",
                        *(f"LRE {n}" for n in names),
                        "max difference",
                    ]
                ),  # fmt: skip
                "|" + "---|" * (len(names) + 2),
            ]
            for r in sorted(big, key=lambda r: -float(r["max_abs_diff"])):
                parts = (r["suite"], r["dataset"], r["adapter"], r["start"])
                cell = " ".join(x for x in parts if x)
                out.append(md_row([cell, *(f"{float(r[f'lre_{n}']):.1f}" for n in names),
                                   f"{float(r['max_abs_diff']):.1f}"]))  # fmt: skip
            out.append("")
        if changed:
            out += [
                md_row(
                    [
                        "Class change (part, suite, dataset, adapter, start)",
                        *(f"class {n}" for n in names),
                    ]
                ),  # fmt: skip
                "|" + "---|" * (len(names) + 1),
            ]
            for r in changed:
                cell = " ".join(
                    x for x in (r["part"], r["suite"], r["dataset"], r["adapter"], r["start"]) if x
                )
                out.append(md_row([cell, *(r[f"class_{n}"] for n in names)]))
            out.append("")
        else:
            out += ["No cell changes verdict class.", ""]
    return out


def sentences(names: list[str], rows: list[dict[str, str]]) -> list[str]:
    a = [r for r in rows if r["run"] == "quick" and r["part"] == "A"]
    b = [r for r in rows if r["run"] == "quick" and r["part"] == "B"]
    diffs = [float(r["max_abs_diff"]) for r in a]
    changed = sum(r["class_changes"] == "True" for r in a + b)
    first = (
        f"Across the {len(names)} platforms the {len(a):,} Part A cells of the quick run differ by "
        f"more than 0.5 digits in {sum(d > 0.5 for d in diffs)} cells and by more than 2 digits "
        f"in {sum(d > 2 for d in diffs)}; {sum(r['bitwise'] == 'True' for r in a):,} are bitwise "
        "identical"
    )
    parts = "Part A or Part B" if b else "Part A"
    second = (
        f"no {parts} cell changes verdict class"
        if changed == 0
        else f"{changed} {parts} cells change verdict class"
    )
    return [first, second]


def render(names: list[str], rows: list[dict[str, str]], folders: dict[str, Path]) -> str:
    out = [
        "# Platform spread",
        "",
        "Generated by `scripts/platform_spread.py` from the table folders of each platform; "
        "REPORT.md copies these rows. Do not edit by hand. Cell definitions: the script's "
        "docstring.",
        "",
        md_row(
            ["Platform", "CPU", "OS", "Python", "BLAS/LAPACK", "OpenBLAS kernel", "torch", "commit"]
        ),  # fmt: skip
        "|" + "---|" * 8,
        *(md_row([n, *_env(p)]) for n, p in folders.items()),
        "",
        *summary(names, rows),
        "## REPORT and README sentences",
        "",
        *(f"- {s}" for s in sentences(names, rows)),
    ]
    return "\n".join(out).rstrip("\n") + "\n"


def build() -> tuple[str, str]:
    """(spread.csv text, spread.md text) from the committed table folders."""
    names, rows = rows_all()
    folders = platforms()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields(names), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue(), render(names, rows, folders)


def main() -> None:
    csv_text, md_text = build()
    PLATFORMS.mkdir(parents=True, exist_ok=True)
    (PLATFORMS / "spread.csv").write_text(csv_text, encoding="utf-8", newline="")
    (PLATFORMS / "spread.md").write_text(md_text, encoding="utf-8", newline="\n")
    print(f"wrote {PLATFORMS / 'spread.csv'} and {PLATFORMS / 'spread.md'}")


if __name__ == "__main__":
    main()
