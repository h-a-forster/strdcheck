"""REPORT.md publishes copies of tables rendered from the committed results, never typed numbers.

`scripts/report_tables.py` renders results/report_tables.md from the CSVs under results/ (Part B
rates, intervals and tests are recomputed from the per-problem counts with the study's own
code). CI reruns it and fails on a diff; these tests check the copies.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "report_tables.md"
# The run record (machine, wall time, versions) and file maps hold no results.
NOT_RESULTS = ("| Item |", "| Option |", "| Path |")


def _script():
    spec = importlib.util.spec_from_file_location(
        "report_tables", ROOT / "scripts" / "report_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result_rows(path: Path) -> list[str]:
    rows, skipping = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            skipping = False
        elif line.startswith(NOT_RESULTS):
            skipping = True
        elif line.startswith("| ") and not skipping:
            rows.append(line)
    return rows


def test_report_tables_render_from_the_committed_results(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert TABLES.read_text(encoding="utf-8") == _script().render()


def test_every_report_table_row_is_copied_from_the_rendered_tables():
    source = TABLES.read_text(encoding="utf-8").splitlines()
    rows = _result_rows(ROOT / "REPORT.md")
    assert len(rows) > 50
    assert [row for row in rows if row not in source] == []
    assert _result_rows(ROOT / "README.md") == []  # the README states results in prose only


def test_readme_key_results_are_the_rendered_figures():
    text = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    lines = TABLES.read_text(encoding="utf-8").split("## README: key results")[1].splitlines()
    fragments = [line[2:] for line in lines if line.startswith("- ")]
    assert len(fragments) == 5
    assert [f for f in fragments if f not in text] == []
