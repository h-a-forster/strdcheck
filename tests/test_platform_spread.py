"""scripts/platform_spread.py: the committed spread files are its output, REPORT.md and README.md
copy its sentences, and its cell comparison finds differences and class changes."""

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPREAD = ROOT / "results" / "platforms"


def _script():
    spec = importlib.util.spec_from_file_location(
        "platform_spread", ROOT / "scripts" / "platform_spread.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _words(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def test_spread_files_render_from_the_committed_tables(monkeypatch):
    monkeypatch.chdir(ROOT)
    csv_text, md_text = _script().build()
    assert (SPREAD / "spread.csv").read_text(encoding="utf-8") == csv_text
    assert (SPREAD / "spread.md").read_text(encoding="utf-8") == md_text


def test_report_and_readme_state_the_rendered_spread():
    lines = (SPREAD / "spread.md").read_text(encoding="utf-8").split("## REPORT and README")[1]
    sentences = [line[2:] for line in lines.splitlines() if line.startswith("- ")]
    assert len(sentences) == 2
    report = _words(ROOT / "REPORT.md")
    assert [s for s in sentences if s not in report] == []
    assert sentences[1] in _words(ROOT / "README.md")
    assert "One platform." not in report


def _table(folder: Path, rows: list[dict[str, str]]) -> None:
    folder.mkdir(parents=True)
    fields = ["suite", "dataset", "adapter", "role", "dtype", "start", "quantity", "derived",
              "computed", "lre", "ceiling", "success", "status"]  # fmt: skip
    with (folder / "lre_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_compare_finds_digit_differences_and_class_changes(tmp_path):
    def unit(lre_b1: str, success: str, computed: str) -> list[dict[str, str]]:
        base = {"suite": "nonlinear", "dataset": "Misra1a", "adapter": "a", "role": "engine",
                "dtype": "f64", "start": "S1", "derived": "False", "ceiling": "15.0",
                "success": success, "status": "ok"}  # fmt: skip
        return [
            {**base, "quantity": "b1", "computed": computed, "lre": lre_b1},
            {**base, "quantity": "b2", "computed": "2.0", "lre": "9.0"},
            {**base, "quantity": "resid_sd", "derived": "True", "computed": "1.0", "lre": "nan"},
        ]

    lin = {"suite": "linear", "dataset": "Filip", "adapter": "l", "role": "engine",
           "dtype": "f64", "start": "", "derived": "False", "computed": "1.0", "ceiling": "8.0",
           "success": "", "status": "ok", "quantity": "b0"}  # fmt: skip
    _table(tmp_path / "x", [*unit("8.0", "True", "1.0"), {**lin, "lre": "7.0"}])
    _table(tmp_path / "y", [*unit("3.5", "True", "1.1"), {**lin, "lre": "6.5"}])
    rows = _script().compare("quick", {"x": tmp_path / "x", "y": tmp_path / "y"})
    by = {r["suite"]: r for r in rows}
    fit, ls = by["nonlinear"], by["linear"]
    assert (fit["class_x"], fit["class_y"], fit["class_changes"]) == ("ok", "silent", "True")
    assert (fit["lre_x"], fit["lre_y"], fit["max_abs_diff"], fit["bitwise"]) == (
        "8.0000", "3.5000", "4.5000", "False")  # fmt: skip
    assert (ls["class_x"], ls["class_y"], ls["max_abs_diff"]) == ("pass", "pass", "0.5000")
    assert ls["bitwise"] == "True"
