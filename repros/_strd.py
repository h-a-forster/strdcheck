"""Minimal StRD reader shared by the repro scripts (no strdcheck imports).

Data rows start at line 61 of every StRD file; columns are whitespace-separated.
"""

from __future__ import annotations

import sys
from pathlib import Path


def rows(name: str) -> list[list[str]]:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    lines = (d / f"{name}.dat").read_text("ascii").splitlines()
    return [ln.split() for ln in lines[60:] if ln.strip()]
