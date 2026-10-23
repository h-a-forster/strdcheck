"""Parser for NIST StRD ``.dat`` files (``docs/DESIGN.md`` section 5.2).

A file is read as bytes, decoded as ASCII and split into lines (``\\r`` stripped). The data
rows are taken from the line range stated in the header (``Data ... lines a to b``); the row
count must equal the manifest's ``n_obs`` and every row must have the same number of fields
as the column header, otherwise :class:`ParseError` is raised. Certified values are parsed
with one set of patterns per suite and kept verbatim (``Certified.text``) so that LRE caps
can be derived from NIST's printed precision.

Quantity names (``Certified.name``):

- univariate: ``mean``, ``sd``, ``r1``
- ANOVA: ``ss_between``, ``ms_between``, ``f``, ``ss_within``, ``ms_within``, ``r2``,
  ``resid_sd``
- linear: ``b0`` ... ``bk`` (as NIST numbers them), ``se_b0`` ..., ``resid_sd``, ``r2``
- nonlinear: ``b1`` ... ``bp``, ``se_b1`` ..., ``rss``, ``resid_sd``
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from strdcheck.lre import EXACT_CAP, significant_digits
from strdcheck.manifest import DatasetInfo, get_dataset_info


class ParseError(ValueError):
    """The file does not have the layout of a NIST StRD file."""


@dataclass(frozen=True, slots=True)
class Certified:
    """One certified value as printed by NIST."""

    name: str
    text: str
    value: float
    cap: int
    exact: bool = False

    @classmethod
    def from_text(cls, name: str, text: str, *, exact: bool = False) -> Certified:
        value = float(text)
        cap = EXACT_CAP if exact or value == 0.0 else significant_digits(text)
        return cls(name=name, text=text, value=value, cap=cap, exact=exact)


@dataclass(frozen=True, slots=True)
class LinearSpec:
    """Design of a linear StRD model.

    ``degree`` is set for polynomial models in one predictor ``x`` (columns ``x^0..x^degree``
    or ``x^1..x^degree`` without intercept); ``predictors`` lists the columns of a multiple
    regression (Longley).
    """

    intercept: bool
    degree: int | None = None
    predictors: tuple[str, ...] = ()

    @property
    def n_params(self) -> int:
        if self.degree is not None:
            return self.degree + (1 if self.intercept else 0)
        return len(self.predictors) + (1 if self.intercept else 0)

    def param_names(self) -> tuple[str, ...]:
        start = 0 if self.intercept else 1
        return tuple(f"b{j}" for j in range(start, start + self.n_params))


LINEAR_SPECS: dict[str, LinearSpec] = {
    "Norris": LinearSpec(intercept=True, degree=1),
    "Pontius": LinearSpec(intercept=True, degree=2),
    "NoInt1": LinearSpec(intercept=False, degree=1),
    "NoInt2": LinearSpec(intercept=False, degree=1),
    "Filip": LinearSpec(intercept=True, degree=10),
    "Longley": LinearSpec(intercept=True, predictors=("x1", "x2", "x3", "x4", "x5", "x6")),
    **{f"Wampler{i}": LinearSpec(intercept=True, degree=5) for i in range(1, 6)},
}


@dataclass(frozen=True)
class Dataset:
    """A parsed StRD file."""

    info: DatasetInfo
    model: str
    columns: tuple[str, ...]
    raw: dict[str, tuple[str, ...]]
    certified: dict[str, Certified]
    n_params: int
    starts: tuple[tuple[float, ...], ...] = ()
    start_texts: tuple[tuple[str, ...], ...] = ()
    df: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def suite(self) -> str:
        return self.info.suite

    @property
    def n(self) -> int:
        return len(next(iter(self.raw.values())))

    def array(self, column: str, dtype: npt.DTypeLike = np.float64) -> np.ndarray:
        """Column as an array: each string is parsed by ``float`` (correct rounding), then cast."""
        values = np.array([float(s) for s in self.raw[column]], dtype=np.float64)
        return values.astype(dtype, copy=False)

    @property
    def response(self) -> str:
        return self.columns[-1] if self.suite == "anova" else self.columns[0]

    def linear_spec(self) -> LinearSpec:
        if self.suite != "linear":
            raise ValueError(f"{self.name} is not a linear dataset")
        return LINEAR_SPECS[self.name]

    def param_names(self) -> tuple[str, ...]:
        """Certified parameter names in order (linear and nonlinear)."""
        if self.suite == "linear":
            return self.linear_spec().param_names()
        if self.suite == "nonlinear":
            return tuple(f"b{j}" for j in range(1, self.n_params + 1))
        return ()


_RANGE = re.compile(r"Data\s*[:(]?\s*lines\s+(\d+)\s+to\s+(\d+)", re.IGNORECASE)
_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"


def _data_range(lines: list[str]) -> tuple[int, int]:
    for line in lines[:15]:
        m = _RANGE.search(line)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise ParseError("no 'Data (lines a to b)' entry in the header")


def _columns(lines: list[str], first_data: int) -> tuple[str, ...]:
    for line in reversed(lines[: first_data - 1]):
        if line.startswith("Data:"):
            cols = tuple(c.lower() for c in line[5:].split())
            if cols:
                return cols
    raise ParseError("no 'Data:' column header before the data block")


def _model(lines: list[str]) -> str:
    out: list[str] = []
    capture = False
    for line in lines[:60]:
        if line.startswith("Model:"):
            capture = True
        elif capture and line and not line.startswith(" "):
            break
        if capture:
            out.append(line.strip())
    return " ".join(s for s in out if s)


def _search(pattern: str, text: str, what: str) -> re.Match[str]:
    m = re.search(pattern, text, re.MULTILINE)
    if m is None:
        raise ParseError(f"certified value not found: {what}")
    return m


def _univariate(header: str) -> dict[str, Certified]:
    out = {}
    for name, label in (("mean", r"ybar:"), ("sd", r"\bs:"), ("r1", r"r\(1\):")):
        m = _search(rf"{label}\s+({_NUM})\s*(\(exact\))?", header, name)
        out[name] = Certified.from_text(name, m.group(1), exact=m.group(2) is not None)
    return out


def _anova(header: str) -> tuple[dict[str, Certified], dict[str, int]]:
    b = _search(rf"^Between\s+\S+\s+(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})", header, "between")
    w = _search(rf"^Within\s+\S+\s+(\d+)\s+({_NUM})\s+({_NUM})", header, "within")
    r2 = _search(rf"Certified R-Squared\s+({_NUM})", header, "r2")
    sd = _search(rf"Standard Deviation\s+({_NUM})", header, "resid_sd")
    texts = {
        "ss_between": b.group(2),
        "ms_between": b.group(3),
        "f": b.group(4),
        "ss_within": w.group(2),
        "ms_within": w.group(3),
        "r2": r2.group(1),
        "resid_sd": sd.group(1),
    }
    cert = {k: Certified.from_text(k, v) for k, v in texts.items()}
    return cert, {"between": int(b.group(1)), "within": int(w.group(1))}


def _linear(header: str, spec: LinearSpec) -> dict[str, Certified]:
    out: dict[str, Certified] = {}
    for m in re.finditer(rf"^\s+B(\d+)\s+({_NUM})\s+({_NUM})\s*$", header, re.MULTILINE):
        j = int(m.group(1))
        out[f"b{j}"] = Certified.from_text(f"b{j}", m.group(2))
        out[f"se_b{j}"] = Certified.from_text(f"se_b{j}", m.group(3))
    names = spec.param_names()
    if sorted(k for k in out if not k.startswith("se_")) != sorted(names):
        raise ParseError(f"expected parameters {names}, found {sorted(out)}")
    sd = _search(rf"Residual\s*\n\s*Standard Deviation\s+({_NUM})", header, "resid_sd")
    r2 = _search(rf"R-Squared\s+({_NUM})", header, "r2")
    out["resid_sd"] = Certified.from_text("resid_sd", sd.group(1))
    out["r2"] = Certified.from_text("r2", r2.group(1))
    return out


def _nonlinear(
    header: str,
) -> tuple[dict[str, Certified], tuple[tuple[str, ...], ...], dict[str, int]]:
    out: dict[str, Certified] = {}
    s1: list[str] = []
    s2: list[str] = []
    pat = rf"^\s+b(\d+)\s*=\s*({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s*$"
    for k, m in enumerate(re.finditer(pat, header, re.MULTILINE), start=1):
        j = int(m.group(1))
        if j != k:
            raise ParseError(f"parameter b{j} out of order")
        s1.append(m.group(2))
        s2.append(m.group(3))
        out[f"b{j}"] = Certified.from_text(f"b{j}", m.group(4))
        out[f"se_b{j}"] = Certified.from_text(f"se_b{j}", m.group(5))
    if not s1:
        raise ParseError("no parameter lines found")
    rss = _search(rf"Residual Sum of Squares:\s+({_NUM})", header, "rss")
    sd = _search(rf"Residual Standard Deviation:\s+({_NUM})", header, "resid_sd")
    dof = _search(r"Degrees of Freedom:\s+(\d+)", header, "dof")
    out["rss"] = Certified.from_text("rss", rss.group(1))
    out["resid_sd"] = Certified.from_text("resid_sd", sd.group(1))
    return out, (tuple(s1), tuple(s2)), {"residual": int(dof.group(1))}


def parse_text(text: str, info: DatasetInfo) -> Dataset:
    """Parse the content of an StRD file described by ``info``."""
    lines = text.replace("\r", "").split("\n")
    first, last = _data_range(lines)
    if last < first or last > len(lines):
        raise ParseError(f"data range {first}-{last} outside the file ({len(lines)} lines)")
    columns = _columns(lines, first)
    rows = [lines[i - 1].split() for i in range(first, last + 1)]
    rows = [r for r in rows if r]
    if len(rows) != info.n_obs:
        raise ParseError(f"{info.name}: {len(rows)} data rows, manifest says {info.n_obs}")
    for i, r in enumerate(rows):
        if len(r) != len(columns):
            raise ParseError(
                f"{info.name}: data row {i + 1} has {len(r)} fields, expected {columns}"
            )
        for s in r:
            if not re.fullmatch(_NUM, s):
                raise ParseError(f"{info.name}: data row {i + 1}: not a number: {s!r}")
    raw = {c: tuple(r[j] for r in rows) for j, c in enumerate(columns)}
    header = "\n".join(lines[: first - 1])
    model = _model(lines)
    starts_t: tuple[tuple[str, ...], ...] = ()
    df: dict[str, int] = {}
    if info.suite == "univariate":
        cert = _univariate(header)
        n_params = 2
    elif info.suite == "anova":
        cert, df = _anova(header)
        n_params = df["between"] + 1
    elif info.suite == "linear":
        spec = LINEAR_SPECS.get(info.name)
        if spec is None:
            raise ParseError(f"no linear model specification for {info.name}")
        cert = _linear(header, spec)
        n_params = spec.n_params
        df = {"residual": info.n_obs - n_params}
    elif info.suite == "nonlinear":
        cert, starts_t, df = _nonlinear(header)
        n_params = len(starts_t[0])
        # NIST prints the residual degrees of freedom; Rat43 prints 9 although n - p = 11
        # (its certified residual SD uses 11). Keep the printed value for the record and use
        # n - p everywhere.
        df = {"residual": info.n_obs - n_params, "printed_residual": df["residual"]}
    else:  # pragma: no cover - manifest guarantees the suite
        raise ParseError(f"unknown suite {info.suite!r}")
    return Dataset(
        info=info,
        model=model,
        columns=columns,
        raw=raw,
        certified=cert,
        n_params=n_params,
        starts=tuple(tuple(float(v) for v in s) for s in starts_t),
        start_texts=starts_t,
        df=df,
    )


def parse_file(path: str | os.PathLike[str], info: DatasetInfo | None = None) -> Dataset:
    """Parse an StRD file; ``info`` defaults to the manifest entry named after the file."""
    p = Path(path)
    if info is None:
        info = get_dataset_info(p.stem)
    try:
        text = p.read_bytes().decode("ascii")
    except UnicodeDecodeError as exc:
        raise ParseError(f"{p.name}: not ASCII") from exc
    return parse_text(text, info)


def load(name: str, data_dir: str | os.PathLike[str] | None = None) -> Dataset:
    """Parse a cached, checksum-verified StRD file by dataset name."""
    from strdcheck.fetch import is_cached, resolve_data_dir

    info = get_dataset_info(name)
    d = resolve_data_dir(data_dir)
    if not is_cached(info, d):
        raise FileNotFoundError(
            f"{info.filename} is not in {d} (or fails its checksum); run `strdcheck fetch`"
        )
    return parse_file(d / info.filename, info)
