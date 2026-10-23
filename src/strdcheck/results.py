"""Result schema (``docs/DESIGN.md`` section 5.5).

An adapter returns an :class:`EngineResult` (raw computed values). :func:`score` turns it
into :class:`QuantityRecord` rows (LRE against the certified value, ceiling, deficit), and a
finished unit of work is stored as one :class:`UnitRecord` JSON file by
:mod:`strdcheck.checkpoint`.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from strdcheck.lre import lre


@dataclass
class EngineResult:
    """What an adapter computed.

    Attributes:
        values: computed values keyed by certified quantity name.
        derived: names computed by strdcheck's shared float64 routine rather than the engine
            (DESIGN 4.4), reported separately.
        status: engine status (``success``, ``message``, ``nfev``, ``rank``, ...).
        warnings: warning messages raised by the engine call.
        seconds: wall time of the call (metadata only; never compared).
    """

    values: dict[str, float]
    derived: set[str] = field(default_factory=set)
    status: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def digest(self) -> str:
        """Hash of the bit patterns of all values (bitwise reproducibility check)."""
        h = hashlib.sha256()
        for k in sorted(self.values):
            h.update(k.encode())
            h.update(struct.pack("<d", float(self.values[k])))
        return h.hexdigest()[:16]


@dataclass
class QuantityRecord:
    name: str
    computed: float
    certified: str
    cap: int
    lre_raw: float
    lre: float
    ceiling: float  # ceiling for the adapter's dtype (nan if unknown)
    deficit: float  # max(ceiling - lre, 0); nan if either is nan
    derived: bool = False


@dataclass
class UnitRecord:
    unit_id: str
    part: str
    dataset: str
    adapter: str
    suite: str
    status: str  # "ok" | "error" | "skipped"
    start: str | None = None
    error: str | None = None
    quantities: list[QuantityRecord] = field(default_factory=list)
    engine_status: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    repeat_hashes: list[str] = field(default_factory=list)
    seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def repeat_identical(self) -> bool | None:
        if len(self.repeat_hashes) < 2:
            return None
        return len(set(self.repeat_hashes)) == 1

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(_jsonable(d), indent=1, sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> UnitRecord:
        d = _unjson(json.loads(text))
        d["quantities"] = [QuantityRecord(**q) for q in d.get("quantities", [])]
        return cls(**d)


def _jsonable(x: Any) -> Any:
    """Floats that JSON cannot carry (nan, inf) become strings; sets become sorted lists."""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted(_jsonable(v) for v in x)
    if hasattr(x, "item") and callable(x.item):  # numpy scalar
        return _jsonable(x.item())
    return x


_FLOAT_FIELDS = {"computed", "lre_raw", "lre", "ceiling", "deficit", "seconds"}


def _unjson(x: Any, key: str = "") -> Any:
    if isinstance(x, dict):
        return {k: _unjson(v, k) for k, v in x.items()}
    if isinstance(x, list):
        return [_unjson(v, key) for v in x]
    if key in _FLOAT_FIELDS and isinstance(x, str):
        return float(x)
    return x


def score(
    certified: Mapping[str, Any],
    result: EngineResult,
    ceilings: Mapping[str, float] | None = None,
) -> list[QuantityRecord]:
    """LRE rows for every certified quantity the adapter returned.

    ``certified`` maps names to :class:`strdcheck.parse.Certified` objects (anything with
    ``text``, ``value`` and ``cap``); quantities whose certified value is not finite are
    skipped. ``ceilings`` maps names to the ceiling for the adapter's dtype.
    """
    rows: list[QuantityRecord] = []
    ceilings = ceilings or {}
    for name, value in result.values.items():
        cert = certified.get(name)
        if cert is None or not math.isfinite(cert.value):
            continue
        x = float(value)
        r = lre(x, cert.value, cap=cert.cap)
        ceil = float(ceilings.get(name, math.nan))
        unknown = math.isnan(ceil) or math.isnan(r.value)
        deficit = math.nan if unknown else max(ceil - r.value, 0.0)
        rows.append(
            QuantityRecord(
                name=name,
                computed=x,
                certified=cert.text,
                cap=cert.cap,
                lre_raw=r.raw,
                lre=r.value,
                ceiling=ceil,
                deficit=deficit,
                derived=name in result.derived,
            )
        )
    return rows
