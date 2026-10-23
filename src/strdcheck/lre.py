"""Log relative error (LRE): the number of correct significant digits.

Definitions follow McCullough (1998), "Assessing the reliability of statistical software:
Part I", *The American Statistician* 52(4), 358-366. Specification and conventions:
``docs/DESIGN.md`` section 4.1.

For a computed value ``x`` and a certified value ``c``::

    LRE = -log10(|x - c| / |c|)    if c != 0   (log relative error)
    LRE = -log10(|x - c|)          if c == 0   (log absolute error, "LAE")

The *raw* LRE is unbounded (``inf`` when ``x == c``, negative when ``x`` is far off). The
*reported* LRE is clipped to ``[0, cap]``, where ``cap`` is the number of significant digits
of the certified value (15 for an exact certified value), and values below 1 are reported
as 0 because not even the leading digit is right.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

EXACT_CAP = 15
"""Cap used for certified values that NIST marks ``(exact)`` and for certified zeros."""

_NUMBER = re.compile(r"^\s*[+-]?(?P<int>\d*)(?:\.(?P<frac>\d*))?(?:[eEdD](?P<exp>[+-]?\d+))?\s*$")


def significant_digits(certified: str) -> int:
    """Number of significant digits in a certified value as printed by NIST.

    Leading zeros (including those after the decimal point) are not significant; trailing
    zeros are, because NIST prints certified values to their full certified precision (for
    example ``4.53480000000000`` has 15). A value whose digits are all zero returns 0; the
    caller decides the cap for certified zeros (``EXACT_CAP``).
    """
    m = _NUMBER.match(certified)
    if m is None or (not m.group("int") and not m.group("frac")):
        raise ValueError(f"not a decimal number: {certified!r}")
    digits = (m.group("int") or "") + (m.group("frac") or "")
    return len(digits.lstrip("0"))


def raw_lre(computed: float, certified: float) -> float:
    """Unclipped LRE (``nan`` if ``computed`` is not finite, ``inf`` if exact)."""
    if not math.isfinite(certified):
        raise ValueError(f"certified value must be finite, got {certified!r}")
    if not math.isfinite(computed):
        return math.nan
    err = abs(computed - certified)
    if err == 0.0:
        return math.inf
    if certified == 0.0:
        return -math.log10(err)
    return -math.log10(err / abs(certified))


def clip_lre(raw: float, cap: int) -> float:
    """Reported LRE: ``raw`` clipped to ``[0, cap]``; below 1 becomes 0; ``nan`` stays."""
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")
    if math.isnan(raw):
        return math.nan
    if raw < 1.0:
        return 0.0
    return float(min(raw, cap))


@dataclass(frozen=True, slots=True)
class LRE:
    """One LRE comparison.

    Attributes:
        raw: unclipped LRE (may be ``inf``, negative or ``nan``).
        value: reported LRE in ``[0, cap]`` (``nan`` when the computed value is not finite).
        cap: maximum attainable LRE for this certified value.
        absolute: True when the certified value is zero and the log absolute error was used.
    """

    raw: float
    value: float
    cap: int
    absolute: bool


def lre(computed: float, certified: float, *, cap: int = EXACT_CAP) -> LRE:
    """LRE of ``computed`` against ``certified``, reported to at most ``cap`` digits."""
    computed = float(computed)
    certified = float(certified)
    raw = raw_lre(computed, certified)
    return LRE(raw=raw, value=clip_lre(raw, cap), cap=cap, absolute=certified == 0.0)


def lre_from_string(computed: float, certified: str, *, exact: bool = False) -> LRE:
    """LRE against a certified value given as the NIST string (sets the cap from its digits)."""
    value = float(certified.replace("D", "E").replace("d", "e"))
    digits = significant_digits(certified)
    cap = EXACT_CAP if exact or value == 0.0 else digits
    return lre(computed, value, cap=cap)
