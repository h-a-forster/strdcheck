"""Synthetic files in NIST StRD layout (written for these tests; no NIST data).

Each builder returns ``(text, DatasetInfo)``. The header occupies lines 1-60 and data start
at line 61, as in the real files; certified blocks use the same wording as NIST.
"""

from __future__ import annotations

from collections.abc import Sequence

from strdcheck.manifest import DatasetInfo


def _info(name: str, suite: str, n: int) -> DatasetInfo:
    return DatasetInfo(
        name=name, suite=suite, url="", sha256="0" * 64, bytes=0, difficulty="lower", n_obs=n
    )


def _layout(header: list[str], data_header: str, rows: Sequence[str]) -> str:
    first, last = 61, 60 + len(rows)
    header = [
        "NIST/ITL StRD",
        "Dataset Name:  Synthetic",
        "",
        "File Format:   ASCII",
        f"               Data              (lines {first} to {last})",
        "",
        *header,
    ]
    if len(header) > 59:
        raise ValueError("header too long")
    lines = header + [""] * (59 - len(header)) + [data_header, *rows]
    return "\r\n".join(lines) + "\r\n"


def univariate(
    values: Sequence[str],
    mean: str,
    sd: str,
    r1: str,
    *,
    exact: bool = False,
    name: str = "NumAcc1",
) -> tuple[str, DatasetInfo]:
    tag = " (exact)" if exact else ""
    header = [
        "Model:         Lower Level of Difficulty",
        "               y    = mu + e",
        "",
        "                                                  Certified Values",
        f"Sample Mean                                ybar:  {mean}{tag}",
        f"Sample Standard Deviation (denom. = n-1)      s:  {sd}{tag}",
        f"Sample Autocorrelation Coefficient (lag 1) r(1):  {r1}{tag}",
        "",
        f"Number of Observations:                           {len(values)}",
    ]
    # Univariate files state the range as "Data : lines a to b (= n)", put "Data: Y" on
    # line 59 and a dashed rule on line 60.
    top = [
        "NIST/ITL StRD",
        "Dataset Name:  Synthetic",
        "",
        "File Format:   ASCII",
        f"               Data            : lines 61 to {60 + len(values)} (= {len(values)})",
        "",
        *header,
    ]
    lines = top + [""] * (58 - len(top)) + ["Data: Y", "---------", *[f"  {v}" for v in values]]
    return "\r\n".join(lines), _info(name, "univariate", len(values))


def anova(
    groups: Sequence[int],
    values: Sequence[str],
    cert: dict[str, str],
    df: tuple[int, int],
    name: str = "SiRstv",
) -> tuple[str, DatasetInfo]:
    header = [
        "Model:          k Parameters",
        "                y_{ij} = mu + tau_i + epsilon_{ij}",
        "",
        "Certified Values:",
        "",
        "Source of                  Sums of               Mean",
        "Variation          df      Squares              Squares             F Statistic",
        "",
        f"Between Instrument {df[0]:>2} {cert['ss_between']} {cert['ms_between']} {cert['f']}",
        f"Within Instrument  {df[1]:>2} {cert['ss_within']} {cert['ms_within']}",
        "",
        f"                   Certified R-Squared {cert['r2']}",
        "",
        "                   Certified Residual",
        f"                   Standard Deviation  {cert['resid_sd']}",
    ]
    rows = [f"           {g}         {v}" for g, v in zip(groups, values, strict=True)]
    return _layout(header, "Data:  Instrument   Resistance", rows), _info(name, "anova", len(rows))


def linear(
    y: Sequence[str],
    x: Sequence[str],
    params: dict[str, tuple[str, str]],
    resid_sd: str,
    r2: str,
    name: str = "Norris",
) -> tuple[str, DatasetInfo]:
    header = [
        "Model:         Linear Class",
        "               y = B0 + B1*x + e",
        "",
        "               Certified Regression Statistics",
        "",
        "                                          Standard Deviation",
        "     Parameter          Estimate             of Estimate",
        "",
        *[f"        {k.upper():<6}{est:>24}  {se:>24}" for k, (est, se) in params.items()],
        "",
        "     Residual",
        f"     Standard Deviation   {resid_sd}",
        "",
        f"     R-Squared            {r2}",
    ]
    rows = [f"        {a}    {b}" for a, b in zip(y, x, strict=True)]
    return _layout(header, "Data:     y     x", rows), _info(name, "linear", len(rows))


def nonlinear(
    y: Sequence[str],
    x: Sequence[str],
    params: list[tuple[str, str, str, str]],
    rss: str,
    resid_sd: str,
    name: str = "Misra1a",
) -> tuple[str, DatasetInfo]:
    header = [
        "Model:         Exponential Class",
        "               y = b1*(1-exp[-b2*x])  +  e",
        "",
        "          Starting values                  Certified Values",
        "",
        "        Start 1     Start 2           Parameter     Standard Deviation",
        *[f"  b{j + 1} =   {s1}   {s2}   {p}  {se}" for j, (s1, s2, p, se) in enumerate(params)],
        "",
        f"Residual Sum of Squares:                    {rss}",
        f"Residual Standard Deviation:                {resid_sd}",
        f"Degrees of Freedom:                         {len(y) - len(params)}",
        f"Number of Observations:                     {len(y)}",
    ]
    rows = [f"      {a}      {b}" for a, b in zip(y, x, strict=True)]
    return _layout(header, "Data:   y               x", rows), _info(name, "nonlinear", len(rows))
