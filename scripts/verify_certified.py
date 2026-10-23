"""V1 (docs/DESIGN.md section 9): recompute every certified value and write the ceilings.

For univariate, ANOVA and linear datasets the certified values are recomputed in exact
rational arithmetic from the decimal data (``dec``) and from the float64- and
float32-rounded data (the representation ceilings). For nonlinear problems the RSS and SEs
are evaluated at the certified parameters (``at_bstar``) and the parameters are refined by
Gauss-Newton in mpmath on the decimal and float64-rounded data.

Usage:  uv run python scripts/verify_certified.py [--out results/certified_check.csv]
Exit status 1 if any univariate/ANOVA/linear value falls below ``cap - 1`` or a nonlinear
refinement does not reproduce a certified value to 10 digits.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

from strdcheck import exact
from strdcheck.manifest import all_datasets
from strdcheck.parse import load


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/certified_check.csv")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    rows: list[dict[str, object]] = []
    bad: list[str] = []
    for info in all_datasets():
        t0 = time.perf_counter()
        ds = load(info.name, args.data_dir)
        if ds.suite == "nonlinear":
            ceil = exact.nonlinear_ceilings(ds)
            at = exact.nonlinear_at(ds, [ds.certified[p].text for p in ds.param_names()])
        else:
            ceil = exact.ceilings(ds)
            at = {}
        for name, cert in ds.certified.items():
            c = ceil.get(name)
            if c is None:
                rows.append(
                    {
                        "dataset": ds.name,
                        "suite": ds.suite,
                        "quantity": name,
                        "certified": cert.text,
                        "cap": cert.cap,
                        "note": "not finite",
                    }
                )
                continue
            at_b = exact.lre_decimal(at[name], cert)[1] if name in at else math.nan
            limit = 10 if ds.suite == "nonlinear" else cert.cap - 1
            ok = c.dec >= limit
            if not ok:
                bad.append(f"{ds.name}.{name}")
            rows.append(
                {
                    "dataset": ds.name,
                    "suite": ds.suite,
                    "quantity": name,
                    "certified": cert.text,
                    "cap": cert.cap,
                    "lre_dec": round(c.dec, 3),
                    "ceiling_f64": round(c.f64, 3),
                    "ceiling_f32": "" if math.isnan(c.f32) else round(c.f32, 3),
                    "lre_at_bstar": "" if math.isnan(at_b) else round(at_b, 3),
                    "note": "" if ok else "below limit",
                }
            )
        print(f"{ds.name:<10} {time.perf_counter() - t0:6.2f}s", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "suite",
        "quantity",
        "certified",
        "cap",
        "lre_dec",
        "ceiling_f64",
        "ceiling_f32",
        "lre_at_bstar",
        "note",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} certified values -> {out}; below limit: {bad or 'none'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
