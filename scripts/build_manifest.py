"""Build ``src/strdcheck/data/manifest.json`` from the NIST StRD files.

Developer tool. It downloads every StRD file used by strdcheck from
``https://www.itl.nist.gov/div898/strd/`` (or reads them from ``--from-dir``), records the
SHA-256 and size of each file together with a few header facts (difficulty, number of
observations), and writes the manifest. The raw files are never committed.

Usage::

    uv run python scripts/build_manifest.py --out-dir tmp/strd        # download
    uv run python scripts/build_manifest.py --from-dir tmp/strd       # reuse a download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

BASE_URL = "https://www.itl.nist.gov/div898/strd/"

SUITES: dict[str, tuple[str, list[str]]] = {
    "univariate": (
        "univ/data/",
        ["PiDigits", "Lottery", "Lew", "Mavro", "Michelso"] + [f"NumAcc{i}" for i in range(1, 5)],
    ),
    "anova": (
        "anova/",
        ["SiRstv"] + [f"SmLs{i:02d}" for i in range(1, 10)] + ["AtmWtAg"],
    ),
    "linear": (
        "lls/data/LINKS/DATA/",
        ["Norris", "Pontius", "NoInt1", "NoInt2", "Filip", "Longley"]
        + [f"Wampler{i}" for i in range(1, 6)],
    ),
    "nonlinear": (
        "nls/data/LINKS/DATA/",
        [
            "Misra1a",
            "Chwirut2",
            "Chwirut1",
            "Lanczos3",
            "Gauss1",
            "Gauss2",
            "DanWood",
            "Misra1b",
            "Kirby2",
            "Hahn1",
            "Nelson",
            "MGH17",
            "Lanczos1",
            "Lanczos2",
            "Gauss3",
            "Misra1c",
            "Misra1d",
            "Roszman1",
            "ENSO",
            "MGH09",
            "Thurber",
            "BoxBOD",
            "Rat42",
            "MGH10",
            "Eckerle4",
            "Rat43",
            "Bennett5",
        ],
    ),
}

_DIFFICULTY = re.compile(r"(Lower|Average|Higher)\s+Level\s+of\s+Difficulty", re.IGNORECASE)
_NOBS = re.compile(r"(\d+)\s+Observations|Number of Observations:\s+(\d+)")
# Michelso.dat has no difficulty line; the StRD univariate index page lists it as "lower".
_DIFFICULTY_OVERRIDE = {"Michelso": "lower"}


def _download(url: str, timeout: float = 60.0, attempts: int = 3) -> bytes:
    # strdcheck's opener: NIST-only prefix check and a User-Agent (NIST answers 403 to the
    # default "Python-urllib" agent).
    from strdcheck.fetch import _urlopen

    last: Exception | None = None
    for i in range(attempts):
        try:
            return _urlopen(url, timeout)
        except Exception as exc:  # pragma: no cover - network
            last = exc
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"download failed after {attempts} attempts: {url}") from last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-dir", type=Path, help="directory that already holds the .dat files")
    src.add_argument("--out-dir", type=Path, help="download the files into this directory")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src/strdcheck/data/manifest.json",
    )
    ap.add_argument(
        "--retrieved",
        default=None,
        help="retrieval date YYYY-MM-DD for --from-dir (default: today)",
    )
    args = ap.parse_args(argv)

    entries = []
    for suite, (subdir, names) in SUITES.items():
        for name in names:
            url = f"{BASE_URL}{subdir}{name}.dat"
            if args.from_dir is not None:
                raw = (args.from_dir / f"{name}.dat").read_bytes()
            else:
                raw = _download(url)
                args.out_dir.mkdir(parents=True, exist_ok=True)
                (args.out_dir / f"{name}.dat").write_bytes(raw)
            text = raw.decode("ascii")
            diff = _DIFFICULTY.search(text)
            nobs = _NOBS.search(text)
            entries.append(
                {
                    "name": name,
                    "suite": suite,
                    "url": url,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "difficulty": (
                        diff.group(1).lower() if diff else _DIFFICULTY_OVERRIDE.get(name)
                    ),
                    "n_obs": int(nobs.group(1) or nobs.group(2)) if nobs else None,
                }
            )
    manifest = {
        "source": "NIST Statistical Reference Datasets (StRD), SRD 140",
        "doi": "10.18434/T43G6C",
        "base_url": BASE_URL,
        "retrieved": args.retrieved or time.strftime("%Y-%m-%d"),
        "datasets": entries,
    }
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries to {args.manifest.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
