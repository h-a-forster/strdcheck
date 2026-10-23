"""Command-line interface: ``strdcheck list | fetch | verify``.

``study`` and ``table`` arrive with milestones M7 and M8 (``docs/DESIGN.md`` section 8).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from strdcheck import __version__
from strdcheck.fetch import fetch, is_cached, resolve_data_dir, source_note
from strdcheck.manifest import SUITES, list_datasets


def _cmd_list(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    for d in list_datasets(args.suite, args.difficulty):
        mark = "cached" if is_cached(d, data_dir) else "-"
        print(f"{d.suite:<11} {d.name:<10} {d.difficulty:<8} n={d.n_obs:<6} {mark}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    names = args.names or [d.name for d in list_datasets(args.suite)]
    paths = fetch(names, args.data_dir)
    print(f"{len(paths)} file(s) verified in {resolve_data_dir(args.data_dir)}")
    print(source_note())
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    bad = [d.name for d in list_datasets(args.suite) if not is_cached(d, data_dir)]
    if bad:
        print(f"missing or altered: {', '.join(bad)}", file=sys.stderr)
        return 1
    print("all files present and verified")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strdcheck", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=f"strdcheck {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-dir", default=None, help="StRD cache directory")
        p.add_argument("--suite", choices=SUITES, default=None)

    p = sub.add_parser("list", help="list the StRD datasets and their cache state")
    common(p)
    p.add_argument("--difficulty", choices=("lower", "average", "higher"), default=None)
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("fetch", help="download and verify StRD files from NIST")
    common(p)
    p.add_argument("names", nargs="*", help="dataset names (default: all in --suite)")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("verify", help="check cached files against the manifest checksums")
    common(p)
    p.set_defaults(func=_cmd_verify)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
