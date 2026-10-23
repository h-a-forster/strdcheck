"""Command-line interface: ``strdcheck list | fetch | verify | study | table | env``."""

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


def _cmd_study(args: argparse.Namespace) -> int:
    from strdcheck.checkpoint import RunStore

    if args.status:
        store = RunStore(args.run)
        if not (store.dir / "manifest.json").exists():
            print(f"no run in {args.run}", file=sys.stderr)
            return 1
        from strdcheck.study.config import StudyConfig, load_config
        from strdcheck.study.runner import build_units

        cfg = load_config(args.config) if args.config else None
        done = store.count()
        if cfg is None:
            print(f"{done} units done in {args.run}")
            return 0
        from strdcheck.fetch import resolve_data_dir

        units = build_units(cfg, args.run, str(resolve_data_dir(args.data_dir)))
        total = sum(len(v) for v in units.values()) if isinstance(cfg, StudyConfig) else 0
        print(f"{done}/{total} units done in {args.run}")
        return 0
    from strdcheck.study.config import load_config
    from strdcheck.study.runner import run_study

    if not args.config:
        print("--config is required", file=sys.stderr)
        return 2
    out = run_study(load_config(args.config), args.run, args.data_dir)
    print(f"{out['done']}/{out['total']} units done; ran now: {out['ran']}")
    return 0 if out["done"] == out["total"] else 1


def _cmd_table(args: argparse.Namespace) -> int:
    from strdcheck.study.analysis import write_results

    written = write_results(args.run, args.out)
    for p in written:
        print(p)
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    import json

    from strdcheck.env import environment_record

    text = json.dumps(environment_record(), indent=1, sort_keys=True)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
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

    p = sub.add_parser("study", help="run or resume a study (checkpointed; see configs/)")
    p.add_argument("--config", default=None, help="study config (TOML)")
    p.add_argument("--run", required=True, help="run directory")
    p.add_argument("--data-dir", default=None, help="StRD cache directory")
    p.add_argument("--status", action="store_true", help="print progress of the run and exit")
    p.set_defaults(func=_cmd_study)

    p = sub.add_parser("table", help="write results tables from a run directory")
    p.add_argument("run", help="run directory")
    p.add_argument("--out", default="results", help="output directory")
    p.set_defaults(func=_cmd_table)

    p = sub.add_parser("env", help="print the environment record (paths scrubbed)")
    p.add_argument("--out", default=None, help="write JSON to this file")
    p.set_defaults(func=_cmd_env)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
