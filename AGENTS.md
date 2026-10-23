# AGENTS.md

Guide for coding agents (and humans) working on strdcheck. Read `docs/DESIGN.md` before
changing any formula, adapter or study setting. It holds the formulas, conventions,
validation strategy, study design and milestones this file points to.

## What this is

A library, pytest plugin and study that measure the numerical accuracy (log relative error,
LRE) of numpy, pandas, polars, DuckDB, statsmodels, scipy and PyTorch (CPU) on the NIST
Statistical Reference Datasets, reporting each LRE next to its float64/float32
representation ceiling, plus silent-failure rates of scipy's nonlinear least squares solvers.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11 (local dev pinned to 3.13 by
  `.python-version`), typed code, `ruff check` + `ruff format` (line length 100), pytest.
- Core dependency is numpy only. Engines live in the `engines` extra (and the dev group);
  torch in the `torch` extra, resolved from the PyTorch CPU index. Adapters import engines
  lazily inside `run`.
- LRE conventions: DESIGN 4.1 (cap from the certified string, 15 for `(exact)` and zeros,
  below 1 reported as 0, `nan` is a failure). Quantity names: DESIGN 5.2.
- Engine conventions that cause adapter mistakes (ddof defaults, pandas `autocorr` is not
  NIST r(1), no-intercept R² is uncentred, `lstsq` `rcond` truncation): DESIGN 4.2-4.4.
  Every adapter gets a synthetic exact-answer test before it is run on StRD.
- No LLM or model API calls. CPU only. At most 3 worker processes; BLAS, polars, torch and
  DuckDB threads set to 1 in workers except where a configuration sets threads explicitly.
- Long jobs: every finished unit is written by `strdcheck.checkpoint` (atomic file per unit,
  skip on rerun, `progress.log`), and the study runs inside the retry wrapper. Never keep
  results only in memory. Keep each run under 45 minutes; the full-size command is in
  DESIGN 6.7.
- Data: the StRD files are fetched from `https://www.itl.nist.gov/div898/strd/` and verified
  against `src/strdcheck/data/manifest.json`. Never commit them (licence question O1 in
  DESIGN 10). Commit only small derived results under `results/`, with local paths scrubbed.
- Tests: no network; synthetic StRD-format fixtures under `tests/fixtures/`; tests on the
  real files skip when the cache is empty; slow tests marked `@pytest.mark.slow`.
- Scratch files go in `tmp/` (gitignored). Do not create `.bak` copies; use git.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push and
  do not create remote repositories; publishing is done separately after review.
- Statistical honesty: report sizes, seeds, versions and intervals; only the five
  pre-specified comparisons of DESIGN 6.3 get p-values (Holm); never overstate. Anything that
  looks like an engine defect goes through the triage protocol (DESIGN 6.6): standalone
  repro, adapter ruled out, ceiling checked.

## Architecture map

| Path | Role | Milestone |
|---|---|---|
| `src/strdcheck/lre.py` | LRE/LAE, significant digits, clipping | done |
| `src/strdcheck/manifest.py`, `data/manifest.json` | 58 StRD files with URL, SHA-256, size, difficulty, n | done |
| `src/strdcheck/fetch.py` | NIST-only downloader with checksum and atomic cache writes | done |
| `src/strdcheck/cli.py` | `list`, `fetch`, `verify` (later `study`, `table`, `env`) | partial, M7 |
| `src/strdcheck/pytest_plugin.py` | `strd` marker, `strd_assert_lre`, options (later dataset fixtures, report) | partial, M6 |
| `src/strdcheck/parse.py` | StRD parser -> `Dataset` | M1 |
| `src/strdcheck/exact.py` | exact rational references and representation ceilings | M1 |
| `src/strdcheck/results.py`, `checkpoint.py` | result schema, resumable unit store | M2 |
| `src/strdcheck/engines/` | adapters per suite + registry | M3-M5 |
| `src/strdcheck/models.py` | 27 nonlinear models with analytic Jacobians | M5 |
| `src/strdcheck/testing.py` | public plugin helpers | M6 |
| `src/strdcheck/study/` | config, runner, analysis, report, tiled Part C | M7-M9 |
| `scripts/build_manifest.py` | regenerate the manifest from NIST | done |
| `configs/`, `results/` | study configs; committed small outputs | M7, M9 |

## How to test

```sh
uv sync --group dev                  # add --extra torch for torch adapters
uv run pytest                        # add --runslow for slow tests
uv run ruff check . && uv run ruff format --check .
uv run strdcheck fetch               # real StRD files into the user cache (needed for M1+ data tests)
```

## Status and next steps

Updated: 2026-10-24.

Done:
- M0 scaffold: uv project with `uv.lock` (numpy/scipy forks for Python 3.11 vs >= 3.12;
  torch 2.14.1+cpu from the PyTorch CPU index), CI, hygiene files, `docs/DESIGN.md`;
  manifest, downloader (sends its own User-Agent: NIST returns 403 to `Python-urllib`), LRE
  module, CLI `list/fetch/verify`, plugin stub.
- M3: univariate (15 engine configs + 2 controls) and ANOVA (7 + 2 controls) adapters in
  `engines/`, registry; synthetic exact-answer test per adapter; controls behave on real
  data. Found: the shared ANOVA `combine` step cancels in `m_g - m` (DESIGN 4.3 updated).
- M2: `results.py` (EngineResult, QuantityRecord, UnitRecord, `score`), `checkpoint.py`
  (RunStore, atomic unit files, progress.log, config-hash guard, `execute` with at most 3
  spawned workers, `with_retries`); kill-and-resume test passes.
- M1: `parse.py` (all 58 files parse; Rat43 header dof typo recorded), `exact.py` (exact
  rational references, float64/float32 ceilings, tiled references, mpmath Gauss-Newton
  refinement), `models.py` (27 models, analytic Jacobians checked against mpmath),
  `scripts/verify_certified.py` -> `results/certified_check.csv` (530 values; none below the
  limit; about 40 s).

Next (in order): M4-M5 adapters; M6 plugin; M7 runner and
quick study; M9 report. Cut order if short of time: DESIGN 8.

Open questions for the owner: O1 data licence (bundle the 2.1 MB of StRD files or keep
fetching). "Hill et al. 2024" not located yet (M9).
