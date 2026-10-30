# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-30

First release: library, pytest plugin and a published study.

### Added

- `strdcheck.manifest`: the 58 StRD files (univariate, ANOVA, linear and nonlinear
  regression) with source URLs, SHA-256 checksums, sizes, difficulty and number of
  observations.
- `strdcheck.fetch`: downloader restricted to the NIST StRD host, with timeouts, retries,
  checksum verification, atomic writes into a user cache and its own User-Agent.
- `strdcheck.lre`: log relative error, log absolute error for certified zeros, significant
  digit counting of NIST certified strings, clipping to `[0, cap]`.
- `strdcheck.parse`: parser for all 58 files (certified values kept verbatim, NIST starting
  values, row-count and field checks).
- `strdcheck.exact`: exact rational references from the decimal, float64-rounded and
  float32-rounded data; representation ceilings; exact references for tiled data; mpmath
  Gauss-Newton references for the nonlinear problems.
- `strdcheck.models`: the 27 nonlinear models with analytic Jacobians (checked against mpmath
  derivatives).
- Engine adapters (`strdcheck.engines`): 15 univariate, 7 ANOVA, 17 linear and 14 nonlinear
  configurations across numpy, pandas, polars, DuckDB, scipy, statsmodels, PyTorch (CPU) and
  the Python `statistics` module, plus positive and negative controls.
- Resumable study runner (`strdcheck study`, `scripts/run_study.py` with automatic retries),
  checkpoint store, analysis with cluster-bootstrap intervals, sign-flip permutation tests
  and Holm correction (`strdcheck table`), scrubbed environment record (`strdcheck env`).
- pytest plugin: `strd` marker, `strd_assert_lre`, `strd_data_dir` and `strd_dataset`
  fixtures; `--strd-data-dir`, `--strd-min-lre`, `--strd-fetch`, `--strd-report`; summary of
  the lowest LREs. `strdcheck.testing`: `params`, `check_univariate`, `check_linear`,
  `check_nonlinear`.
- `scripts/verify_certified.py` and `results/certified_check.csv`: every certified value
  recomputed (V1).
- Study results in `results/` and `REPORT.md`; standalone reproductions in `repros/`.
- Full-size study results (`configs/full.toml`: 200 starts per problem at three spreads, Part C
  up to n = 2e7) in `results/full/` (REPORT 4.4 and 5).
- Secondary analysis of silent nonlinear failures by residual sum of squares
  (`nonlinear_silent_split.csv`), condition numbers of the linear designs
  (`conditioning.csv`), and `configs/partb-extended.toml` (Part B over all three spreads).
  The split is ceiling-aware: a silent fit counts as at the minimum when its RSS reaches
  `min(tau, float64 RSS ceiling - 1)` digits.
- Run provenance: every open of a run directory is recorded in `attempts.jsonl` (git commit,
  dirty-tree flag, package versions); resuming with different code is logged as a warning and
  `summary.json` reports `mixed_provenance` and `error_units`. The analysis skips failed units
  instead of stopping.
- `least_squares` standard errors and the nonlinear residual SD are marked `derived`; the
  numerical rank of the Jacobian is recorded.
- `strdcheck study --max-minutes` and `scripts/run_study.py --max-minutes`: no unit starts
  after the limit (counted from the end of setup); a run stopped there exits with code 3 and
  resumes on the next call, so the full study can run in bounded chunks.
