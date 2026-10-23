# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-24

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
- Secondary analysis of silent nonlinear failures by residual sum of squares
  (`nonlinear_silent_split.csv`), condition numbers of the linear designs
  (`conditioning.csv`), and `configs/partb-extended.toml` (Part B over all three spreads).
