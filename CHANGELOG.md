# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: uv/hatchling package, CI (pull requests and manual dispatch), design
  document with formulas, validation strategy, study design and milestones.
- `strdcheck.manifest`: the 58 StRD files (univariate, ANOVA, linear and nonlinear
  regression) with source URLs, SHA-256 checksums, sizes, difficulty and number of
  observations.
- `strdcheck.fetch`: downloader restricted to the NIST StRD host, with timeouts, retries,
  checksum verification and atomic writes into a user cache.
- `strdcheck.lre`: log relative error, log absolute error for certified zeros, significant
  digit counting of NIST certified strings, clipping to `[0, cap]`.
- pytest plugin: `strd` marker, `strd_assert_lre` and `strd_data_dir` fixtures,
  `--strd-data-dir` and `--strd-min-lre` options.
- CLI: `strdcheck list | fetch | verify`.
