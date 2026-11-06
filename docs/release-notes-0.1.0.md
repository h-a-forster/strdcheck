# strdcheck 0.1.0

First release: a library, a pytest plugin and a study of the numerical accuracy of the Python
data stack on the 58 NIST Statistical Reference Datasets (StRD).

## What is in it

- **Library.** Manifest and checksum-verified downloader for the StRD files, parser, log
  relative error (LRE), exact rational references and float64/float32 representation
  ceilings, 27 nonlinear models with analytic Jacobians, and engine adapters for numpy,
  pandas, polars, DuckDB, scipy, statsmodels, PyTorch (CPU) and `statistics`.
- **pytest plugin.** `strd` marker, `strd_dataset` and `strd_assert_lre` fixtures,
  `--strd-*` options and a CSV report; helpers in `strdcheck.testing`.
- **Study.** Resumable, checkpointed runner (`strdcheck study`, `scripts/run_study.py` with
  retries and an optional `--max-minutes` limit per call), analysis with cluster-bootstrap
  intervals and Holm-corrected permutation tests (`strdcheck table`).

## Results

The quick study (`configs/quick.toml`, 1,884 units, none failed, one attempt) and a secondary
Part B run over three random-start spreads are in `results/`, the full-size study
(`configs/full.toml`, 17,628 units, none failed) in `results/full/`, all summarised in
`REPORT.md`:

- every float64 univariate path is within 1.5 digits of its representation ceiling;
- on Filip the default rank cutoffs of `numpy.linalg.lstsq`, `torch.linalg.lstsq` and
  statsmodels `OLS(method="pinv")` give 0 correct digits, `Polynomial.fit` 13.8;
- from random starts about a third of scipy nonlinear fits report success with fewer than
  4 correct digits (0.31, 95% CI 0.18-0.45, for trf with an analytic Jacobian and tight
  tolerances); tight tolerances lower the trf silent rate by 0.067 (Holm p = 0.005);
- the full-size run (200 starts per problem, three spreads) gives 0.28 (0.16-0.41) for the
  same configuration and repeats the tolerance effect (-0.077, Holm p = 0.0005) and no other;
  at n = 1e7 and 2e7 numpy and pandas keep all 15 digits of the SD, the other engines 8.3-9.9.

No result is claimed as an engine bug; notable cells have standalone reproductions in
`repros/`.

## Failing loudly

- `strdcheck study` exits 4 when every unit ran but some failed (a rerun skips them), and 3
  when `--max-minutes` stopped it; `scripts/run_study.py` retries neither.
- `strdcheck table` refuses a run with units left and copies `study.log` next to the tables,
  so run times are committed with the results.
- `check_univariate`, `check_linear` and `check_nonlinear` raise on an unknown dataset or
  Start and record a returned name that is not a certified quantity as an error for that
  dataset, instead of ignoring it.
- The git commit in the environment record comes from a source checkout only, never from a
  repository that happens to contain the installed package.

## Limitations

One machine (Windows 11, OpenBLAS, CPU only); LAPACK-based results can differ on other
builds. The primary analysis is the quick study; the full-size run was made afterwards and is
reported as secondary.

## Install

```sh
uv add "strdcheck @ git+https://github.com/h-a-forster/strdcheck@v0.1.0"
```
