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

The quick study (`configs/quick.toml`, 1,884 units, 0 errors, one attempt) and a secondary
Part B run over three random-start spreads are in `results/` and summarised in `REPORT.md`:

- every float64 univariate path is within 1.5 digits of its representation ceiling;
- on Filip the default singular-value cutoffs of `numpy.linalg.lstsq`, `torch.linalg.lstsq`
  and statsmodels `OLS(method="pinv")` give 0 correct digits, `Polynomial.fit` 13.8;
- from random starts about a third of scipy nonlinear fits report success with fewer than
  4 correct digits (0.31, 95% CI 0.18-0.45, for trf with an analytic Jacobian and tight
  tolerances); tight tolerances lower the trf silent rate by 0.067 (Holm p = 0.005).

No result is claimed as an engine bug; notable cells have standalone reproductions in
`repros/`.

## Limitations

One machine (Windows 11, OpenBLAS, CPU only); LAPACK-based results can differ on other
builds. The full-size configuration (`configs/full.toml`) is not part of this release's
results.

## Install

```sh
uv add "strdcheck @ git+https://github.com/h-a-forster/strdcheck@v0.1.0"
```
