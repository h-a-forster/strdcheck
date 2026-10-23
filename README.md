# strdcheck

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

How many digits does your data stack get right? strdcheck measures the numerical accuracy of
the Python data stack on the NIST Statistical Reference Datasets (StRD), using the log
relative error (LRE), the number of correct significant digits.

> **Status: pre-alpha (scaffold).** The dataset manifest, the checksum-verified downloader,
> the LRE functions and a first pytest fixture work. The parsers, engine adapters, study and
> LRE table are planned; see [docs/DESIGN.md](docs/DESIGN.md) for the specification and
> milestones. No results are published yet.

## What it will cover

| StRD suite | Datasets | Certified quantities | Engines (CPU) |
|---|---|---|---|
| Univariate summary statistics | 9 | mean, standard deviation, lag-1 autocorrelation | numpy, pandas, polars, DuckDB, scipy, statsmodels, PyTorch (float32/float64), Python `statistics` |
| One-way ANOVA | 11 | between/within sums of squares, mean squares, F, R², residual SD | statsmodels, scipy, group-by aggregates in pandas, polars, DuckDB |
| Linear least squares | 11 | coefficients, standard errors, residual SD, R² | numpy, scipy (`lstsq` drivers, `linregress`), statsmodels, PyTorch, DuckDB/polars/pandas (simple regression) |
| Nonlinear least squares | 27 x 2 starts | parameters, standard errors, residual sum of squares | scipy `least_squares` (trf, dogbox, lm) and `curve_fit` |

Every LRE is reported next to its *representation ceiling*: the best LRE any engine can reach
once the decimal data have been rounded to binary float64 (or float32). This keeps engines from
being blamed for digits the input format has already lost.

GPU execution (CUDA, TF32 matrix multiplies) is out of scope for v0.1 and listed as a
follow-up.

## Install (development)

```sh
uv sync --group dev                 # core, engines and test tools
uv sync --group dev --extra torch   # add CPU-only PyTorch
uv run strdcheck fetch              # download and verify the 58 StRD files from NIST
uv run strdcheck list
```

The data files are not bundled. `strdcheck fetch` downloads them from
`https://www.itl.nist.gov/div898/strd/` and checks each one against the SHA-256 recorded in
[`src/strdcheck/data/manifest.json`](src/strdcheck/data/manifest.json).

## pytest plugin

Installing strdcheck registers a pytest plugin. Today it provides an LRE assertion:

```python
def test_my_mean(strd_assert_lre):
    strd_assert_lre([my_mean(y)], [4.53480000000000], min_lre=12, cap=15)
```

Options: `--strd-data-dir` (dataset cache), `--strd-min-lre` (override every threshold).
Dataset fixtures and an LRE summary report are planned (DESIGN section 7).

## Data source and attribution

The Statistical Reference Datasets are published by the National Institute of Standards and
Technology (NIST), Statistical Engineering Division: *NIST Statistical Reference Datasets*,
SRD 140, doi:[10.18434/T43G6C](https://doi.org/10.18434/T43G6C). strdcheck is not affiliated
with or endorsed by NIST.

## License

MIT. See [LICENSE](LICENSE).
