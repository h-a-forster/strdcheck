# strdcheck

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

How many digits does your data stack get right? strdcheck measures the numerical accuracy of
numpy, pandas, polars, DuckDB, scipy, statsmodels and PyTorch on the NIST Statistical
Reference Datasets (StRD), using the log relative error (LRE): the number of correct
significant digits against values NIST certifies to 11-15 digits.

It ships three things:

- a **library** that parses all 58 StRD files, computes exact rational references and the
  *representation ceiling* of every certified value (the best LRE any engine can reach once
  the decimal data are rounded to float64 or float32), and scores engine results;
- a **pytest plugin** so a numerical library can assert, in its own test suite, that it keeps
  N digits on the StRD problems;
- a **reproducible study** of the Python data stack, with a published LRE table, silent-failure
  rates of scipy's nonlinear least squares solvers from random starts, and a large-n
  experiment, in [REPORT.md](REPORT.md).

## Why

The StRD were built to catch statistical software that returns confident wrong answers, and
the classic accuracy studies (McCullough 1998, 1999) covered the commercial packages of their
time. The Python stack that most analyses now run on has not had the same treatment across
engines. Two details make the comparison fair: every LRE is read against the representation
ceiling, so an engine is not blamed for digits the input format already lost; and for
nonlinear least squares the study counts *silent* failures, fits that report success at the
wrong answer, which a success flag cannot reveal.

## Key results (v0.1.0)

From [REPORT.md](REPORT.md) (one machine, CPU only; see its limitations):

- Univariate statistics: every float64 path is within 1.4 digits of the ceiling; float32
  paths keep 0-7.6 digits.
- Linear least squares follows conditioning. On Filip (cond(X) about 1.8e15) the default
  singular-value cutoffs of `numpy.linalg.lstsq`, `torch.linalg.lstsq` and statsmodels
  `OLS(method="pinv")` give 0 correct digits; `numpy.polynomial.Polynomial.fit` gets 13.8.
- Nonlinear least squares from random starts: about a third of fits report success with fewer
  than 4 correct digits (for example 0.31, 95% CI 0.18-0.45, for `least_squares(method="trf")`
  with an analytic Jacobian and tight tolerances). Tight tolerances lower the silent rate of
  trf by 0.067 (Holm p = 0.005).
- At n = 1e6, numpy and pandas keep all 15 digits of the SD of tiled NumAcc data; DuckDB,
  polars and torch keep 8.9-11.2 in their worst cell, and multi-threaded DuckDB and polars
  streaming are not bitwise repeatable.

## Install

strdcheck is installed from GitHub (it is not on PyPI):

```sh
# library and pytest plugin (needs numpy only)
uv add "strdcheck @ git+https://github.com/h-a-forster/strdcheck@v0.1.0"
# with the engines the study measures (scipy, pandas, polars, DuckDB, statsmodels, mpmath)
uv add "strdcheck[engines] @ git+https://github.com/h-a-forster/strdcheck@v0.1.0"
```

For development or to reproduce the study:

```sh
git clone https://github.com/h-a-forster/strdcheck
cd strdcheck
uv sync --group dev --extra torch      # engines, CPU-only PyTorch and test tools
```

## Quick start

The data files are not bundled. Download and verify them (SHA-256 against
[`src/strdcheck/data/manifest.json`](src/strdcheck/data/manifest.json)) into a user cache:

```sh
uv run strdcheck fetch        # 58 files from https://www.itl.nist.gov/div898/strd/
uv run strdcheck list         # suite, name, difficulty, n, cache state
```

Score a solver on the linear suite:

```python
import numpy as np
from strdcheck.testing import check_linear

table = check_linear(
    lambda X, y: np.linalg.lstsq(X, y, rcond=None)[0],
    datasets=["Norris", "Pontius", "Filip", "Longley", "Wampler5"],
)
print(table.by_dataset())
```

```text
{'Norris': 11.88, 'Pontius': 6.24, 'Filip': 0.0, 'Longley': 10.99, 'Wampler5': 5.67}
```

(Values rounded here; numpy 2.5.4 with OpenBLAS 0.3.34.) `check_univariate` and
`check_nonlinear` work the same way; the nonlinear version passes the model, its analytic
Jacobian, the data and the NIST start to your fitting function.

## pytest plugin

Installing strdcheck registers a pytest plugin (entry point `pytest11`):

```python
import pytest
from strdcheck.exact import ceilings
from strdcheck.testing import params


@pytest.mark.parametrize("strd_dataset", params(suite="univariate"), indirect=True)
def test_my_std(strd_dataset, strd_assert_lre):
    c = strd_dataset.certified["sd"]
    ceiling = ceilings(strd_dataset)["sd"].f64  # best LRE possible from float64 data
    sd = my_std(strd_dataset.array("y"))
    strd_assert_lre([sd], [c.value], min_lre=ceiling - 1, cap=c.cap, label="sd")
```

Thresholds relative to the ceiling matter: on NumAcc4 no float64 computation can get more than
8.3 of the 15 certified digits of the SD.

| Option | Effect |
|---|---|
| `--strd-data-dir PATH` | dataset cache (default: user cache directory) |
| `--strd-fetch` | download missing files instead of skipping |
| `--strd-min-lre X` | override every threshold |
| `--strd-report PATH` | write every recorded LRE to a CSV |

The terminal summary lists the five lowest LREs of the session.

## Reproducing the study

```sh
uv run python scripts/verify_certified.py                                         # about 40 s
uv run python scripts/run_study.py --config configs/quick.toml --run runs/quick   # about 7 min
uv run strdcheck table runs/quick --out results
```

Runs are checkpointed per unit (one JSON file each), resume after an interruption, retry
automatically, use at most 3 worker processes and log to `runs/<name>/study.log`. The
full-size configuration (`configs/full.toml`, about 2-2.5 hours) and a secondary run over all
three random-start spreads (`configs/partb-extended.toml`) are described in
[docs/DESIGN.md](docs/DESIGN.md) section 6.7.

## Validation

- Exact rational arithmetic reproduces all 236 certified values of the univariate, ANOVA and
  linear suites, and 40-digit mpmath Gauss-Newton refinement reproduces the 294 nonlinear
  ones ([`results/certified_check.csv`](results/certified_check.csv)).
- Positive and negative controls (exact arithmetic; one-pass variance and normal equations)
  behave as expected in every run.
- The test suite checks reference values exactly, every model Jacobian against mpmath, the
  adapters on synthetic data with known answers, the checkpoint store under interruption and
  the statistics against exact values: `uv run pytest` (add `--runslow` for the end-to-end
  study test).
- Notable results were triaged against documentation and come with standalone reproductions
  in [`repros/`](repros/). None is claimed as a bug.

## Project layout

| Path | Contents |
|---|---|
| `src/strdcheck/` | manifest, fetch, parser, exact references, models, LRE, engine adapters, study runner and analysis, pytest plugin, CLI |
| `configs/` | study configurations (quick, full, partb-extended) |
| `results/` | committed tables from the published run |
| `repros/` | standalone reproductions of notable cells |
| `docs/DESIGN.md` | specification, formulas and study design |

## Citation

If you use strdcheck, please cite it ([CITATION.cff](CITATION.cff)) and the NIST StRD:

```bibtex
@software{strdcheck,
  title   = {strdcheck: numerical accuracy of the Python data stack on the NIST StRD},
  author  = {{strdcheck contributors}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/h-a-forster/strdcheck}
}
```

## Data source and attribution

The Statistical Reference Datasets are published by the National Institute of Standards and
Technology (NIST), Statistical Engineering Division: *NIST Statistical Reference Datasets*,
SRD 140, doi:[10.18434/T43G6C](https://doi.org/10.18434/T43G6C). strdcheck is not affiliated
with or endorsed by NIST.

## License

MIT. See [LICENSE](LICENSE).
