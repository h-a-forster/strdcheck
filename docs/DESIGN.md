# strdcheck design

Status: scaffold (2026-10-23). This document is the specification. Code follows it; when they
disagree, fix one of them in the same commit.

Contents: 1 Goal - 2 Prior art - 3 Scope - 4 Formulas - 5 Architecture and modules -
6 Study design - 7 pytest plugin - 8 Milestones - 9 Validation strategy - 10 Risks and open
questions - 11 Feasibility probe - 12 References.

## 1. Goal and headline deliverable

Measure, reproducibly and fairly, how many correct significant digits the 2026 Python data
stack delivers on the NIST Statistical Reference Datasets (StRD), on CPU.

Headline deliverable (v0.1):

1. **The LRE table**: for every StRD dataset (58 files, four suites) and every engine call
   path (about 60 configurations), the log relative error (LRE) of each certified quantity,
   shown next to the *representation ceiling* (section 4.6), so a reader can tell lost digits
   caused by the engine from digits already lost when the decimal data were stored as binary
   floats. Published as `results/lre_table.csv` and `results/lre_table.md`, summarised in
   `REPORT.md`.
2. **Nonlinear least squares outcome rates**: for scipy's `least_squares` (trf, dogbox, lm)
   and `curve_fit`, the share of fits that report success but are wrong (*silent failures*),
   on the 54 NIST problem/start pairs and on seeded random starts, with cluster-bootstrap
   intervals and Holm-adjusted paired comparisons (section 6.3).
3. **A pytest plugin** so that library authors can put StRD accuracy checks in their own test
   suites (section 7).

Secondary (should, cut first if time runs short): **error growth with n** for columnar
engines on StRD-derived tiled datasets (section 6.4), where chunked and parallel aggregation
paths are exercised. The StRD files are small (at most 18,009 rows) and mostly hit
single-chunk code paths.

Non-goals: GPU or TF32 accuracy (no GPU here; listed as follow-up F1), speed benchmarking,
claims about R, SAS or Stata, and changes to any engine.

## 2. Prior art and positioning

Checked 2026-10-23.

- **Methodology.** McCullough (1998, 1999) defined the StRD/LRE assessment of statistical
  software; McCullough and Wilson (1999) applied it to Excel; Keeling and Pavur (2007)
  compared nine packages. Altman, Gill and McDonald (2003) cover the wider numerical issues.
  NIST's StRD pages describe the datasets and how certified values were computed
  (multiple-precision arithmetic; 15 significant digits for univariate, ANOVA and linear
  regression, 11 for nonlinear regression, which can be seen in the files).
- **Variance algorithms.** Welford (1962), Chan, Golub and LeVeque (1983), Pebay (2008)
  (parallel merge formulas), Higham (2002) for summation error bounds. These explain the
  expected failure modes (one-pass formulas, large constant offsets, float32 accumulation).
- **Nonlinear test problems.** MGH09, MGH10 and MGH17 come from More, Garbow and Hillstrom
  (1981).
- **Vendor certification pages.** Stata, JMP, SAS and Analyse-it publish StRD LREs for their
  own products (self-reported).
- **Code.** lmfit (BSD-3, 1.3.4) has the NIST nonlinear models in its test suite
  (`tests/NISTModels.py`, `tests/test_NIST_Strd.py`); scipy's test suite ships a few NIST
  files (`stats/tests/data/nist_anova`, `nist_linregress`); the R package NISTnls 0.9-13
  (GPL >= 2) holds the nonlinear datasets. None of these produces a cross-engine LRE table, a
  representation ceiling, or a pytest plugin. We read none of their source; models are written
  from the NIST files.
- **Package name.** `strdcheck`, `strd`, `nistrd`, `pystrd`, `nist-strd` are not on PyPI
  (PyPI JSON API returned 404 for each, 2026-10-23).
- **Unverified lead.** The project notes mention "Hill et al. 2024" (R versus statsmodels on
  StRD). Two web searches did not locate it. M9 task: find and cite it precisely, or drop the
  reference. Until then no claim of novelty relative to it is made.

Positioning: lead with the columnar engines (polars, DuckDB, pandas) and the nonlinear
silent-failure rates, which the older literature does not cover, and with the ceiling-aware
reporting, which avoids blaming engines for input rounding.

## 3. Scope

### 3.1 In scope (v0.1)

| Suite | Datasets | Certified quantities | Engine configurations (section 5.4) |
|---|---|---|---|
| Univariate | PiDigits, Lottery, Lew, Mavro, Michelso, NumAcc1-4 | mean, sd (n-1), r(1) | numpy, pandas, polars eager and streaming, DuckDB (DOUBLE, FLOAT, DECIMAL), scipy.stats, statsmodels, torch, Python `statistics`; float32 variants |
| ANOVA | SiRstv, SmLs01-09, AtmWtAg | SS and MS between/within, F, R², residual SD | statsmodels `anova_lm`, scipy `f_oneway` (F only), group-by aggregates in pandas, polars, DuckDB |
| Linear | Norris, Pontius, NoInt1-2, Filip, Longley, Wampler1-5 | coefficients, SEs, residual SD, R² | numpy `lstsq`/`polyfit`/`Polynomial.fit`, scipy `lstsq` (gelsd, gelsy, gelss), `linregress`, statsmodels OLS (pinv, qr), torch `lstsq` (gelsy, gelsd, gelss), DuckDB `regr_*`, polars/pandas covariance (simple regression only) |
| Nonlinear | 27 problems x 2 NIST starts | parameters, SEs, RSS | scipy `least_squares` {trf, dogbox, lm} x {analytic, 2-point Jacobian} x {default, tight tolerances}, `curve_fit` {default, analytic Jacobian} |

### 3.2 Cuts and why

- **GPU, TF32, JAX, CuPy, cuDF**: no GPU on this machine. Follow-up F1: run the same table
  on a CUDA box (torch with `allow_tf32` on and off, cuDF, JAX x64 on and off).
- **Torch for nonlinear least squares**: torch has no standard NLS solver; writing one would
  measure our solver, not torch. Torch covers univariate and linear.
- **statsmodels nonlinear**: statsmodels has no general NLS estimator. Not covered.
- **lmfit, scipy.odr**: optional follow-up F2. lmfit wraps scipy's MINPACK and would mostly
  duplicate `least_squares(method="lm")`.
- **pandas `Series.autocorr`**: computes the Pearson correlation of `y[1:]` and `y[:-1]`,
  a different statistic from NIST's r(1) (section 4.2). It is not compared against the
  certified value (a classic adapter mistake). r(1) is measured for statsmodels `acf`
  (same definition) and a numpy reference formula.
- **Rolling/window statistics, MCMC datasets**: StRD has no certified values for them.
- **R, SAS, Stata comparisons**: not run; cited from the literature only.
- **Performance**: timings are recorded as metadata only, never compared.

### 3.3 Size of the study

Everything in Part A (certified datasets) is small: about 100 univariate, 50 ANOVA, 150
linear and 750 nonlinear engine calls, each repeated 3 times for a reproducibility check.
The default ("quick") study, Parts A + B-quick + C-quick, is budgeted at under 15 minutes on
3 worker processes and must stay under 45 minutes. The full-size command (Part B with 200
random starts per problem and three spreads, Part C up to n = 2 x 10^7) is documented in
section 6.7 and estimated at 60-75 minutes; it is not part of the default run.

## 4. Formulas

### 4.1 Log relative error (`strdcheck.lre`, implemented)

For computed `x` and certified `c` (McCullough 1998):

    LRE(x, c) = -log10(|x - c| / |c|)    if c != 0
    LAE(x, c) = -log10(|x - c|)          if c == 0     (log absolute error)

Conventions (ours, stated so that tables are reproducible):

- `raw` is unbounded: `inf` if `x == c`, possibly negative; `nan` if `x` is not finite.
- The reported value is `min(raw, cap)`; values below 1 are reported as 0 (not even the
  leading digit is right). `nan` is reported as a failure, never as 0.
- `cap` is the number of significant digits in NIST's printed certified value (leading zeros
  excluded, trailing zeros included because NIST prints to full certified precision), so 15
  for univariate, ANOVA and linear values and 11 for nonlinear values. Values NIST marks
  `(exact)` (NumAcc1-4) and certified zeros (Wampler1 SEs and residual SD) use cap 15.
- Certified values that are not finite (the Wampler1 F statistic, printed `Infinity`) are
  skipped and listed as such.
- A group of quantities (for example all coefficients) is summarised by its minimum LRE.

### 4.2 Univariate (NIST definitions)

    ybar = (1/n) sum y_i
    s    = sqrt( sum (y_i - ybar)^2 / (n - 1) )
    r(1) = sum_{i=2..n} (y_i - ybar)(y_{i-1} - ybar) / sum_{i=1..n} (y_i - ybar)^2

Engine conventions to respect: numpy `std` defaults to `ddof=0`; pandas and polars default
to `ddof=1`; DuckDB `stddev` is `stddev_samp`; statsmodels `DescrStatsW.std` uses `ddof=0`
unless `ddof` is given; torch `std` uses `correction=1`. statsmodels
`tsa.stattools.acf(y, nlags=1, adjusted=False, fft=False)[1]` equals r(1) above.

### 4.3 One-way ANOVA

With groups g = 1..k, sizes n_g, means m_g, sample variances s_g^2, grand mean m, N total:

    SS_between = sum_g n_g (m_g - m)^2           df_b = k - 1
    SS_within  = sum_g (n_g - 1) s_g^2           df_w = N - k
    MS = SS / df,  F = MS_between / MS_within,  R^2 = SS_b / (SS_b + SS_w),
    residual SD = sqrt(MS_within)

The group-by adapters (pandas, polars, DuckDB) take `count`, `mean` and `var` (ddof 1) per
group from the engine plus the engine's grand mean, and combine them with the formulas above
in float64 Python arithmetic, identical for all three engines. The combination step is
benign (sums of non-negative terms); the engine aggregates carry the numerical risk, which is
what is measured. statsmodels uses `ols("y ~ C(g)")` + `anova_lm(typ=1)`; scipy
`f_oneway` returns F only.

### 4.4 Linear least squares

Model `y = X b + e` with `X` built from the predictors exactly as the NIST model line says
(intercept column unless the model has none; polynomial datasets use raw powers `x^j`,
computed in the engine's dtype). With `p` columns and RSS the residual sum of squares:

    s^2 = RSS / (n - p),   SE(b_j) = s * sqrt([(X'X)^{-1}]_jj)
    R^2 = 1 - RSS / sum (y_i - ybar)^2        (model with intercept)
    R^2 = 1 - RSS / sum y_i^2                 (no-intercept: NoInt1, NoInt2; uncentred)

Engines that return only coefficients (numpy/scipy/torch `lstsq`) get SEs, residual SD and
R² from one shared float64 routine (QR of `X`, `R^{-1}` by triangular solve). These are
marked `derived = true` in the results and reported in separate columns; the headline uses
coefficient LREs. Engines that compute them natively (statsmodels, `linregress`, `polyfit`
with `cov=True`, DuckDB `regr_r2`) are measured as returned. The rank each engine reports
(numpy/scipy/torch `lstsq` return it) is recorded: a rank below `p` is truncation by the
engine's `rcond` default, a documented behaviour, and the table says so rather than calling
it a defect.

### 4.5 Nonlinear least squares

Residuals `r_i(b) = y_i - f(x_i; b)`, `RSS(b) = sum r_i^2`, Jacobian `J_ij = d f(x_i; b)/d b_j`.
NIST certifies `b*`, `RSS(b*)`, residual SD `sqrt(RSS/(n - p))` and

    SE(b_j) = s * sqrt([(J'J)^{-1}]_jj)  at b*,  s^2 = RSS / (n - p).

`curve_fit` returns `pcov` (with `absolute_sigma=False`, scaled by `s^2`); SEs are
`sqrt(diag(pcov))`. For `least_squares` the adapter computes the same formula from `res.jac`
at the returned point via the SVD of `J` (singular values below `eps * max(n, p) * s_max`
treated as zero, rank recorded). `RSS = 2 * res.cost`.

Nelson's model is `log(y) = b1 - b2 x1 exp(-b3 x2)`: the response is `log(y)` computed in
float64 by the adapter, as in NIST's certification. ENSO and Roszman1 use `pi` to full
float64 precision.

Outcome classes per fit (threshold `tau = 4` correct digits on the minimum parameter LRE;
sensitivity at `tau` = 2 and 6):

| Solver reports | min param LRE >= tau | min param LRE < tau or non-finite |
|---|---|---|
| success | `ok` | `silent` (silent failure) |
| failure (status <= 0, `RuntimeError`, max evaluations) | `flagged_ok` | `flagged` |
| exception other than a reported failure | - | `error` |

`curve_fit` raising `RuntimeError` ("optimal parameters not found") is a reported failure;
`OptimizeWarning` (covariance not estimable) is recorded as a warning, not a failure.

### 4.6 Representation ceiling

NIST certifies statistics of the *decimal* data. Engines see binary floats: `1000000000000.4`
(SmLs07-09) is stored in float64 with an absolute error up to about `6e-5`, which already
limits SS_within to about 3-4 correct digits whatever the engine does.

For every univariate, ANOVA and linear quantity we compute, with exact rational arithmetic
(`fractions.Fraction`; square roots with `decimal` at 50 digits):

    exact_dec  = statistic of the decimal data          (must reproduce NIST: section 9, V1)
    exact_f64  = statistic of the float64-rounded data  (Fraction(float(s)) is exact)
    exact_f32  = statistic of the float32-rounded data
    ceiling_dtype = LRE(exact_dtype, certified)

For polynomial models the powers are taken exactly of the rounded `x`. The report shows each
engine LRE with its ceiling and the *deficit* `ceiling - LRE` (clipped at 0). For nonlinear
problems the float64 ceiling is computed with mpmath (40 digits) by Gauss-Newton refinement
from `b*` on float64-rounded data (for Nelson, on `log(y)` as numpy computes it); a run that
does not reach a relative step of `1e-20` within 100 iterations marks the ceiling unknown.
(Gauss-Newton converges only linearly on large-residual problems: MGH09 gains about 0.2
digits per iteration, so the 20 iterations first planned were not enough. M1 measured all
27 problems converging, ENSO slowest at about 18 s for both data versions.)

### 4.7 StRD-derived tiled datasets (Part C)

From a univariate dataset `y_1..y_n0` (as float64 values) build `Y` = `k` concatenated
copies (n = k n0), optionally permuted with a seeded permutation. With `d_i = y_i - ybar`
(exact rationals of the float64 values):

    mean(Y) = ybar,   s^2(Y) = k * sum d_i^2 / (k n0 - 1)

Both are invariant under permutation. These are *derived references*, exact for the float64
data, not NIST-certified values, and are labelled as such everywhere. They are validated by
checking that `k = 1` reproduces `exact_f64` from 4.6.

## 5. Architecture and module specifications

### 5.1 Package layout

| Path | Role | Milestone |
|---|---|---|
| `src/strdcheck/lre.py` | LRE, LAE, significant digits, clipping | done |
| `src/strdcheck/manifest.py` + `data/manifest.json` | 58 entries: URL, SHA-256, size, difficulty, n | done |
| `src/strdcheck/fetch.py` | NIST-only downloader, checksum, atomic cache writes | done |
| `src/strdcheck/cli.py` | `list`, `fetch`, `verify`; later `study`, `table`, `env` | done (partial), M7 |
| `src/strdcheck/pytest_plugin.py` | marker, options, `strd_assert_lre`; later dataset fixtures, report | done (partial), M6 |
| `src/strdcheck/parse.py` | StRD file parser -> `Dataset` | M1 |
| `src/strdcheck/exact.py` | exact rational references and ceilings (4.6, 4.7) | M1 |
| `src/strdcheck/models.py` | 27 nonlinear models with analytic Jacobians | M5 |
| `src/strdcheck/results.py` | `Quantity`, `EngineResult`, `UnitRecord` schema | M2 |
| `src/strdcheck/checkpoint.py` | one-file-per-unit store, progress log, retry loop | M2 |
| `src/strdcheck/engines/` | adapters: `univariate.py`, `anova.py`, `linear.py`, `nonlinear.py`, `registry.py` | M3-M5 |
| `src/strdcheck/study/` | `config.py`, `runner.py`, `analysis.py`, `report.py`, `tiled.py` | M7-M9 |
| `src/strdcheck/testing.py` | public helpers for the pytest plugin | M6 |
| `scripts/build_manifest.py` | regenerate the manifest from NIST | done |
| `configs/` | `quick.toml`, `full.toml` | M7 |
| `results/` | committed small outputs (CSV, Markdown, JSON environment record) | M9 |

### 5.2 Parser (`parse.py`, M1)

```python
@dataclass(frozen=True)
class Certified:
    name: str  # "mean", "sd", "r1", "b1", "se_b1", "rss", "ss_between", ...
    text: str  # NIST string, verbatim
    value: float  # float(text)
    cap: int  # section 4.1
    exact: bool  # NIST "(exact)"


@dataclass(frozen=True)
class Dataset:
    info: DatasetInfo
    model: str  # model line(s) from the header
    columns: tuple[str, ...]  # e.g. ("y", "x") or ("group", "y")
    raw: dict[str, tuple[str, ...]]  # decimal strings per column, verbatim
    certified: dict[str, Certified]
    starts: tuple[tuple[float, ...], ...] = ()  # nonlinear: Start 1, Start 2
    n_params: int

    def array(self, column: str, dtype=np.float64) -> np.ndarray: ...
```

Rules: read bytes, decode ASCII, strip `\r`. Rat43's header prints 9 residual degrees of
freedom although n - p = 11 (its certified residual SD uses 11): the parser keeps the printed
value as `df["printed_residual"]` and uses n - p everywhere. Take the data line range from the header
(`Data (lines a to b)`); check the row count equals the manifest `n_obs` and every row has the
expected number of fields; raise `ParseError` otherwise. Certified blocks are parsed with one
regex set per suite. `array(dtype)` parses each string with Python `float` (correctly rounded
to float64) and then casts; for float32 this double rounding can differ from direct decimal ->
float32 rounding by one float32 ulp in rare ties; `exact_f32` uses the same path, so the
ceiling stays consistent. Offline tests use synthetic files in StRD layout under
`tests/fixtures/strd/` written by us; tests on the real files skip when the cache is empty.

### 5.3 Exact references (`exact.py`, M1)

Pure standard library (`fractions`, `decimal`) for univariate, ANOVA and linear: inputs are a
sequence of `Fraction` (from decimal strings or from floats). Linear regression solves the
normal equations exactly in rationals (exact arithmetic makes the normal equations safe; the
largest case, Filip, is 11 x 11). Nonlinear refinement uses mpmath (in the `engines` extra
and the dev group), imported lazily. Results are cached per dataset under the run directory.

### 5.4 Engine adapters (M3-M5)

Interface:

```python
class Adapter(Protocol):
    id: str  # "polars.lazy-streaming.f64"
    suite: Suite
    engine: str  # distribution name for version capture, e.g. "polars"

    def available(self) -> bool: ...  # engine importable
    def applicable(self, ds: Dataset) -> bool: ...  # e.g. linregress only for Norris
    def run(self, ds: Dataset, **kw) -> EngineResult: ...
```

`EngineResult`: `values: dict[str, float]` (keys match `Certified.name`), `derived:
set[str]`, `status: dict` (success flag, message, nfev, rank), `warnings: list[str]`,
`seconds: float`. Adapters import the engine inside `run`, never at module import.

Configuration ids (v0.1 target list; each id appears once in `registry.py`):

- Univariate (14 + 2 controls): `numpy.f64`, `numpy.f32`, `pandas.f64`, `pandas.f32`,
  `polars.eager.f64`, `polars.eager.f32`, `polars.lazy-streaming.f64`, `duckdb.double`,
  `duckdb.float`, `duckdb.decimal`, `scipy.describe.f64`, `statsmodels.f64` (mean, sd, acf),
  `torch.f64`, `torch.f32`, `stdlib.statistics` (exact-rational internally; sanity anchor);
  controls `control.onepass.f64` (textbook `sum(y^2) - n ybar^2`, expected to fail) and
  `control.exact` (exact rationals, expected to hit the cap).
- ANOVA (7 + controls): `statsmodels.anova_lm`, `scipy.f_oneway`, `pandas.groupby`,
  `polars.groupby`, `duckdb.groupby.double`, `duckdb.groupby.decimal`, `numpy.twopass`
  (reference implementation of 4.3), controls as above.
- Linear (16 + control): `numpy.lstsq`, `numpy.lstsq.f32`, `numpy.polyfit` (polynomial
  datasets), `numpy.Polynomial.fit` (polynomial; `.convert()` back to the power basis),
  `scipy.lstsq.gelsd`, `scipy.lstsq.gelsy`, `scipy.lstsq.gelss`, `scipy.linregress` (Norris),
  `statsmodels.ols.pinv`, `statsmodels.ols.qr`, `torch.lstsq.gelsy.f64`,
  `torch.lstsq.gelsd.f64`, `torch.lstsq.gelss.f64`, `torch.lstsq.gelsy.f32`,
  `duckdb.regr` (Norris), `polars.cov` and `pandas.cov` (Norris: slope = cov/var),
  control `control.normal-equations.f64`. All defaults as shipped (no `rcond` tuning); a
  `rcond=0` variant is a possible follow-up, not v0.1.
- Nonlinear (14): `scipy.least_squares.{trf,dogbox,lm}.{jac-analytic,jac-2point}.{tol-default,tol-tight}`
  (tight: `ftol = xtol = gtol = 1e-15`), `scipy.curve_fit.default`,
  `scipy.curve_fit.jac-analytic`. No bounds; default `max_nfev`.

Threading: worker processes set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `POLARS_MAX_THREADS` to 1 before importing engines, call
`torch.set_num_threads(1)` and open DuckDB connections with `threads = 1`. Part C sets
threads explicitly per configuration (1 or 3) and runs in a single worker.

### 5.5 Results and checkpointing (`results.py`, `checkpoint.py`, M2)

A *unit* is one `(part, dataset, adapter id[, start id])`. Each finished unit is written as
one JSON file (atomic temp-file + `os.replace`) under `runs/<run>/units/`, and a line is
appended to `runs/<run>/progress.log` (timestamp, unit id, status, seconds). The run
directory has `manifest.json` (config hash, package versions, environment record); a rerun
skips units whose file exists and refuses to mix a different config hash. A unit records:
computed values, certified text, LRE raw and reported, cap, ceiling, deficit, status,
warnings, three repeat hashes (bitwise reproducibility), seconds. Units that raise are
recorded with `status = "error"` and the exception text (never silently dropped).

The study command runs under a retry wrapper (`scripts/run_study.py`, up to 3 attempts with
resume) and logs to `runs/<run>/study.log`. Workers: `concurrent.futures.ProcessPoolExecutor`
with at most 3 processes (`MAX_WORKERS = 3`), `spawn` start method on all platforms.

### 5.6 Environment record

Captured once per run: Python, OS, CPU model and logical count, package versions of every
engine, numpy BLAS/LAPACK build info (`np.show_config(mode="dicts")`), `threadpoolctl`
info, torch parallel info, DuckDB and polars thread settings, strdcheck version and git
commit. Written to `results/environment.json` with local user paths scrubbed (paths replaced
by `<home>` / basename) before commit.

## 6. Study design

### 6.1 Part A: certified datasets (headline table)

All 58 datasets x all applicable adapters, NIST starts only for nonlinear. Each adapter call
is repeated 3 times in the same process; the record stores whether the 3 results are bitwise
identical. Output per (dataset, adapter): LRE per quantity, minimum over the coefficient group,
ceiling, deficit, status. Descriptive only: no hypothesis tests on Part A; numbers are
deterministic given the environment record. Rows are grouped by suite and NIST difficulty.

### 6.2 What counts as a finding in Part A

A cell is *notable* if (a) a float64 adapter's deficit exceeds 2 digits on a quantity whose
float64 ceiling is at least 6, or (b) a solver reports success with fewer than 4 correct
digits, or (c) the 3 repeats are not bitwise identical. Every notable cell goes through the
triage protocol (6.6) before REPORT.md describes it, and REPORT.md says which documented
behaviour explains it (for example `rcond` truncation in `lstsq`).

### 6.3 Part B: nonlinear outcome rates

- **NIST starts.** 27 problems x 2 starts x 14 configurations = 756 fits; reported as a full
  outcome table and counts per configuration and difficulty tier (8 lower, 11 average,
  8 higher). Descriptive.
- **Random starts.** For problem `p` with certified `b*`, start `s` draws
  `b0_j = b*_j * exp(sigma * z_j)`, `z_j ~ N(0, 1)`, from
  `numpy.random.default_rng(SeedSequence([20261023, p_index, s_index, sigma_index]))`. The
  multiplicative form keeps signs (it tests basin size, not sign recovery; stated as a
  limitation). The same starts are used by every configuration (paired design).
  Quick: `sigma = 1.0`, `M = 20` starts per problem (540 starts, 7,560 fits). Full:
  `sigma in {0.25, 1.0, 2.0}`, `M = 200` (16,200 starts, 226,800 fits).
- **Estimand.** Silent-failure rate of configuration `c`:
  `R_c = (1/27) sum_p (1/M) sum_s 1[outcome = silent]`, each problem weighted equally; also
  reported per difficulty tier and per sigma, together with the `ok`, `flagged` and `error`
  rates.
- **Intervals.** 95% percentile cluster bootstrap over problems (B = 10,000, fixed seed);
  per-problem rates with Wilson intervals in the appendix table.
- **Pre-specified comparisons** (paired by start, family-wise Holm at 0.05, written here
  before any run):
  1. `lm` vs `trf` (analytic Jacobian, default tolerances);
  2. `dogbox` vs `trf` (analytic, default);
  3. analytic vs 2-point Jacobian (`trf`, default);
  4. tight vs default tolerances (`trf`, analytic);
  5. `curve_fit.default` vs `least_squares.trf.jac-analytic.tol-tight`.
  Statistic: mean over problems of the per-problem difference in silent-failure rate. Test:
  sign-flip permutation over problems (20,000 random flips, `p = (1 + count) / (B + 1)`),
  interval by paired cluster bootstrap. Everything else is exploratory and labelled so.
- **Thresholds.** Primary `tau = 4`; the tables are repeated for `tau = 2` and `tau = 6`.

### 6.4 Part C: error growth with n (should; cut first)

Datasets NumAcc3, NumAcc4 (hard: large offset) and PiDigits (control). Sizes quick
`n in {1e4, 1e5, 1e6}`, full adds `{1e7, 2e7}` (k = round(n / n0) copies). Orders: tiled and
one seeded permutation. Adapters: numpy, pandas, polars eager, polars streaming, DuckDB with
1 and 3 threads, torch f64. Quantities: mean, sd against the derived references of 4.7. Each
cell repeated 3 times (bitwise reproducibility, which matters for multithreaded aggregation).
Peak memory stays under about 1.5 GB at n = 2e7 (float64 array plus engine copies).

### 6.5 Long-run safety

Units are checkpointed (5.5). The study command is resumable and wrapped in a retry loop.
Progress is logged per unit; `strdcheck study --status <run>` prints done/total and rate.
A stall watch (log not growing for 10 minutes) is part of the operator procedure in AGENTS.md.

### 6.6 Triage protocol for anything that looks like an engine defect

1. Reproduce with a standalone script that uses only the engine and the data (no
   strdcheck code) and commit it under `repros/`.
2. Rule out the adapter: check the engine documentation for the convention involved (ddof,
   definition, default `rcond`, tolerance semantics, dtype promotion), and check the adapter
   on a synthetic dataset with an exactly known answer.
3. Compare with the representation ceiling: digits lost to input rounding are not a defect.
4. Check the engine's issue tracker for a known report.
5. Only then describe it, with versions and platform, as "observed behaviour" in REPORT.md.
   Filing upstream is the maintainers' decision, not part of the study.

### 6.7 Commands and time budget

```sh
uv sync --group dev --extra torch
uv run strdcheck fetch
uv run python scripts/run_study.py --config configs/quick.toml --run runs/quick   # < 15 min target
uv run strdcheck table runs/quick --out results/
# Full size (documented; about 60-75 min on 3 workers):
uv run python scripts/run_study.py --config configs/full.toml --run runs/full
```

Estimates (to be replaced by measured values in REPORT.md): Part A 3-5 min (exact ceilings
for SmLs03/06/09 and the nonlinear mpmath refinements dominate), Part B quick 3-6 min,
Part C quick 2-4 min.

### 6.8 Reporting rules

REPORT.md states the environment, versions, sizes, seeds and the exact commands; every number
comes from committed files in `results/`; intervals accompany every rate; the five
pre-specified comparisons are reported with raw and Holm-adjusted p-values whatever their
outcome; platform dependence (BLAS/LAPACK build, CPU) is stated as a limitation, since LREs
of LAPACK-based paths can differ on other machines.

## 7. pytest plugin (M6)

Registered through the `pytest11` entry point. Done now: marker `strd(name)`, options
`--strd-data-dir` and `--strd-min-lre`, fixtures `strd_data_dir` and `strd_assert_lre`.

M6 adds:

- `strdcheck.testing.params(suite=None, difficulty=None)` -> list of `pytest.param(name,
  id=name, marks=pytest.mark.strd(name))`, for `@pytest.mark.parametrize`.
- Fixture `strd_dataset` (use with `indirect=True`): returns a parsed `Dataset`; skips with a
  clear reason when the file is not cached, unless `--strd-fetch` is given, which downloads
  and verifies it first.
- Every `strd_assert_lre` call is recorded; `--strd-report PATH` writes a CSV (test id,
  label, quantity, LRE, cap, threshold, passed) and the terminal summary prints the five
  lowest LREs.
- `strdcheck.testing.check_linear(fit)`, `check_univariate(stats)`, `check_nonlinear(fit)`:
  run a user callable over a suite and return a table of LREs, so a library can assert
  `min_lre` per difficulty tier in one test.
- Plugin tests use `pytester` and synthetic fixture files; no network.

## 8. Milestones and done-checks

Each milestone ends with `uv run pytest`, `uv run ruff check .` and
`uv run ruff format --check .` passing, AGENTS.md "Status" updated, and a commit.

| ID | Deliverable | Done-check |
|---|---|---|
| M0 | Scaffold, manifest, fetch, LRE, plugin stub, DESIGN | done: tests, ruff, `strdcheck --version` |
| M1 | `parse.py`, `exact.py`, synthetic fixtures, `scripts/verify_certified.py` | all 58 files parse with row counts equal to the manifest; for every univariate, ANOVA and linear certified value `LRE(exact_dec, certified) >= cap - 1`; for every nonlinear problem `LRE(RSS(b*), certified RSS) >= 9` and SEs from 4.5 at `b*` reach `>= 8` (mpmath); any miss is documented in DESIGN as a certified-value question, not patched. **Done 2026-10-24**: worst `LRE(exact_dec)` is cap - 0.65 (Filip b6); mpmath refinement from `b*` reproduces every nonlinear certified value to >= 10.3 digits. Evaluated *at* the printed `b*`, all problems pass except Lanczos1, whose certified RSS (1.4e-25) is below the resolution of 11-digit parameters (RSS at the printed `b*` is 4e-21); the refined check covers it |
| M2 | `results.py`, `checkpoint.py`, retry wrapper | kill-and-resume test: a run interrupted after k units resumes and finishes with exactly N unit files and no duplicates; config-hash mismatch refused |
| M3 | univariate and ANOVA adapters + controls | each adapter exact on a synthetic integer dataset; `control.exact` hits the cap on all 20 datasets; `control.onepass.f64` has LRE(sd) < 2 on NumAcc3/4 (expected failure, proves the pipeline detects loss) |
| M4 | linear adapters + control | synthetic exact-fit test per adapter; `control.normal-equations.f64` loses at least 5 digits on Filip or Longley versus `numpy.Polynomial.fit` / statsmodels qr; rank recorded for lstsq paths |
| M5 | `models.py`, nonlinear adapters, outcome classes | analytic Jacobians agree with mpmath derivatives at `b*` to relative 1e-8 for all 27 models; `scipy.least_squares.trf.jac-analytic.tol-tight` from Start 2 reaches min LRE >= 6 on all lower-difficulty problems |
| M6 | pytest plugin features of section 7 | pytester tests for params, skip/fetch, report CSV and summary |
| M7 | study config, runner, `scripts/run_study.py`, CLI `study`/`table`/`env` | quick study completes on this machine under 45 min (target 15) with 3 workers; rerun skips all units |
| M8 | Part C tiled study | `k = 1` reproduces `exact_f64`; quick grid completes; reproducibility recorded |
| M9 | analysis (bootstrap, permutation, Holm), `results/`, REPORT.md, README table | every REPORT number traceable to `results/`; triage done for every notable cell; Hill et al. located or dropped |
| M10 | release hygiene | CHANGELOG 0.1.0, CITATION version, scrub check for local paths, CI green on a PR |

Cut order if time runs short: M8 first, then the f32 variants and DuckDB DECIMAL configs,
then `curve_fit.jac-analytic`. M1-M7 and M9 are the minimum for the headline.

## 9. Validation strategy

- **V1 Certified-value integrity.** Exact rational recomputation from the decimal data must
  reproduce every univariate, ANOVA and linear certified value (`LRE >= cap - 1`); mpmath
  must reproduce nonlinear RSS and SEs at `b*`. This validates the parser and the formulas of
  section 4 and documents any certified value that cannot be reproduced.
- **V2 LRE unit tests.** Hand-computed cases: relative, absolute (certified zero), exact
  match, non-finite, caps from strings, `(exact)` (implemented in `tests/test_lre.py`).
- **V3 Positive and negative controls in every run.** `control.exact` must sit at the cap or
  ceiling; `control.onepass.f64` and `control.normal-equations.f64` must fail where numerical
  analysis says they fail (NumAcc3/4, SmLs07-09, Filip, Longley). If a control does not
  behave, the run is invalid until explained.
- **V4 Adapter correctness.** Every adapter is tested on a synthetic dataset with a known
  exact answer (integer data, exact-fit regressions), which catches `ddof`, definition and
  column-order mistakes before any StRD comparison. Nonlinear analytic Jacobians are checked
  against mpmath numerical derivatives.
- **V5 Ceilings.** Engine LREs are reported with float64/float32 ceilings (4.6); a deficit,
  not a raw LRE, is what can be attributed to an engine.
- **V6 Reproducibility.** Three in-process repeats per unit, a recorded environment, a config
  hash per run, and a rerun that skips completed units and reproduces the table bit for bit
  on the same machine.
- **V7 Cross-platform code.** CI (Ubuntu 3.11 and 3.13, Windows 3.13) runs the offline test
  suite. The study itself runs on one machine; LREs of BLAS/LAPACK paths are not claimed to
  hold elsewhere.

## 10. Risks and open questions

- **O1 Data licence.** NIST's open-data page says SRD products may carry copyright under the
  Standard Reference Data Act, while catalog.data.gov lists SRD 140 under the NIST open
  licence. Until this is settled the raw files are not committed or bundled: they are fetched
  from NIST and verified by checksum. Bundling (about 2.1 MB) is a later decision.
- **R1 Platform dependence.** LAPACK-based results (lstsq drivers, statsmodels pinv/qr,
  torch) depend on the BLAS/LAPACK build (OpenBLAS vs MKL vs Accelerate) and CPU. Mitigation:
  environment record; claims scoped to the recorded machine.
- **R2 Moving targets.** polars 2.0, DuckDB 2.0 and pandas 3.1 are recent major versions
  (DuckDB 2.0 already changed its relation API during the probe). Versions are pinned in
  `uv.lock`; adapters target the locked versions and lower bounds are tested in CI on 3.11.
- **R3 Adapter mistakes posing as engine bugs.** Mitigated by V4 and the triage protocol
  6.6; the probe already shows the `rcond` trap (section 11).
- **R4 Ceiling computation cost.** Exact rationals on SmLs03/06/09 (18,009 rows) and Filip
  powers are fine (seconds), but the nonlinear mpmath refinement could be slow on Gauss1-3
  and Hahn1. Mitigation: cache per dataset; if a refinement exceeds 60 s the ceiling is
  marked unknown.
- **R5 Random-start design.** Multiplicative perturbations around `b*` do not explore sign
  changes and depend on parameter scaling; rates are conditional on this design and are not a
  universal "failure probability".
- **R6 Torch wheels.** CPU wheels come from `download.pytorch.org/whl/cpu` (configured in
  `pyproject.toml`); if that index is unavailable, torch adapters are skipped and the table
  says so.
- **R7 Statistical overreach.** Part A is descriptive; only Part B's five pre-specified
  comparisons get p-values (Holm). Notable cells are described with their documented cause.

## 11. Feasibility probe (2026-10-23)

A throwaway script (not committed) ran a few engine paths on this machine (Windows 11, CPU,
locked versions: numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, polars 2.0.0, DuckDB 2.0.0,
statsmodels 0.15.0, torch 2.14.1+cpu). Indicative only; these are not results and they will
be redone by the study with the full protocol.

- NumAcc3/NumAcc4 sd: every float64 engine reached the float64 ceiling (about 9.5 and 8.3
  digits), confirming that the ceiling, not the engine, limits these cells; float32 paths
  had 0-1 digits; the one-pass control produced `nan` on NumAcc4.
- NumAcc4 tiled to n = 1.0e6 (sd against the exact float64 reference): numpy and pandas 15
  digits, DuckDB 12.3-12.8, torch f64 12.6, polars 11.3-11.8. Differences of up to 3-4 digits
  at n = 1e6 support Part C.
- Filip coefficients: `numpy.linalg.lstsq` (default `rcond`), statsmodels OLS (pinv) and
  torch `lstsq` (gelsd, gelsy) 0 digits; scipy `lstsq` 5.7-7.9; `np.polyfit` 7.5;
  statsmodels OLS qr 8.1; `Polynomial.fit(...).convert()` 13.8. The zeros are consistent with
  singular-value truncation by default `rcond`, a documented behaviour: the adapters must
  record the reported rank (4.4) before anything is said about them.
- BoxBOD from Start 1: `least_squares(method="lm")` and `curve_fit` reported success with
  0 correct digits (a silent failure in the sense of 4.5); trf and dogbox reached 5+ digits.
  MGH09 from Start 1: dogbox reported failure; trf and lm reported success with 3.8-4.0
  digits.

## 12. References

- Altman, M., Gill, J., McDonald, M. P. (2003). *Numerical Issues in Statistical Computing
  for the Social Scientist*. Wiley. doi:10.1002/0471475769.
- Chan, T. F., Golub, G. H., LeVeque, R. J. (1983). Algorithms for computing the sample
  variance: analysis and recommendations. *The American Statistician* 37(3), 242-247.
  doi:10.1080/00031305.1983.10483115.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.
  doi:10.1137/1.9780898718027.
- Keeling, K. B., Pavur, R. J. (2007). A comparative study of the reliability of nine
  statistical software packages. *Computational Statistics & Data Analysis* 51(8),
  3811-3831. doi:10.1016/j.csda.2006.02.013.
- McCullough, B. D. (1998). Assessing the reliability of statistical software: Part I.
  *The American Statistician* 52(4), 358-366. doi:10.1080/00031305.1998.10480597.
- McCullough, B. D. (1999). Assessing the reliability of statistical software: Part II.
  *The American Statistician* 53(2), 149-159. doi:10.1080/00031305.1999.10474450.
- McCullough, B. D., Wilson, B. (1999). On the accuracy of statistical procedures in
  Microsoft Excel 97. *Computational Statistics & Data Analysis* 31(1), 27-37.
  doi:10.1016/S0167-9473(99)00004-3.
- More, J. J., Garbow, B. S., Hillstrom, K. E. (1981). Testing unconstrained optimization
  software. *ACM Transactions on Mathematical Software* 7(1), 17-41. doi:10.1145/355934.355936.
- NIST. Statistical Reference Datasets, SRD 140. doi:10.18434/T43G6C;
  https://www.itl.nist.gov/div898/strd/.
- Pebay, P. (2008). Formulas for robust, one-pass parallel computation of covariances and
  arbitrary-order statistical moments. Sandia report SAND2008-6212. doi:10.2172/1028931.
- Welford, B. P. (1962). Note on a method for calculating corrected sums of squares and
  products. *Technometrics* 4(3), 419-420. doi:10.1080/00401706.1962.10490022.
