"""NumAcc4 tiled to n = 1e6: SD from numpy, polars, DuckDB and torch against the exact value.

Usage: python tiled_numacc4.py <dir with StRD .dat files>

numpy reduces with pairwise summation; the other engines use other accumulation orders
(streaming/parallel updates), which lose digits on data with a large offset (1e7) and small
spread (0.1) as n grows.
"""

import math
from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import torch
from _strd import rows

y0 = np.array([float(r[0]) for r in rows("NumAcc4")])
k = 1000
y = np.tile(y0, k)
q = [Fraction(v) for v in y0]
m = sum(q) / len(q)
ss = sum((v - m) ** 2 for v in q)
exact = math.sqrt(k * ss / (k * len(q) - 1))
con = duckdb.connect()
con.execute("SET threads = 1")
con.register("t", pd.DataFrame({"y": y}))
out = {
    "numpy": y.std(ddof=1),
    "polars": pl.Series(y).std(),
    "duckdb": con.execute("SELECT stddev_samp(y) FROM t").fetchone()[0],
    "torch": torch.from_numpy(y).std().item(),
}
for k_, v in out.items():
    err = abs(v - exact) / exact
    print(f"{k_:7} {v!r:24} LRE {(-math.log10(err) if err else 15):.2f}")
