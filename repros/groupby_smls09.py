"""SmLs09: between-group sum of squares from group means near 1e12.

Usage: python groupby_smls09.py <dir with StRD .dat files>

Values are 1000000000000.x; float64 spacing there is about 1.2e-4, so the float64 ceiling for
SS_between is about 3.9 digits. Compare group means from numpy (pairwise summation) with
DuckDB avg (one thread) and the exact means of the float64 values.
"""

from fractions import Fraction

import duckdb
import numpy as np
import pandas as pd
from _strd import rows

r = rows("SmLs09")
g = np.array([int(a) for a, _ in r])
y = np.array([float(b) for _, b in r])
con = duckdb.connect()
con.execute("SET threads = 1")
con.register("t", pd.DataFrame({"g": g, "y": y}))
dd = dict(con.execute("SELECT g, avg(y) FROM t GROUP BY g").fetchall())
for lv in range(1, 10):
    sel = y[g == lv]
    ex = sum(Fraction(v) for v in sel) / sel.size
    print(
        lv,
        f"numpy err {float(Fraction(sel.mean()) - ex):+.3e}",
        f"duckdb err {float(Fraction(dd[lv]) - ex):+.3e}",
    )
