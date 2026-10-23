"""DuckDB: stddev_samp over an exact DECIMAL column returns DOUBLE.

Usage: python duckdb_decimal_stddev.py <dir with StRD .dat files>

Loading NumAcc4 as DECIMAL(18, 1) keeps the data exact, yet the SD matches the float64
representation ceiling (about 8.3 digits) instead of the decimal one (15): the variance is
computed in DOUBLE. Check the result type and compare with exact arithmetic.
"""

from fractions import Fraction

import duckdb
from _strd import rows

vals = [r[0] for r in rows("NumAcc4")]
con = duckdb.connect()
con.execute("CREATE TABLE t (y DECIMAL(18, 1))")
con.executemany("INSERT INTO t VALUES (CAST(? AS DECIMAL(18, 1)))", [(v,) for v in vals])
print(con.execute("SELECT typeof(stddev_samp(y)), typeof(avg(y)) FROM t").fetchone())
sd = con.execute("SELECT stddev_samp(y) FROM t").fetchone()[0]
q = [Fraction(v) for v in vals]
m = sum(q) / len(q)
exact = float((sum((v - m) ** 2 for v in q) / (len(q) - 1)) ** 0.5)
print(f"duckdb {sd!r}  exact {exact!r}  certified 0.1")
