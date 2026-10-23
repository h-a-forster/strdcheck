"""Filip: statsmodels OLS(method="qr") standard errors.

Usage: python statsmodels_qr_filip.py <dir with StRD .dat files>

With method="qr" the coefficients are good to about 8 digits, but bse comes from
normalized_cov_params computed as inv(R'R)-like products on a design with condition number
near 1e15; some diagonal entries come out negative and bse is nan (with a RuntimeWarning).
"""

import warnings

import numpy as np
import statsmodels.api as sm
from _strd import rows

d = np.array([[float(v) for v in r] for r in rows("Filip")])
y, x = d[:, 0], d[:, 1]
X = np.vander(x, 11, increasing=True)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fit = sm.OLS(y, X).fit(method="qr")
print("params[10]", fit.params[10], "(certified -4.02962525080404e-05)")
print("diag(normalized_cov_params) < 0:", int(np.sum(np.diag(fit.normalized_cov_params) < 0)))
print("bse:", fit.bse)
print([str(x.message) for x in w])
