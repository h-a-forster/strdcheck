"""Filip: least squares paths with a default singular-value cutoff return rank 10 of 11.

Usage: python filip_rcond.py <dir with StRD .dat files>

Expected (documented behaviour, not a defect): numpy.linalg.lstsq and torch.linalg.lstsq
truncate singular values below rcond * s_max (default rcond = eps * max(M, N)); statsmodels
OLS(method="pinv") uses a pinv cutoff. The raw-power Filip design has a 2-norm condition
number near 1e15, so one singular value falls below the cutoff and the solution is the
minimum-norm solution of a rank-10 problem, far from the certified coefficients.
"""

import numpy as np
from _strd import rows

data = np.array([[float(v) for v in r] for r in rows("Filip")])
y, x = data[:, 0], data[:, 1]
X = np.vander(x, 11, increasing=True)
s = np.linalg.svd(X, compute_uv=False)
print(f"cond(X) = {s[0] / s[-1]:.3e}; s_min/s_max = {s[-1] / s[0]:.3e}")
print(f"default numpy cutoff eps*max(M,N) = {np.finfo(float).eps * max(X.shape):.3e}")
b, _, rank, _ = np.linalg.lstsq(X, y)
print("numpy.linalg.lstsq rank:", rank, " b10:", b[10], " certified b10: -4.02962525080404e-05")
b, _, rank, _ = np.linalg.lstsq(X, y, rcond=0.0)
print("numpy.linalg.lstsq(rcond=0) rank:", rank, " b10:", b[10])
