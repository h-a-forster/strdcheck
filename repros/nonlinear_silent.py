"""BoxBOD from NIST Start 1: scipy reports success at a wrong point (a silent failure).

Usage: python nonlinear_silent.py <dir with StRD .dat files>

Certified b1 = 213.80940889, b2 = 0.54723748542. From Start 1 (b1 = 1, b2 = 1) the
Levenberg-Marquardt paths stop with success flags far from the certified minimum; trf
reaches it. Convergence to a stationary point is all a local solver promises, so this is
expected behaviour that a success flag does not reveal.
"""

import numpy as np
from _strd import rows
from scipy.optimize import curve_fit, least_squares

d = np.array([[float(v) for v in r] for r in rows("BoxBOD")])
y, x = d[:, 0], d[:, 1]


def f(xx, b1, b2):
    return b1 * (1 - np.exp(-b2 * xx))


for method in ("lm", "trf"):
    r = least_squares(lambda b: f(x, *b) - y, [1.0, 1.0], method=method)
    print(method, r.success, r.status, r.x)
p, *_ = curve_fit(f, x, y, p0=[1.0, 1.0])
print("curve_fit", p)
