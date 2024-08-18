# -*- coding: utf-8 -*-
#
#  lower_bound.py
#
#  Copyright 2023 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import numpy as np
import scipy.special
import tqdm
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from dagip.core import wasserstein_distance


def wdist_(X: np.ndarray, n: int, frac: float, p: int = 1) -> float:

    n1 = max(1, int(np.round(n * frac)))
    n2 = max(1, n - n1)
    n1 = n - n2
    assert not ((n1 == 0) or (n2 == 0))

    idx = np.arange(n)
    np.random.shuffle(idx)
    X1 = X[idx[:n1], :]
    X2 = X[idx[n1:n1+n2], :]

    return float(wasserstein_distance(X1, X2, p=p))


def estimate_wasserstein_lower_bound(
        X: np.ndarray,
        m: int,
        p: int = 1,
        n_iter: int = 200
) -> float:

    n = len(X)
    assert n >= 2
    frac = n / (n + m)
    xs, ys = [], []
    for _ in tqdm.tqdm(range(n_iter), desc='Estimating lower bound'):
        size = int(np.random.randint(2, n))
        xs.append(size)
        ys.append(wdist_(X, size, frac, p=p))
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    def f(x, a, b, c, d, e, f):
        x = x - f
        return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

    popt, _ = curve_fit(f, xs, ys)

    plt.scatter(xs, ys, alpha=0.4)
    xs = np.append(np.sort(xs), len(X) + m)
    y_hat = f(xs, *popt)
    plt.plot(xs, y_hat)
    plt.colorbar()
    plt.show()

    return float(y_hat[-1])
