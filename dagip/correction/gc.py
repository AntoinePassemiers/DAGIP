# -*- coding: utf-8 -*-
#
#  gc.py
#
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
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

import tqdm

#import loess.loess_1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np


def read_counts_correction_(x, gc_content, mappability):
    valid = (x > 0)
    range_ = np.quantile(x[valid], [0, 0.99])
    domain = np.quantile(gc_content[valid], [0.001, 0.999])

    ideal = np.ones(len(valid), dtype=bool)
    ideal[~valid] = False
    ideal[mappability < 0.9] = False
    ideal[np.logical_or(x <= range_[0], x > range_[1])] = False
    ideal[np.logical_or(gc_content < domain[0], gc_content > domain[1])] = False

    # GC correction
    binned = np.linspace(0, 1, 1001)
    import loess.loess_1d
    _, y_out, _ = loess.loess_1d.loess_1d(gc_content[ideal], x[ideal], xnew=binned[ideal], frac=0.03, degree=2)
    assert not np.any(np.isnan(y_out))
    _, final, _ = loess.loess_1d.loess_1d(binned, y_out, xnew=gc_content[ideal], frac=0.3, degree=2)
    assert not np.any(np.isnan(final))
    assert not np.any(final == 0)
    x[ideal] = x[ideal] / final

    # Mappability correction
    #range_ = np.quantile(x[valid], [0, 0.99])
    #ideal = np.logical_and(x > range_[0], x <= range_[1])
    #x = x / lowess(mappability[ideal], x[ideal], xvals=mappability, frac=2./3.)

    return x


def read_counts_correction(x, gc_content, mappability, centromeric):
    x = np.copy(x)
    x[~centromeric] = read_counts_correction_(x[~centromeric], gc_content[~centromeric], mappability[~centromeric])
    return x


def lowess_correction(X: np.ndarray, exog: np.ndarray, desc: str, frac: float) -> np.ndarray:
    exog = np.round(exog * 1000).astype(int) // 10
    X_adapted = np.copy(X)
    for i in tqdm.tqdm(range(len(X)), desc=desc):

        y_pred = lowess(X[i, :], exog, frac=frac, return_sorted=False)
        #_, y_pred, _ = loess.loess_1d(exos, X[i, :], xnew=None, degree=2, frac=0.3,
        #                    npoints=None, rotate=False, sigy=None)

        mask = (y_pred > 0)
        X_adapted[i, mask] = X[i, mask] / y_pred[mask]

        mask = (exog == 0)
        X_adapted[i, mask] = X[i, mask]

    return X_adapted


def mappability_correction(X: np.ndarray, mappability: np.ndarray) -> np.ndarray:
    return lowess_correction(X, mappability, 'Mappability correction', 2. / 3.)


def gc_correction(X: np.ndarray, gc: np.ndarray, frac: float = 0.3) -> np.ndarray:
    return lowess_correction(X, gc, 'GC correction', frac)
