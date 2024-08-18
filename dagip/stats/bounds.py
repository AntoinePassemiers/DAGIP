# -*- coding: utf-8 -*-
#
#  bounds.py
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

from typing import Dict, Optional

import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds


def compute_theoretical_bounds(
        X: np.ndarray,
        gc_content: np.ndarray,
        replicates1: Optional[np.ndarray] = None,
        replicates2: Optional[np.ndarray] = None,
        ddof: int = 0
) -> Dict[str, float]:

    n = len(X)

    if replicates1 is not None:
        replicates1 = gc_correction(replicates1, gc_content)
        replicates2 = gc_correction(replicates2, gc_content)

    if (replicates1 is not None) and (replicates2 is not None):
        uncertainty = np.sum((replicates1 - replicates2) ** 2, axis=0) / (len(replicates1) - ddof)
    else:
        X_right = X[:, 2:]
        X_left = X[:, :-2]
        X_estimate = np.copy(X)
        X_estimate[:, 1:-1] = 0.5 * (X_left + X_right)
        uncertainty = np.sum((X_estimate - X) ** 2., axis=0) / (n - ddof)

    variance = np.var(X, axis=0, ddof=ddof)

    ss_res = np.sum(2. * n * uncertainty)
    ss_tot = np.sum(n * uncertainty + n * variance)
    r2 = 1. - ss_res / ss_tot

    return {'r2': r2}
