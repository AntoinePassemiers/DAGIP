# -*- coding: utf-8 -*-
#
#  r2.py
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

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


def smooth(X: np.ndarray) -> np.ndarray:
    X = np.copy(X)
    for i in range(len(X)):
        X[i, :] = savgol_filter(X[i, :], 5, 2)
    return X


def r2_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    #X = smooth(X)
    #Y = smooth(Y)

    ss_res = np.sum((X - Y) ** 2.)
    ss_tot = np.sum((Y - np.mean(Y, axis=0)[np.newaxis, :]) ** 2.)
    r2 = 1. - ss_res / ss_tot
    return float(r2)
