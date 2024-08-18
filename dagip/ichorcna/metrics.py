# -*- coding: utf-8 -*-
#
#  metrics.py
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

import functools
from typing import Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from dagip.segmentation import SovRefine


def evaluation_metric(func: Callable) -> Callable:
    @functools.wraps(func)
    def new_func(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        results = []
        for i in range(len(y)):
            if (y[i] is None) or (y_hat[i] is None):
                continue
            if np.any(np.isnan(y[i])) or np.any(np.isnan(y_hat[i])):
                continue
            res = func(y[i], y_hat[i])
            if not np.isnan(res):
                results.append(res)
        return np.asarray(results)
    return new_func


def check_ploidy(y: np.ndarray, y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert y.dtype == int
    assert y_hat.dtype == int
    assert np.max(y) <= 5
    assert np.max(y_hat) <= 5

    y_hat = np.copy(y_hat)
    last = y_hat[np.where(y_hat >= 0)[0][0]]
    for i in range(len(y_hat)):
        if y_hat[i] < 0:
            y_hat[i] = last
        else:
            last = y_hat[i]

    mask = (y >= 0)

    return y[mask], y_hat[mask]


@evaluation_metric
def ploidy_accuracy(y, y_hat) -> float:
    y, y_hat = check_ploidy(y, y_hat)
    return float(np.mean(np.equal(y, y_hat)))


@evaluation_metric
def sign_accuracy(y, y_hat) -> float:
    y, y_hat = check_ploidy(y, y_hat)
    return float(np.mean(np.equal(np.sign(y - 2), np.sign(y_hat - 2))))


@evaluation_metric
def cna_accuracy(y, y_hat) -> float:
    y, y_hat = check_ploidy(y, y_hat)
    return float(np.mean(np.equal(y == 2, y_hat == 2)))


@evaluation_metric
def sov_refine(y, y_hat) -> float:
    y, y_hat = check_ploidy(y, y_hat)
    return SovRefine(y_hat, y).sov_refine()


@evaluation_metric
def absolute_error(y, y_hat) -> float:
    assert not np.any(np.isnan(y))
    assert not np.any(np.isnan(y_hat))
    return float(np.mean(np.abs(y - y_hat)))
