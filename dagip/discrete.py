# -*- coding: utf-8 -*-
#
#  discrete.py
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

import random
import time
from typing import Tuple, Union, Callable

import numpy as np
import ot
import scipy.stats
import torch
import tqdm
from scipy.spatial.distance import cdist

from dagip.retraction import Identity
from dagip.retraction.base import Manifold


def pairwise_distances(
        X1: np.ndarray,
        X2: np.ndarray,
) -> np.ndarray:
    distances = cdist(X1, X2, metric='cityblock')
    return distances


def transport_plan(
        X1: np.ndarray,
        X2: np.ndarray,
        p: int = 2
) -> np.ndarray:
    n = len(X1)
    m = len(X2)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    assert not np.any(np.isnan(X1))
    assert not np.any(np.isnan(X2))
    distances = pairwise_distances(X1, X2)
    assert not np.any(np.isnan(distances ** p))
    assert not np.any(np.isinf(distances ** p))

    distances += 1e-7
    np.fill_diagonal(distances, 0)

    # gamma = ot.emd(a, b, distances ** p)
    gamma = ot.sinkhorn(a, b, distances ** p, 1e-7)
    return gamma


def default_u_test(x1: np.ndarray, x2: np.ndarray) -> float:
    return float(scipy.stats.ks_2samp(x1, x2).pvalue)


def compute_p_values(X1: np.ndarray, X2: np.ndarray, u_test: Callable) -> np.ndarray:
    p_values = []
    for k in range(X1.shape[1]):
        p_values.append(u_test(X1[:, k], X2[:, k]))
    return np.asarray(p_values)


@torch.no_grad()
def ot_da_discrete(
        X1: np.ndarray,
        X2_prime: np.ndarray,
        ret: Manifold = Identity(),
        u_test: Callable = default_u_test,
        min_p_value: float = 0.1,
        max_n_iter: int = 1000,  # 1000,
        threshold: float = 0.5,
        timeout: float = np.inf
) -> np.ndarray:

    t0 = time.time()

    X1_adapted = np.copy(X1)

    # Probability, for each variable, to keep the original values
    p = np.full(X1.shape[1], 1.0, dtype=float)

    pbar = tqdm.tqdm(range(max_n_iter), desc='OT-DA')
    for iteration in pbar:

        # Retraction mapping of source domain
        X1_prime = ret(X1_adapted).cpu().data.numpy()

        # Check statistical dissimilarities between the 2 domains using univariate tests
        p_values = compute_p_values(X1_prime, X2_prime, u_test=u_test)

        # Compute target p-values (according to a uniform distribution)
        expected_p_values = np.linspace(min_p_value, 1, len(p_values))
        idx = np.argsort(np.argsort(p_values))
        expected_p_values = expected_p_values[idx]

        # Stopping criterion
        crit = np.mean(np.abs(p_values - expected_p_values))
        # _, uniformity_p_value = scipy.stats.kstest(p_values, scipy.stats.uniform(loc=0, scale=1).cdf)
        print(f'{crit:.7f} - {np.median(p_values):.3f}')
        pbar.set_description(f'{crit:.7f} - {np.median(p_values):.3f}')
        if (crit < 0.01) and np.all(p_values >= min_p_value):
            break

        # Decide which variables to correct more
        p[p_values < expected_p_values] -= 0.1
        p[p_values > expected_p_values] += 0.1
        p = np.clip(p, 0.0, 1.0)

        # Solve OT problem to determine a direction for correction
        gamma = transport_plan(X1, X2_prime)
        gamma /= np.sum(gamma, axis=1)[:, np.newaxis]

        # Bias correction
        # TODO: correlation between loci
        idx2 = np.arange(X2_prime.shape[1])
        for i in range(X1_adapted.shape[0]):
            idx = np.random.choice(np.arange(len(gamma[i, :])), size=X1_adapted.shape[1], p=gamma[i, :])
            mask = (np.random.rand(len(p)) < p)
            X1_adapted[i, mask] = X1[i, mask]
            X1_adapted[i, ~mask] = X2_prime[idx[~mask], idx2[~mask]]

        # Timeout
        if time.time() - t0 >= timeout:
            break

    # Retraction mapping of source domain
    X1_prime = ret(X1_adapted).cpu().data.numpy()

    return X1_prime
