# -*- coding: utf-8 -*-
#
#  binning.py
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

from typing import List

import numpy as np


class ChromosomeBounds:

    @staticmethod
    def get_10kb():
        return np.asarray([
            0,        # chr1 (24896)
            24896,    # chr2 (24220)
            49116,    # chr3 (19830)
            68946,    # chr4 (19022)
            87968,    # chr5 (18154)
            106122,   # chr6 (17081)
            123203,   # chr7 (15935)
            139138,   # chr8 (14514)
            153652,   # chr9 (13840)
            167492,   # chr10 (13380)
            180872,   # chr11 (13509)
            194381,   # chr12 (13328)
            207709,   # chr13 (11437)
            219146,   # chr14 (10705)
            229851,   # chr15 (10200)
            240052,   # chr16 (9034)
            249085,   # chr17 (8326)
            257411,   # chr18 (8038)
            265449,   # chr19 (5862)
            271311,   # chr20 (6445)
            277756,   # chr21 (4671)
            282427,   # chr22 (5082)
            287509])  # end

    @staticmethod
    def get_bounds_(step: int) -> np.ndarray:
        bounds = ChromosomeBounds.get_10kb()
        offsets = (bounds[1:] - bounds[:-1]) // int(step)
        new_bounds = np.empty(len(bounds), dtype=int)
        new_bounds[0] = 0
        new_bounds[1:] = np.cumsum(offsets)
        return new_bounds

    @staticmethod
    def get_50kb() -> np.ndarray:
        return ChromosomeBounds.get_bounds_(5)

    @staticmethod
    def get_100kb() -> np.ndarray:
        return ChromosomeBounds.get_bounds_(10)

    @staticmethod
    def get_200kb() -> np.ndarray:
        return ChromosomeBounds.get_bounds_(20)

    @staticmethod
    def get_500kb() -> np.ndarray:
        return ChromosomeBounds.get_bounds_(50)

    @staticmethod
    def get_1mb() -> np.ndarray:
        return ChromosomeBounds.get_bounds_(100)

    @staticmethod
    def get_bounds(n: int) -> np.ndarray:
        if n == 287509:
            return ChromosomeBounds.get_10kb()
        elif n == 57501:
            return ChromosomeBounds.get_50kb()
        elif n == 2866:
            return ChromosomeBounds.get_1mb()
        else:
            raise ValueError(f'Unknown genome size "{n}"')

    @staticmethod
    def bin_from_10kb_to_50kb(X, **kwargs):
        new_bounds = ChromosomeBounds.get_50kb()
        old_bounds = ChromosomeBounds.get_10kb()
        # assert X.shape[-1] == ChromosomeBounds.get_10kb()[-1] + 1
        return ChromosomeBounds.rebin(X, old_bounds, new_bounds, 5, **kwargs)

    @staticmethod
    def bin_from_10kb_to_100kb(X, **kwargs):
        new_bounds = ChromosomeBounds.get_100kb()
        old_bounds = ChromosomeBounds.get_10kb()
        # assert X.shape[-1] == ChromosomeBounds.get_10kb()[-1] + 1
        return ChromosomeBounds.rebin(X, old_bounds, new_bounds, 10, **kwargs)

    @staticmethod
    def bin_from_10kb_to_200kb(X, **kwargs):
        new_bounds = ChromosomeBounds.get_200kb()
        old_bounds = ChromosomeBounds.get_10kb()
        return ChromosomeBounds.rebin(X, old_bounds, new_bounds, 20, **kwargs)

    @staticmethod
    def bin_from_10kb_to_500kb(X, **kwargs):
        new_bounds = ChromosomeBounds.get_500kb()
        old_bounds = ChromosomeBounds.get_10kb()
        return ChromosomeBounds.rebin(X, old_bounds, new_bounds, 50, **kwargs)

    @staticmethod
    def bin_from_10kb_to_1mb(X, **kwargs):
        new_bounds = ChromosomeBounds.get_1mb()
        old_bounds = ChromosomeBounds.get_10kb()
        return ChromosomeBounds.rebin(X, old_bounds, new_bounds, 100, **kwargs)

    @staticmethod
    def bin_from_10kb_to_k_mb(X, k: int, **kwargs):
        step = int(k // 10)
        new_bounds = ChromosomeBounds.get_bounds_(step)
        old_bounds = ChromosomeBounds.get_10kb()
        # assert X.shape[-1] == ChromosomeBounds.get_10kb()[-1] + 1
        return ChromosomeBounds.rebin(X, old_bounds, new_bounds, step, **kwargs)

    @staticmethod
    def bin_from_50kb_to_1mb(X, **kwargs):
        new_bounds = ChromosomeBounds.get_1mb()
        old_bounds = ChromosomeBounds.get_50kb()
        return ChromosomeBounds.rebin(X, old_bounds, new_bounds, 20, **kwargs)

    @staticmethod
    def rebin(X, old_bounds, new_bounds, ratio, reduction='mean'):
        #print(X.shape, new_bounds.shape)
        Y = []
        for i in range(len(old_bounds) - 1):
            start, end = old_bounds[i], old_bounds[i + 1]
            offset = ((end - start) // ratio) * ratio
            assert new_bounds[i + 1] == new_bounds[i] + (offset // ratio)
            shape = list(X.shape)[:-1]
            X_reshaped = X[..., start:start+offset].reshape(*shape, offset // ratio, ratio)
            if reduction == 'sum':
                X_binned = X_reshaped.sum(axis=-1)
            elif reduction == 'mean':
                X_binned = X_reshaped.mean(axis=-1)
            else:
                raise NotImplementedError(f'Unknown reduction type "{reduction}"')
            Y.append(X_binned)
        Y = np.concatenate(Y, axis=-1)
        return Y

    @staticmethod
    def separate_chromosomes_10kb(X: np.ndarray) -> List[np.ndarray]:
        return ChromosomeBounds.separate_chromosomes_(X, ChromosomeBounds.get_10kb())

    @staticmethod
    def separate_chromosomes_50kb(X: np.ndarray) -> List[np.ndarray]:
        return ChromosomeBounds.separate_chromosomes_(X, ChromosomeBounds.get_50kb())

    @staticmethod
    def separate_chromosomes_1mb(X: np.ndarray) -> List[np.ndarray]:
        return ChromosomeBounds.separate_chromosomes_(X, ChromosomeBounds.get_1mb())

    @staticmethod
    def separate_chromosomes_(X: np.ndarray, bounds: np.ndarray) -> List[np.ndarray]:
        # assert X.shape[-1] == bounds[-1]
        Y = []
        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            Y.append(X[..., start:end])
        return Y
