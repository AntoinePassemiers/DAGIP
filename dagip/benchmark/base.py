# -*- coding: utf-8 -*-
#
#  base.py
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

from abc import abstractmethod, ABCMeta
from typing import Tuple, List

import numpy as np

from dagip.retraction.base import Manifold


class BaseMethod(metaclass=ABCMeta):

    def __init__(self, per_label: bool = True):
        self.per_label: bool = per_label

    def normalize(self, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
        X_normalized = self.normalize_(X, reference)
        assert X_normalized.shape == X.shape
        return X_normalized

    def adapt(
            self,
            X: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            target_domain: int = 0,
            subsample_target: float = 1
    ) -> Tuple[np.ndarray, np.ndarray]:

        X_adapted = np.copy(X)
        weights = np.ones(len(X))

        for domain in np.unique(d):
            if domain != target_domain:

                if self.per_label:

                    # Define groups
                    X1s, X2s, masks_source, idxs_target = [], [], [], []
                    for label in np.unique(y):
                        mask_source = np.logical_and(d == domain, y == label)
                        mask_target = np.logical_and(d == target_domain, y == label)
                        idx_target = np.where(np.logical_and(d == target_domain, y == label))[0]

                        # Subsample target samples to avoid encouraging overfitting during validation
                        if subsample_target < 1:
                            n_target = max(10, int(subsample_target * np.sum(mask_source)))
                            idx_target = idx_target[:n_target]
                        
                        if np.sum(mask_target) == 0:
                            continue
                        if np.sum(mask_source) == 0:
                            continue
                        masks_source.append(mask_source)
                        idxs_target.append(idx_target)
                        X1s.append(X_adapted[mask_source])
                        X2s.append(X_adapted[idx_target])

                    # Correction
                    res = self.adapt_per_label_(X1s, X2s)

                    # Replace samples
                    for mask_source, idx_target, (X1_adapted, weights_source, weights_target) in zip(masks_source, idxs_target, res):
                        X_adapted[mask_source, :] = X1_adapted
                        weights[mask_source] = weights_source
                        weights[idx_target] = weights_target

                else:
                    mask_source = (d == domain)
                    mask_target = (d == target_domain)
                    idx_target = np.where(d == target_domain)[0]
                    if len(idx_target) == 0:
                        continue
                    if np.sum(mask_source) == 0:
                        continue

                    # Subsample target samples to avoid encouraging overfitting during validation
                    if subsample_target < 1:
                        n_target = max(10, int(subsample_target * np.sum(mask_source)))
                        idx_target = idx_target[:n_target]

                    # Correction
                    print(len(X_adapted[mask_source]), len(X_adapted[idx_target]))
                    x_, weights_source, weights_target = self.adapt_(X_adapted[mask_source], X_adapted[idx_target])
                    X_adapted[mask_source, :] = x_
                    weights[mask_source] = weights_source
                    weights[idx_target] = weights_target

        return X_adapted, weights

    @abstractmethod
    def normalize_(self, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
        pass

    def adapt_per_label_(self, Xs: List[np.ndarray], Xt: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [self.adapt_(Xs[i], Xt[i]) for i in range(len(Xs))]

    @abstractmethod
    def adapt_(self, Xs: np.ndarray, Xt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
