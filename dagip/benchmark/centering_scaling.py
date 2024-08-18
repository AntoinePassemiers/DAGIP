# -*- coding: utf-8 -*-
#
#  centering_scaling.py
#
#  Copyright 2024 Antoine Passemiers <antoine.passemiers@gmail.com>
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

from typing import Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler

from dagip.benchmark.base import BaseMethod


class CenteringScaling(BaseMethod):

    def __init__(self, with_std: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.with_std: bool = bool(with_std)

    def normalize_(self, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return X

    def adapt_(self, Xs: np.ndarray, Xt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_scaler = RobustScaler()
        target_scaler.fit(Xt)
        source_scaler = RobustScaler()
        X_adapted = target_scaler.inverse_transform(source_scaler.fit_transform(Xs))
        weights_source = np.ones(len(Xs))
        weights_target = np.ones(len(Xt))
        return X_adapted, weights_source, weights_target

    def name(self) -> str:
        return 'Center-and-scale'
