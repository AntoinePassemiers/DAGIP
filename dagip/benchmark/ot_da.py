# -*- coding: utf-8 -*-
#
#  ot_da.py
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

import os
import uuid
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from dagip.benchmark.base import BaseMethod
from dagip.core import ot_da, train_adapter, DomainAdapter
from dagip.correction.gc import gc_correction
from dagip.plot import scatter_plot
from dagip.retraction import GIPManifold
from dagip.retraction.base import Manifold
from dagip.spatial.base import BaseDistance
from dagip.utils import log_


class OTDomainAdaptation(BaseMethod):

    def __init__(self, ret: Manifold, distance: BaseDistance, folder: str, **kwargs):
        super().__init__(**kwargs)
        self.ret: Manifold = ret
        self.distance: BaseDistance = distance
        self.folder: str = folder

    def normalize_(self, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return X

    def adapt_per_label_(self, Xs: List[np.ndarray], Xt: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        folder = os.path.join(self.folder, str(uuid.uuid4()))
        adapter = DomainAdapter(folder=folder, manifold=self.ret, distance=self.distance)
        res = adapter.fit_transform(Xs, Xt)
        return [(x, np.ones(len(x)), np.ones(len(y))) for x, y in zip(res, Xt)]

    def adapt_(self, Xs: np.ndarray, Xt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        folder = os.path.join(self.folder, str(uuid.uuid4()))
        X_adapted = ot_da(Xs, Xt, folder=folder, manifold=self.ret, distance=self.distance)
        weights_source = np.ones(len(Xs))
        weights_target = np.ones(len(Xt))
        return X_adapted, weights_source, weights_target

    def name(self) -> str:
        return 'DA'
