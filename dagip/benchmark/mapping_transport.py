# -*- coding: utf-8 -*-
#
#  mapping_transport.py
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

from typing import Tuple, List

import numpy as np
import ot.da

from dagip.benchmark.base import BaseMethod


class MappingTransport(BaseMethod):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def normalize_(self, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return X

    def adapt_per_label_(self, Xs: List[np.ndarray], Xt: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        sizes, ys = [], []
        for i, X1 in enumerate(Xs):
            sizes.append(len(X1))
            ys.append(np.full(len(X1), i, dtype=int))

        model = ot.da.MappingTransport()

        print(np.concatenate(ys, axis=0).dtype, set(np.concatenate(ys, axis=0)))

        model.fit(
            Xs=np.concatenate(Xs, axis=0),
            ys=np.concatenate(ys, axis=0),
            Xt=np.concatenate(Xt, axis=0)
        )
        X_adapted = model.transform(Xs)

        sizes = np.cumsum([0] + sizes)
        starts = sizes[:-1]
        ends = sizes[1:]

        output = []
        for start, end, y in zip(sizes[:-1], sizes[1:], Xt):
            weights_source = np.ones(end - start)
            weights_target = np.ones(len(y))
            output.append((X_adapted[start:end, :], weights_source, weights_target))

        return output

    def adapt_(self, Xs: np.ndarray, Xt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ys = np.zeros(len(Xs), dtype=int)
        model = ot.da.MappingTransport()
        model.fit(Xs=Xs, ys=ys, Xt=Xt)
        X_adapted = model.transform(Xs)
        weights_source = np.ones(len(Xs))
        weights_target = np.ones(len(Xt))
        return X_adapted, weights_source, weights_target

    def name(self) -> str:
        return 'MappingTransport'
