# -*- coding: utf-8 -*-
#
#  multimodal.py
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

from typing import List

import torch

from dagip.retraction.base import Manifold


class MultimodalManifold(Manifold):

    def __init__(self):
        self.n_features: int = 0
        self.block_sizes: List[int] = []
        self.manifolds: List[Manifold] = []

    def add(self, manifold: Manifold, n_features: int) -> None:
        self.manifolds.append(manifold)
        self.block_sizes.append(n_features)
        self.n_features += n_features

    def get_blocks(self, X: torch.Tensor) -> List[torch.Tensor]:
        sizes = np.cumsum([0] + list(self.block_sizes))
        starts = sizes[:-1]
        ends = sizes[1:]
        blocks = []
        for start, end in zip(starts, ends):
            blocks.append(X[:, start:end])
        return blocks

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        Y = []
        for manifold, block in zip(self.manifolds, self.get_blocks(X)):
            Y.append(manifold.transform(block))
        return torch.cat(Y, dim=1)

    def _inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        Y = []
        for manifold, block in zip(self.manifolds, self.get_blocks(X)):
            Y.append(manifold.inverse_transform(block))
        return torch.cat(Y, dim=1)
