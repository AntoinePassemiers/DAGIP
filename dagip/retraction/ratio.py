# -*- coding: utf-8 -*-
#
#  ratio.py
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

import torch

from dagip.retraction.base import Manifold


class RatioManifold(Manifold):

    def __init__(self, eps: float = 1e-7):
        self.eps: float = eps

    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(X)

    def _inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.clamp(X, self.eps, 1.0 - self.eps)
        return torch.logit(X)
