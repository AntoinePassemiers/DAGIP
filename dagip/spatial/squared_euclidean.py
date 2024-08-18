# -*- coding: utf-8 -*-
#
#  squared_euclidean.py
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

import torch

from dagip.spatial.base import BaseDistance


class SquaredEuclideanDistance(BaseDistance):

    def distances_(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.square(X - Y), dim=1)

    def pairwise_distances_(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return torch.square(torch.cdist(X, Y, p=2))

    def barycentric_mapping_(self, gamma: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        gamma = gamma / torch.sum(gamma, dim=1).unsqueeze(1)
        return torch.mm(gamma, Y)
