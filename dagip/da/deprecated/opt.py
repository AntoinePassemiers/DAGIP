# -*- coding: utf-8 -*-
#
#  opt.py
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

import numpy as np
import ot
import torch


class WassersteinDistance(torch.nn.Module):

    def __init__(self, p: float = 1.):
        super().__init__()
        self.p: float = float(p)

    def forward(self, X, Y):
        n = len(X)
        m = len(Y)
        k = X.size()[1]
        a = np.full(n, 1. / n)
        b = np.full(m, 1. / m)
        scaling = 30 / k
        distances = torch.cdist(X.contiguous(), Y.contiguous(), p=2)
        distances = scaling * (distances ** self.p)
        gamma = ot.emd(a, b, distances.cpu().data.numpy())
        gamma = torch.FloatTensor(gamma)
        return torch.sum(gamma * distances) ** (1. / self.p)

    @staticmethod
    def compute_scale(X):
        std = torch.std(X, dim=0)
        std = torch.clamp(std, 0.0000001)
        scale = (1. / std).unsqueeze(0)
        return scale
