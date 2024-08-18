# -*- coding: utf-8 -*-
#
#  scaler.py
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

import numpy as np
import torch


class Scaler(torch.nn.Module):

    def __init__(self, n_features, bias=True):
        torch.nn.Module.__init__(self)
        self.n_features = n_features
        size = (1, self.n_features)
        self.weight = torch.nn.Parameter(torch.ones(size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(size))
        else:
            self.bias = None

    def forward(self, X):
        X = self.weight * X
        if self.bias is not None:
            X = X + self.bias
        return X
