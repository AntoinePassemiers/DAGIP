# -*- coding: utf-8 -*-
#
#  pca.py
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

from typing import Optional, Self

import numpy as np
import torch
from sklearn.decomposition import PCA


class DifferentiablePCA(torch.nn.Module):

    def __init__(self, n_pcs: int):
        torch.nn.Module.__init__(self)
        self.n_pcs: int = n_pcs
        self.V: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.transform(X)

    def fit(self, X: torch.Tensor) -> Self:
        self.n_pcs = min(self.n_pcs, X.size(0), X.size(1))
        pca = PCA(n_components=self.n_pcs)
        pca.fit(X.cpu().data.numpy())
        self.V = torch.as_tensor(pca.components_.T).to(X.dtype)
        self.b = torch.as_tensor(-pca.mean_[np.newaxis, :] @ pca.components_.T).to(X.dtype)
        return X

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.V.dtype)
        return torch.mm(X, self.V) + self.b

    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        self.fit(X)
        return self.transform(X)
