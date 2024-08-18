# -*- coding: utf-8 -*-
#
#  adapter.py
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

from typing import Tuple, Optional, List

import numpy as np
import torch

from dagip.retraction.base import Manifold
from dagip.nn.scaler import Scaler


class BiasEstimator(torch.nn.Module):

    def __init__(self, n_features: int, n_degrees: int):
        super().__init__()
        self.m: int = n_features
        self.p: int = n_degrees
        self.c: torch.nn.Parameter = torch.nn.Parameter(torch.randn(self.m))
        self.ln = torch.nn.LayerNorm(self.p)

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        # AX = B
        # A: (m, p)
        # X: (p, n)
        # B: (m, n)
        # n = n_samples
        # m = n_features
        # p = n_degrees
        c = self.c / torch.std(self.c)
        A = c.unsqueeze(1) ** torch.arange(1, self.p + 1).unsqueeze(0)
        A = self.ln(A)
        A = torch.cat((A, torch.ones(self.m, 1)), dim=1)
        X = torch.linalg.lstsq(A.t() @ A, A.t() @ B.t()).solution
        #X = torch.linalg.pinv(A.t() @ A, hermitian=True) @ A.t() @ B.t()

        return -torch.mm(A, X).t()


class MLPAdapter(torch.nn.Module):

    def __init__(
            self,
            n: int,
            m: int,
            latent_shape: Tuple[int, ...],
            manifold: Manifold,
            eta: float = 0.01,
            polyfit_coef: float = 1.0
    ):
        super().__init__()
        self.eta: float = eta
        self.polyfit_coef: float = polyfit_coef
        self.n: int = n
        self.m: int = m
        self.shape = tuple([self.m] + list(latent_shape) + [self.m])
        self.manifold: Manifold = manifold
        self.is_constant: torch.BoolTensor = torch.BoolTensor(np.zeros(self.m, dtype=bool))

        layers = [torch.nn.LayerNorm(self.shape[0])]
        for k in range(len(self.shape) - 1):
            n_in = self.shape[k]
            n_out = self.shape[k + 1]
            if k != len(self.shape) - 2:
                layers.append(torch.nn.Linear(n_in, n_out))
                layers.append(torch.nn.LayerNorm(n_out))
                layers.append(torch.nn.PReLU(n_out))
            else:
                layers.append(torch.nn.Linear(n_in, n_out, bias=False))
        self.mlp = torch.nn.Sequential(*layers)
        self.mlp.apply(MLPAdapter.init_weights)

        self.ln1 = torch.nn.LayerNorm(self.shape[0])
        self.bias_estimator = BiasEstimator(self.shape[0], 9)

        self.b = torch.nn.Parameter(torch.zeros((1, self.shape[-1])))

    def l2_reg_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.mlp.parameters()) + list(self.ln1.parameters())

    def estimate_bias(self, X: torch.Tensor) -> torch.Tensor:
        b1 = self.mlp(X)
        out = self.eta * b1 + self.b
        if self.polyfit_coef > 0:
            b2 = self.ln1(self.bias_estimator(X))
            out = out + self.polyfit_coef * self.eta * b2
        return out

    def forward(self, X_original: torch.Tensor) -> torch.Tensor:
        X = self.manifold.inverse_transform(X_original)
        return self.manifold.transform(X + self.estimate_bias(X_original))

    @torch.no_grad()
    def adapt(self, X: np.ndarray) -> np.ndarray:
        X = torch.FloatTensor(X)
        X = self.forward(X)
        return X.cpu().data.numpy()

    def set_constant_mask(self, is_constant: torch.Tensor) -> None:
        self.is_constant = torch.BoolTensor(is_constant)

    @torch.no_grad()
    def init_bias(self, X_original: torch.Tensor, y_median: torch.Tensor) -> None:
        X = self.manifold.inverse_transform(X_original)
        bias = self.estimate_bias(X_original)
        bias = torch.squeeze(self.manifold.inverse_transform(y_median.unsqueeze(0)) - torch.quantile(X + bias, 0.5, dim=0).unsqueeze(0))
        self.b.data = bias.unsqueeze(0)

    @staticmethod
    def init_weights(m: torch.nn.Module) -> None:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.001)

    def save(self, filepath: str) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'constant_mask': self.is_constant
        }, filepath)

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.is_constant = checkpoint['constant_mask']
