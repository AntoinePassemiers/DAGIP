# -*- coding: utf-8 -*-
#
#  core.py
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
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from dagip.da.deprecated.dataset import TrainDataset
from dagip.da.deprecated.opt import WassersteinDistance


class Model(torch.nn.Module):

    def __init__(self, n_samples: int, n_hidden: int = 400, n_layers: int = 10, residual: bool = True):
        torch.nn.Module.__init__(self)
        self.n_samples: int = int(n_samples)
        self.residual: bool = bool(residual)
        self.n_hidden: int = int(n_hidden)
        self.n_layers: int = int(n_layers)

        ns = [3] + [self.n_hidden] * self.n_layers + [n_samples]

        self.layers = torch.nn.ModuleList()

        for i in range(len(ns) - 1):

            layer = torch.nn.Sequential()

            n_in, n_out = ns[i], ns[i + 1]
            is_last_layer = (i + 1 == len(ns) - 1)

            # Linear layer
            linear = torch.nn.Linear(n_in, n_out, bias=True)
            if not is_last_layer:
                gain = torch.nn.init.calculate_gain('tanh')
            else:
                gain = torch.nn.init.calculate_gain('linear')
            torch.nn.init.xavier_uniform_(linear.weight, gain=gain)
            linear.bias.data.fill_(np.random.normal(0, 0.005))
            layer.add_module(f'linear-{i + 1}', linear)

            # Layer norm
            #if not is_last_layer:
            #    layer.add_module(f'layer-norm-{i + 1}', torch.nn.LayerNorm(n_out))

            # Non-linear activation
            if not is_last_layer:
                layer.add_module(f'activation-{i + 1}', torch.nn.Tanh())

            # Add layer
            self.layers.append(layer)

    def preprocess(self, X):
        read_counts = X[:, :self.n_samples]
        extra = X[:, self.n_samples:]
        return torch.cat((
            extra,
            torch.mean(read_counts, dim=1).unsqueeze(1),
            torch.std(read_counts, dim=1).unsqueeze(1),
        ), dim=1)

    def forward(self, X):
        in_ = X[:, :self.n_samples]
        #X = self.preprocess(X)
        X = X[:, self.n_samples:]
        for layer in self.layers:
            out = torch.nan_to_num(layer(X), nan=0)
            if self.residual:
                if out.size() == X.size():
                    X = X + out
                else:
                    X = out
            else:
                X = out

        X = 0.02 * X
        # X = X - torch.mean(X, dim=0).unsqueeze(0)  # TODO: implement running mean
        X = torch.clamp(in_ + X, in_ * 0.6, in_ * 1.4)

        return X

    @staticmethod
    def pairwise_rbf_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(X, Y)
        gamma = 500. / X.size()[1]
        return torch.exp(-gamma * distances ** 2)

    def adapt(
            self,
            X1: np.ndarray,
            X2: np.ndarray,
            t: np.ndarray,
            side_info: np.ndarray,
            max_n_iter: int = 10000,
            batch_size: int = 4096,
            reg_rate: float = 0,  # 2
            lr=2*1e-4
    ) -> np.ndarray:
        n_samples, n_features = X1.shape

        # Compute the weights of the regularisation function
        t = np.squeeze(LabelEncoder().fit_transform(t))
        assert t.shape == (n_samples,)
        weights = t[:, np.newaxis] == t[np.newaxis, :]
        sum_ = np.sum(weights)
        if sum_ > 0:
            weights = weights / sum_
        weights = torch.FloatTensor(weights)

        self.train()
        wasserstein = WassersteinDistance()

        training_data = TrainDataset(X1, X2, side_info)
        dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
        iterator = iter(dataloader)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0)

        losses = []
        for iteration in tqdm.tqdm(range(max_n_iter), desc='Domain adaptation'):
            optimizer.zero_grad()

            try:
                x1, x2, si = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                x1, x2, si = next(iterator)
            x1_prime = self.forward(torch.cat((x1, si), dim=1))

            loss = wasserstein.forward(x1_prime.t(), x2.t())
            print(loss)
            if reg_rate > 0:
                k_target = Model.pairwise_rbf_kernel(x1.t(), x1.t())
                k1 = Model.pairwise_rbf_kernel(x1_prime.t(), x1_prime.t())

                reg = torch.sum(weights * (k1 - k_target) ** 2)

                loss = loss + reg_rate * reg
                print(f'Loss at iteration {iteration + 1}: {loss.item()} (reg={reg.item()})')

            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        self.eval()

        return self.forward(torch.cat((
            torch.FloatTensor(X1.T),
            torch.FloatTensor(side_info.T)
        ), dim=1)).cpu().data.numpy().T
