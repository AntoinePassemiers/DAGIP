# -*- coding: utf-8 -*-
#
#  core.py
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

import os
import os.path
import math
from typing import Optional, Tuple, Union, Callable, List, Dict

import numpy as np
import ot
import scipy.stats
import scipy.linalg
import torch
import tqdm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

from dagip.nn.adapter import MLPAdapter
from dagip.nn.pca import DifferentiablePCA
from dagip.retraction import Identity
from dagip.retraction.base import Manifold
from dagip.optimize.loss_scaling import LossScaling
from dagip.spatial.base import BaseDistance
from dagip.spatial.squared_euclidean import SquaredEuclideanDistance
from dagip.utils import log_


def compute_weights(X: np.ndarray) -> np.ndarray:
    return np.full(len(X), 1. / len(X))


def ot_emd_with_nuclear_norm_reg(C: np.ndarray, X: np.ndarray, Y: np.ndarray, gamma0: np.ndarray, reg_rate: float = 1e-4) -> np.ndarray:
    n = len(X)
    m = len(Y)
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    def f(G: np.ndarray) -> float:
        M = X - np.dot(G / a[:, np.newaxis], Y)
        s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
        return np.sum(s[1:]) / np.sum(s)

    def df(G: np.ndarray) -> np.ndarray:
        M = X - np.dot(G / a[:, np.newaxis], Y)
        U, s, Vh = np.linalg.svd(M, full_matrices=False, compute_uv=True)
        U = np.real(U)
        V = np.real(Vh.T)

        grad = (np.dot(U[:, 1:], V[:, 1:].T) * np.sum(s) - np.sum(s[1:]) * np.dot(U, V.T)) / np.square(np.sum(s))

        grad = np.dot(grad / a[:, np.newaxis], -Y.T)
        return grad
    return ot.optim.cg(a, b, C, reg_rate, f, df, G0=gamma0, log=False)


def compute_transport_plan(distances: np.ndarray, X: np.ndarray, Y: np.ndarray, gamma0: Optional[np.ndarray]) -> np.ndarray:

    # Solve the classical unregularized OT problem
    n = len(distances)
    m = distances.shape[1]
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)
    assert not np.any(np.isnan(distances))
    assert not np.any(np.isinf(distances))
    if gamma0 is None:
        gamma0 = ot.emd(a, b, distances)

    """
    # Solve the regularized OT problem
    gamma = ot_emd_with_nuclear_norm_reg(distances, X, Y, gamma0, reg_rate=1e-4)
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(gamma0)
    plt.subplot(1, 2, 2)
    plt.imshow(gamma)
    plt.show()
    import sys; sys.exit(0)
    """
    gamma = gamma0

    return gamma


def default_u_test(x1: np.ndarray, x2: np.ndarray) -> float:
    return float(scipy.stats.ks_2samp(x1, x2).pvalue)


def compute_p_values(X1: np.ndarray, X2: np.ndarray, u_test: Callable) -> np.ndarray:
    p_values = []
    for k in range(X1.shape[1]):

        # Check if feature is constant
        if np.all(X1[:, k] == X1[0, k]) and np.all(X2[:, k] == X1[0, k]):
            continue

        p_values.append(u_test(X1[:, k], X2[:, k]))
    return np.asarray(p_values)


def compute_all_p_values(X1: torch.Tensor, X2: torch.Tensor, u_test: Callable) -> torch.Tensor:
    X1 = X1.cpu().data.numpy()
    X2 = X2.cpu().data.numpy()
    p_values = []
    for k in range(X1.shape[1]):
        if np.all(X1[:, k] == X1[0, k]) and np.all(X2[:, k] == X1[0, k]):
            p_values.append(0.5)
        else:
            p_values.append(u_test(X1[:, k], X2[:, k]))
    return torch.FloatTensor(np.asarray(p_values))


def interquartile_ranges(X: torch.Tensor) -> torch.Tensor:
    return torch.quantile(X, 0.75, dim=0) - torch.quantile(X, 0.25, dim=0)


def compute_s_values(X: torch.Tensor) -> torch.Tensor:

    # Median
    mu = torch.quantile(X, 0.5, dim=0).unsqueeze(0)

    # Inter-quartile range
    sigma = interquartile_ranges(X).unsqueeze(0)
    mask = (sigma > 0)
    sigma = mask * sigma + (~mask) * torch.mean(sigma[mask])

    # Standardization
    scale = 1. / sigma
    return scale * (X - mu)


def compute_stats(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # Median
    mu = torch.quantile(X, 0.5, dim=0).unsqueeze(0)

    # Inter-quartile range
    sigma = (torch.quantile(X, 0.75, dim=0) - torch.quantile(X, 0.25, dim=0)).unsqueeze(0)
    mask = (sigma > 0)
    sigma = mask * sigma + (~mask) * torch.mean(sigma[mask])

    # Standardization
    scale = 1. / sigma

    return mu, scale


def compute_convergence_threshold(X1: np.ndarray, X2: np.ndarray, u_test: Callable = default_u_test) -> float:
    scaler = RobustScaler()
    scaler.fit(X2)
    X1_cas = scaler.inverse_transform(RobustScaler().fit_transform(X1))
    p_values = compute_p_values(X1_cas, X2, u_test)
    return float(np.median(p_values))


def center_and_scale(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    target_scaler = RobustScaler()
    target_scaler.fit(X2)
    source_scaler = RobustScaler()
    X_adapted = source_scaler.fit_transform(X1)
    return target_scaler.inverse_transform(X_adapted)


def univariate_wasserstein_distances(X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    m1 = torch.quantile(X1, 0.5, dim=0)
    m2 = torch.quantile(X2, 0.5, dim=0)
    c1 = torch.square(interquartile_ranges(X1))
    c2 = torch.square(interquartile_ranges(X2))
    return torch.square(m1 - m2) + c1 + c2 - 2 * torch.sqrt(c1 * c2)


class BlockTransportPlan:

    def __init__(self, distance: BaseDistance, block_sizes_1: List[int], block_sizes_2: List[int]):
        self.distance: BaseDistance = distance
        self.block_sizes_1: List[int] = block_sizes_1
        self.block_sizes_2: List[int] = block_sizes_2
        self.n_blocks: int = len(self.block_sizes_1)
        self.n: int = int(sum(self.block_sizes_1))
        self.m: int = int(sum(self.block_sizes_2))
        self.gammas: List[Optional[torch.Tensor]] = [None for _ in range(self.n_blocks)]
        self.X2_barycentric: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, X1: torch.Tensor, X2: torch.Tensor) -> None:
        X2_barycentric = []
        offset1, offset2 = 0, 0
        for k, (size1, size2) in enumerate(zip(self.block_sizes_1, self.block_sizes_2)):
            distances = self.distance.cdist(X1[offset1:offset1+size1], X2[offset2:offset2+size2])
            gamma = torch.FloatTensor(compute_transport_plan(
                distances.cpu().data.numpy(),
                X1[offset1:offset1+size1].cpu().data.numpy(),
                X2[offset2:offset2+size2].cpu().data.numpy(),
                gamma0=(self.gammas[k].cpu().data.numpy() if (self.gammas[k] is not None) else None)
            ))
            self.gammas[k] = gamma
            X2_barycentric.append(self.distance.barycentric_mapping(gamma, X2[offset2:offset2+size2]))
        X2_barycentric = torch.cat(X2_barycentric, dim=0)

        mu = torch.quantile(X2_barycentric, 0.5, dim=0).unsqueeze(0)
        X2_barycentric = X2_barycentric - mu
        variance_ratio = torch.sqrt(torch.mean(np.square(interquartile_ranges(X2))) / torch.mean(np.square(interquartile_ranges(X2_barycentric))))
        X2_barycentric = X2_barycentric * variance_ratio
        X2_barycentric = X2_barycentric + mu
        #X2_barycentric = torch.FloatTensor(center_and_scale(X2_barycentric.cpu().data.numpy(), X2.cpu().data.numpy()))

        self.X2_barycentric = X2_barycentric

    def __len__(self) -> int:
        if self.X2_barycentric is None:
            return 0
        else:
            return len(self.X2_barycentric)

    def __getitem__(self, idx: np.ndarray) -> torch.Tensor:
        assert self.X2_barycentric is not None
        return self.X2_barycentric[idx, :].detach()


def _ot_da(
        X1: Union[np.ndarray, List[np.ndarray]],
        X2: Union[np.ndarray, List[np.ndarray]],
        folder: Optional[str] = None,
        manifold: Manifold = Identity(),
        distance: BaseDistance = SquaredEuclideanDistance(),
        u_test: Callable = default_u_test,
        reg_rate: float = 0.002,
        u_loss_weight: float = 0.001,
        max_n_iter: int = 500,
        convergence_threshold: Union[float, str] = 'auto',
        nn_n_hidden: Union[int, str] = 'auto',
        nn_n_layers: int = 2,
        lr: float = 0.005,
        l2_reg: float = 2e-07,
        batch_size: int = 8,
        verbose: bool = True
) -> Tuple[np.ndarray, MLPAdapter]:

    if folder is not None:
        os.makedirs(folder, exist_ok=True)

    X1s = [X1] if (not isinstance(X1, list)) else X1
    X2s = [X2] if (not isinstance(X2, list)) else X2
    assert len(X1s) == len(X2s)

    block_sizes_1 = [len(X1) for X1 in X1s]
    block_sizes_2 = [len(X2) for X2 in X1s]

    X1 = np.concatenate(X1s, axis=0)
    X2 = np.concatenate(X2s, axis=0)

    # Determine number of hidden neurons
    n_samples = X1.shape[0]
    n_features = X1.shape[1]
    nn_n_hidden = max(8, int(min(0.002 * n_features, 20000 * n_samples / n_features)))

    # Determine convergence threshold
    proposed_threshold = compute_convergence_threshold(X1, X2, u_test=u_test)
    print(f'Automatic convergence threshold selection: {proposed_threshold}')
    if convergence_threshold == 'auto':
        convergence_threshold = proposed_threshold
    else:
        convergence_threshold = float(convergence_threshold)

    n_features = X1.shape[1]
    assert X1.shape[1] == X2.shape[1]

    # Determine convergence criterion
    p_values = compute_p_values(X1, X2, u_test)
    median_p_value = np.median(p_values)
    log_(f'Median p-value: {median_p_value}', verbose=verbose)

    X1 = torch.FloatTensor(X1)
    X2 = torch.FloatTensor(X2)

    # Define subspace mapping
    subspace_mapping = lambda x: x

    with torch.no_grad():
        target_mu, target_scale = compute_stats(X2)
        _, target_scale_subspace = compute_stats(subspace_mapping(X2))
        X1_second = manifold(X1)
        target_diffs = compute_s_values(X1_second)

    loss_scaling = LossScaling()

    # Compute initial transport plan
    transport_plan = BlockTransportPlan(distance, block_sizes_1, block_sizes_2)
    transport_plan.update(subspace_mapping(X1), subspace_mapping(X2))

    # Initialize NN
    adapter = MLPAdapter(X1.size()[0], X1.size()[1], tuple([nn_n_hidden] * nn_n_layers), manifold)
    is_constant = torch.logical_and(
        torch.all(X1 == X1[0, :].unsqueeze(0), dim=0),
        torch.all(X2 == X1[0, :].unsqueeze(0), dim=0)
    )
    adapter.set_constant_mask(is_constant)

    with torch.no_grad():
        adapter.init_bias(X1, torch.quantile(X2, 0.5, dim=0))

    best_state_dict = adapter.state_dict()
    best_global_obj = np.inf

    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    decay_rate = math.exp(math.log(1e-3) / max_n_iter)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=decay_rate)

    variances = []
    losses = []
    median_p_values = []
    pbar = tqdm.tqdm(range(max_n_iter), desc='OT', disable=(not verbose))
    for iteration in pbar:

        optimizer.zero_grad()

        # Update OT solution
        with torch.no_grad():
            X1_second = adapter(X1)
            transport_plan.update(subspace_mapping(X1_second), subspace_mapping(X2))

        total_loss, total_reg, total_reg2 = 0, 0, 0
        all_idx = np.arange(0, len(transport_plan))
        np.random.shuffle(all_idx)
        n_batches = int(math.ceil(len(transport_plan) / float(batch_size)))
        for idx in np.array_split(all_idx, n_batches):

            optimizer.zero_grad()

            # Adaptation on the manifold
            X1_second_batch = adapter(X1[idx, :])

            # Wasserstein distance
            loss = torch.mean(torch.square(subspace_mapping(X1_second_batch) - transport_plan[idx]) * target_scale)

            # Univariate Wasserstein distances
            loss = loss + u_loss_weight * torch.mean(univariate_wasserstein_distances(X1_second_batch * target_scale, X2 * target_scale))

            # Regularization function
            if reg_rate > 0:
                diffs_batch = (X1_second_batch - target_mu) * target_scale
                reg = torch.mean(torch.square(diffs_batch - target_diffs[idx]) * target_scale)
                loss = loss + reg_rate * reg
                total_reg += reg.item()

            # Re-scale loss function
            loss = loss_scaling(loss)

            # L2 regularization
            if l2_reg > 0:
                for param in adapter.l2_reg_parameters():
                    loss = loss + l2_reg * torch.sum(torch.square(param))

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        losses.append([total_loss, total_reg])

        # Check convergence
        #p_values = compute_p_values(X1_second.cpu().data.numpy(), X2.cpu().data.numpy(), u_test)
        #median_p_values.append(float(np.median(p_values)))
        #pbar.set_description(f'p={median_p_values[-1]:.3f}')
        #if median_p_values[-1] >= convergence_threshold:
        #    break
        #reg_rate *= 0.999

        # Check global objective
        if total_loss < best_global_obj:
            best_global_obj = total_loss
            best_state_dict = adapter.state_dict().copy()

    # Restore best NN model
    adapter.load_state_dict(best_state_dict)
    adapter.eval()
    X1_second = adapter(X1).cpu().data.numpy()

    losses = np.asarray(losses, dtype=float)

    X2 = X2.cpu().data.numpy()

    # Save figures
    if folder is not None:
        try:
            if not os.path.isdir(folder):
                os.makedirs(folder)

            p_values = compute_p_values(X1_second, X2, u_test)
            log_(f'Median p-value after correction: {np.median(p_values)}', verbose=verbose)
            plt.subplot(2, 1, 1)
            plt.violinplot(p_values, vert=False, showmeans=True, showextrema=True)
            plt.xlabel('p-value')
            plt.subplot(2, 1, 2)
            plt.violinplot(np.log(p_values), vert=False, showmeans=True, showextrema=True)
            plt.xlabel('log(p-value)')
            plt.savefig(os.path.join(folder, 'p-values-final.png'), dpi=300)
            plt.clf()
            plt.plot(p_values)
            plt.xlabel('Bin')
            plt.ylabel('p-value')
            plt.savefig(os.path.join(folder, 'p-values.png'), dpi=300)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(cdist(X1, X1))
            plt.subplot(1, 2, 2)
            plt.imshow(cdist(X1_second, X1_second))
            plt.savefig(os.path.join(folder, 'D11.png'), dpi=400)
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(cdist(X1, X2))
            plt.subplot(1, 2, 2)
            plt.imshow(cdist(X1_second, X2))
            plt.savefig(os.path.join(folder, 'D12.png'), dpi=400)
            plt.clf()
            plt.plot(median_p_values)
            plt.xlabel('Iterations')
            plt.ylabel('Median p-value')
            plt.savefig(os.path.join(folder, 'median-p-values.png'), dpi=300)
            plt.clf()
            plt.plot(losses[:, 0], label='OT loss')
            plt.plot(losses[:, 1], label='Regularization')
            plt.legend()
            plt.xlabel('Iterations')
            plt.ylabel('Loss function')
            plt.savefig(os.path.join(folder, 'loss.png'), dpi=300)
            plt.clf()

            transport_plan = BlockTransportPlan(distance, block_sizes_1, block_sizes_2)
            transport_plan.update(torch.FloatTensor(X1_second), torch.FloatTensor(X2))
            X1_barycentric = transport_plan.X2_barycentric.cpu().data.numpy()
            pca = PCA(n_components=25)
            pca.fit(X2)
            for transformer, filename in zip([KernelPCA(), TSNE()], ['kpca.png', 'tsne.png']):
                plt.subplot(1, 2, 1)
                coords = transformer.fit_transform(pca.transform(np.concatenate([X1, X2, X1_barycentric], axis=0)))
                plt.scatter(coords[:len(X1), 0], coords[:len(X1), 1], alpha=0.4)
                plt.scatter(coords[len(X1):len(X1)+len(X2), 0], coords[len(X1):len(X1)+len(X2), 1], alpha=0.4)
                plt.scatter(coords[len(X1) + len(X2):, 0], coords[len(X1) + len(X2):, 1], alpha=0.4, marker='x')
                plt.subplot(1, 2, 2)
                coords = transformer.fit_transform(pca.transform(np.concatenate([X1_second, X2, X1_barycentric], axis=0)))
                plt.scatter(coords[:len(X1), 0], coords[:len(X1), 1], alpha=0.4)
                plt.scatter(coords[len(X1):len(X1)+len(X2), 0], coords[len(X1):len(X1)+len(X2), 1], alpha=0.4)
                plt.scatter(coords[len(X1) + len(X2):, 0], coords[len(X1) + len(X2):, 1], alpha=0.4, marker='x')
                plt.savefig(os.path.join(folder, filename), dpi=300)
                plt.clf()

        except (ValueError, IndexError):
            pass

    # Fix numerical issues
    mask = np.logical_or(np.isnan(X1_second), np.isinf(X1_second))
    if verbose:
        if np.any(mask):
            print('WARNING: Output contains NaNs. Replacing them with original data.')
    X1_second[mask] = X1[mask]

    return X1_second, adapter


def ot_da(*args, **kwargs) -> np.ndarray:
    X1_second, adapter = _ot_da(*args, **kwargs)
    return X1_second


def train_adapter(*args, **kwargs) -> MLPAdapter:
    X1_second, adapter = _ot_da(*args, **kwargs)
    return adapter


class DomainAdapter:

    def __init__(self, **kwargs):
        self.kwargs: Dict  = kwargs
        self.adapter: Optional[MLPAdapter] = None
        self.feature_mask: Optional[np.ndarray] = None

    def feature_filtering(
            self,
            X: Union[List[np.ndarray], np.ndarray],
            Y: Union[None, List[np.ndarray], np.ndarray],
            fit: bool = False
    ) -> Tuple[Union[List[np.ndarray], np.ndarray], Union[None, List[np.ndarray], np.ndarray]]:
        if isinstance(X, list):
            X_cat = np.concatenate(X, axis=0)
            if Y is not None:
                Y_cat = np.concatenate(Y, axis=0)
            else:
                Y_cat = None
        else:
            X_cat, Y_cat = X, Y
        if fit or (self.feature_mask is None):
            self.feature_mask = ~np.logical_and(
                np.all(X_cat == X_cat[np.newaxis, 0, :], axis=0),
                np.all(Y_cat == X_cat[np.newaxis, 0, :], axis=0)
            )
        if isinstance(X, list):
            X = [arr[:, self.feature_mask] for arr in X]
            if Y is not None:
                Y = [arr[:, self.feature_mask] for arr in Y]
            else:
                Y = None
            return X, Y
        else:
            if Y is not None:
                return X[:, self.feature_mask], Y[:, self.feature_mask]
            else:
                return X[:, self.feature_mask], Y

    def pad(self, X_original: np.ndarray, X: np.ndarray) -> np.ndarray:
        if X_original.shape == X.shape:
            return X
        else:
            X_out = np.copy(X_original)
            X_out[:, self.feature_mask] = X
            return X_out

    def fit(self, X: Union[List[np.ndarray], np.ndarray], Y: Union[List[np.ndarray], np.ndarray]) -> 'DomainAdapter':
        X_sub, Y_sub = self.feature_filtering(X, Y, fit=True)
        X_adapted, adapter = _ot_da(X_sub, Y_sub, **self.kwargs)
        assert X_adapted.shape[1] == int(np.sum(self.feature_mask))
        self.adapter = adapter
        return self

    def fit_transform(self, X: Union[List[np.ndarray], np.ndarray], Y: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        X_sub, Y_sub = self.feature_filtering(X, Y, fit=True)
        X_adapted, adapter = _ot_da(X_sub, Y_sub, **self.kwargs)
        self.adapter = adapter
        if isinstance(X, list):
            ends = list(np.cumsum([len(mat) for mat in X]))
            starts = [0] + ends[:-1]
            X_adapted = [self.pad(mat, X_adapted[start:end, :]) for mat, start, end in zip(X, starts, ends)]
        return X_adapted

    def transform(self, X: Union[List[np.ndarray], np.ndarray]) -> Union[List[np.ndarray], np.ndarray]:
        X_sub, _ = self.feature_filtering(X, None, fit=False)
        if isinstance(X, list):
            return [self.pad(mat1, self.adapter.adapt(mat2)) for mat1, mat2 in zip(X, X_sub)]
        else:
            return self.pad(X, self.adapter.adapt(X_sub))
