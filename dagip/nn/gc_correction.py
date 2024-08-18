# -*- coding: utf-8 -*-
#
#  gc_correction.py: Differentiable GC-content bias correction
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

import random
from typing import Tuple, Optional

import numpy as np
import torch
from statsmodels.nonparametric.smoothers_lowess import lowess


class Lowess(torch.autograd.Function):

    GRAD_GC: Optional[torch.Tensor] = None

    @staticmethod
    def forward(
            ctx,
            endog: torch.Tensor,
            exog: torch.Tensor,
            frac: float
    ) -> torch.Tensor:

        out = torch.clone(endog)
        for i in range(len(endog)):
            endog_i = endog[i, :].cpu().data.numpy()
            exog_i = exog.cpu().data.numpy()

            # Ignore NaNs and infs
            mask = np.logical_or(np.isnan(endog_i), np.isinf(endog_i))
            mask = ~np.logical_or(mask, endog_i == 0)

            # LOWESS regression
            out[i, :] = torch.FloatTensor(lowess(
                endog_i[mask],
                exog_i[mask],
                xvals=exog_i,
                frac=frac,
                return_sorted=False
            ))

        ctx.save_for_backward(exog)

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:

        x, = ctx.saved_tensors

        if (Lowess.GRAD_GC is None) or (len(Lowess.GRAD_GC) != len(x)):
            Lowess.store_grad_gc(x.cpu().data.numpy())

        grad_input = grad_output * Lowess.GRAD_GC.unsqueeze(0)

        return grad_input, None, None

    @staticmethod
    def store_grad_gc(x: np.ndarray):
        n = len(x)
        grad_gc = np.zeros(n)
        for i in range(n):
            w = np.abs(x - x[i])
            x_bar = np.average(x, weights=w)

            # Partial derivative of the slope w.r.t. Y[k, i]
            numerator = np.average((x - x_bar) * ((np.arange(n) == i) - 1. / n), weights=w)
            denominator = np.average((x - x_bar) ** 2, weights=w)
            grad_a = numerator / denominator

            grad_gc[i] = ((x[i] - x_bar) * grad_a + 1. / n)
        Lowess.GRAD_GC = torch.FloatTensor(grad_gc)


def diff_gc_correction(X: torch.Tensor, exog: torch.Tensor, frac: float = 0.3) -> torch.Tensor:
    y_pred = Lowess.apply(X, exog, frac)
    mask = (y_pred == 0)
    y_pred = mask + (~mask) * y_pred

    X_adapted = X / y_pred

    mask = (exog == 0).unsqueeze(0)
    X_adapted = mask * X + (~mask) * X_adapted

    return X_adapted
