# -*- coding: utf-8 -*-
#
#  dataset.py
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

from typing import Tuple

import numpy as np
import torch
import torch.utils.data


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, X1: np.ndarray, X2: np.ndarray, side_info: np.ndarray):
        self.X1: torch.Tensor = torch.FloatTensor(X1)
        self.X2: torch.Tensor = torch.FloatTensor(X2)
        self.side_info: torch.Tensor = torch.FloatTensor(side_info)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X1[:, idx], self.X2[:, idx], self.side_info[:, idx]

    def __len__(self) -> int:
        return self.X1.size()[1]
