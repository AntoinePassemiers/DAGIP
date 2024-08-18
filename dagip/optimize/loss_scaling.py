# -*- coding: utf-8 -*-
#
#  loss_scaling.py
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

from typing import Optional

import torch


class LossScaling(object):

    def __init__(self):
        self.scale: Optional[float] = None

    def __call__(self, loss: torch.Tensor) -> torch.Tensor:
        if self.scale is None:
            value = loss.item()
            if value == 0.0:
                self.scale = 1.0
            else:
                self.scale = 1. / value
        return self.scale * loss
