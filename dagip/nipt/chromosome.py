# -*- coding: utf-8 -*-
#
#  chromosome.py
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


class Chromosome:

    def __init__(
            self,
            chr_id: int,
            bin_size: int,
            gc_content: np.ndarray,
            mappability: np.ndarray,
            centromeric: np.ndarray
    ):
        self.chr_id: int = chr_id
        self.bin_size: int = bin_size
        self.gc_content: np.ndarray = gc_content
        self.mappability: np.ndarray = mappability
        self.centromeric: np.ndarray = centromeric

