# -*- coding: utf-8 -*-
#
#  utils.py
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

from typing import Dict

import numpy as np
import statsmodels.stats.api as sms


def log_(message: str, verbose: bool = True):
    if verbose:
        print(f'[DAGIP] {message}')


class LaTeXTable:

    def __init__(self):
        self.results = []

    def add(self, model_name: str, res: Dict[str, dict]):
        self.results.append((model_name, res))

    def __str__(self) -> str:

        s = ''

        # Header
        s += f'{"": <20}'
        for key in ['sensitivity', 'specificity', 'mcc', 'auroc', 'aupr']:
            s += f' & {key: <12}'
        s += ' \\\\\n'

        for model_name, res in self.results:
            s += f'{model_name: <20}'
            f = lambda x: f'{x:.3f}'.replace('0.', '.')
            for k in ['reglog', 'rf', 'svm']:

                values = []
                for metric in ['sensitivity', 'specificity', 'mcc', 'auroc', 'aupr']:
                    value = np.mean([x[metric] for x in res[k]])
                    value_str = f(value)
                    values.append(value_str)
                s += f' & ' + ' & '.join(values)
            s += ' \\\\\n'

        return s
