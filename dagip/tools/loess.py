# -*- coding: utf-8 -*-
#
#  loess.py
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

import numpy as np
import rpy2
import rpy2.rinterface_lib.callbacks
import rpy2.robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None


_loess = rpy2.robjects.r('''
function(endog, exog) {
    lo <- loess(endog ~ exog, data.frame(endog=endog, exog=exog, degree=1))
    res <- predict(lo, data.frame(exog=exog), se=TRUE)
    res$fit
}
''')


def loess(endog: np.ndarray, exog: np.ndarray) -> np.ndarray:
    return np.asarray(_loess(endog, exog))
