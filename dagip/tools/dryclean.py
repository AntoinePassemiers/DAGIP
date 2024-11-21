# -*- coding: utf-8 -*-
#
#  dryclean.py
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

import os
from typing import List

import numpy as np
import rpy2
import rpy2.rinterface_lib.callbacks
import rpy2.robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr


def run_dryclean(
        bin_chr_names: List[str],
        bin_starts: List[int],
        bin_ends: List[int],
        normals: np.ndarray,
        samples: np.ndarray,
        tmp_folder: str
) -> np.ndarray:

    # Load dryclean
    rpy2.robjects.numpy2ri.activate()
    rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None
    importr('dplyr')
    importr('GenomicRanges')
    importr('dryclean')
    _dryclean = rpy2.robjects.r('''
    function(normal_filepaths, test_filepath) {
        pon_object = pon$new(
            create_new_pon = TRUE, 
            normal_vector = normal_filepaths,
            field = "reads.corrected",
            build = "hg38",
            wgs = TRUE,
            target_resolution = 1000000,
            all.chr = as.character(1:22),
            nochr = TRUE,
            num.cores = 4,
            verbose = 0
        )
        dryclean_object <- dryclean$new(pon = pon_object)
        res <- dryclean_object$clean(cov = test_filepath)
        chr_names <- seqnames(res)
        chr_names <- with(chr_names, rep(runValue(chr_names), runLength(chr_names)))
        list(foreground=mcols(res)$foreground, chr_names=chr_names, starts=start(res), ends=end(res))
    }
    ''')


    os.makedirs(tmp_folder, exist_ok=True)
    os.makedirs(os.path.join(tmp_folder, 'pon'), exist_ok=True)

    rpy2.robjects.r.assign('r_chr_names', rpy2.robjects.vectors.StrVector(bin_chr_names))
    rpy2.robjects.r.assign('r_starts', rpy2.robjects.vectors.IntVector(bin_starts))
    rpy2.robjects.r.assign('r_ends', rpy2.robjects.vectors.IntVector(bin_ends))
    rpy2.robjects.r.assign('r_strands', rpy2.robjects.vectors.StrVector(['*'] * len(bin_chr_names)))

    normal_filepaths = []
    for i in range(len(normals)):
        rpy2.robjects.r.assign('r_reads_corrected', rpy2.robjects.FloatVector(normals[i, :]))
        rpy2.robjects.r('gr <- GRanges(seqnames=r_chr_names, ranges=IRanges(r_starts, end=r_ends), strand=r_strands, reads.corrected=r_reads_corrected)')
        filepath = os.path.join(tmp_folder, 'pon', f'sample{i + 1}.rds').replace('\\', '/')
        rpy2.robjects.r(f"saveRDS(gr, file='{filepath}')")
        normal_filepaths.append(filepath)
    normal_filepaths = rpy2.robjects.vectors.StrVector(normal_filepaths)

    output = np.copy(samples)
    for i in range(len(samples)):
        rpy2.robjects.r.assign('r_reads_corrected', rpy2.robjects.FloatVector(samples[i, :]))
        rpy2.robjects.r('gr <- GRanges(seqnames=r_chr_names, ranges=IRanges(r_starts, end=r_ends), strand=r_strands, reads.corrected=r_reads_corrected)')
        rpy2.robjects.r('print(gr)')
        filepath = os.path.join(tmp_folder, 'test.rds').replace('\\', '/')
        rpy2.robjects.r(f"saveRDS(gr, file='{filepath}')")

        res = _dryclean(normal_filepaths, filepath)
        new_values = res.rx2('foreground')
        new_chr_names = res.rx2('chr_names')
        new_starts = res.rx2('starts')
        new_ends = res.rx2('ends')

        new_map = {f'{chr_name}-{start}-{end}': i for i, (chr_name, start, end) in enumerate(zip(bin_chr_names, bin_starts, bin_ends))}
        for chr_name, start, end, value in zip(new_chr_names, new_starts, new_ends, new_values):
            key = f'chr{chr_name}-{start}-{end}'
            if key in new_map:
                j = new_map[key]
                output[i, j] = float(value)

    return output
