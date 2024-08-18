# -*- coding: utf-8 -*-
#
#  ichor_cna.py
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

import math
import os
from typing import Any, Dict, Optional

import numpy as np
import scipy.stats

from dagip.nipt.binning import ChromosomeBounds


def find_equiprobable_point_(mu1, mu2, s1, s2, nu1, nu2) -> float:
    if s1 == s2:
        return 0.5 * (mu1 + mu2)
    a = nu2 * s2 ** 2 - nu1 * s1 ** 2
    b = 2 * (nu1 * mu2 * s1 ** 2 - nu2 * mu1 * s2 ** 2)
    c = nu2 * (mu1 * s2) ** 2 - nu1 * (mu2 * s1) ** 2
    sqrt_delta = math.sqrt(b ** 2 - 4 * a * c)
    x1 = (-b + sqrt_delta) / (2 * a)
    x2 = (-b - sqrt_delta) / (2 * a)
    if (mu1 <= x1 <= mu2) or (mu2 <= x1 <= mu1):
        return x1
    else:
        assert (mu1 <= x2 <= mu2) or (mu2 <= x2 <= mu1)
        return x2


class HMM:

    def __init__(self, n_states: int = 8):
        self.n_states: int = n_states
        self.mu = np.zeros(n_states)
        self.sigma = np.ones(n_states)
        self.nu = 2.1

        e = 0.9999
        a = np.full((self.n_states, self.n_states), (1. - e) / (self.n_states - 1))
        np.fill_diagonal(a, e)
        self.log_a = np.log(a)

    def find_equiprobable_point(self, i: int, j: int) -> float:

        return find_equiprobable_point_(self.mu[i], self.mu[j], self.sigma[i], self.sigma[j], self.nu, self.nu)

        assert i != j
        assert self.mu[i] != self.mu[j]
        if i < j:
            assert self.mu[i] <= x <= self.mu[j]
        else:
            assert self.mu[j] <= x <= self.mu[i]
        return x

    def viterbi(self, l: np.ndarray) -> np.ndarray:
        n = len(l)
        log_b = scipy.stats.t.logpdf(l[:, np.newaxis], self.nu, loc=self.mu[np.newaxis, :], scale=self.sigma[np.newaxis, :])

        t1 = np.empty((n, self.n_states), dtype=float)
        t2 = np.empty((n, self.n_states), dtype=int)

        # Initial states
        for i in range(self.n_states):
            t1[0, i] = log_b[0, i]
            t2[0, i] = 0

        # State transitions
        tmp = np.empty(self.n_states)
        for t in range(1, n):
            for i in range(self.n_states):
                for k in range(self.n_states):
                    tmp[k] = t1[t - 1, k] + self.log_a[k, i] + log_b[t, i]
                best_k = np.argmax(tmp)
                t1[t, i] = tmp[best_k]
                t2[t, i] = best_k

        # End state is the state that maximizes log-likelihood
        best_k = np.argmax(t1[n - 1, :])
        z = best_k

        # Traceback
        x = np.empty(n, dtype=int)
        x[-1] = z
        for t in range(n - 1, 0, -1):
            z = t2[t, z]
            x[t - 1] = z

        return x

    def naive_decode(self, l: np.ndarray) -> np.ndarray:
        log_b = scipy.stats.t.logpdf(
            l[:, np.newaxis], self.nu, loc=self.mu[np.newaxis, :],
            scale=self.sigma[np.newaxis, :]
        )
        return np.argmax(log_b, axis=1)

    def decode(self, l: np.ndarray, mappability: np.ndarray, centromeric: np.ndarray) -> np.ndarray:
        states = np.full(len(l), 2, dtype=int)

        mask = np.logical_and(mappability >= 0.9, ~centromeric)
        # states[mask] = self.viterbi(l[mask])
        states[mask] = self.naive_decode(l[mask])

        states[states == 6] = 1
        states[states == 7] = 3
        return states


def create_wig_file(filepath: str, x: np.ndarray):
    chromosomes = ChromosomeBounds.separate_chromosomes_1mb(x)
    with open(filepath, 'w') as f:
        for i, chromosome in enumerate(chromosomes):
            f.write(f'fixedStep chrom=chr{i + 1} start=1 step=1000000 span=1000000\n')
            for value in chromosome:
                f.write(f'{value}\n')


def create_ichor_cna_normal_panel(
        ichor_cna_location: str,
        folder: str,
        X: np.ndarray,
        gc_content: np.ndarray,
        mappability: np.ndarray
):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    gc_filepath = os.path.join(folder, 'gc.wig')
    map_filepath = os.path.join(folder, 'map.wig')
    if not os.path.exists(gc_filepath):
        create_wig_file(gc_filepath, gc_content)
    if not os.path.exists(map_filepath):
        create_wig_file(map_filepath, mappability)

    wig_files_filepath = os.path.join(folder, 'wig-files.txt')
    with open(wig_files_filepath, 'w') as f:
        for i in range(len(X)):
            in_filepath = os.path.join(folder, f'normal{i + 1}.wig')
            if not os.path.exists(in_filepath):
                create_wig_file(in_filepath, X[i, :])
            f.write(f'{in_filepath}\n')
    ichor_cna_script_file = os.path.join(ichor_cna_location, 'scripts', 'createPanelOfNormals.R')
    out_filepath = os.path.join(folder, 'normal-panel')
    centromere_filepath = os.path.join(
        ichor_cna_location, 'inst', 'extdata', 'GRCh38.GCA_000001405.2_centromere_acen.txt')
    os.system(
        f'Rscript {ichor_cna_script_file} --filelist {wig_files_filepath} --gcWig {gc_filepath} '
        f'--mapWig {map_filepath} --outfile {out_filepath} '
        f'--chrs "c(1:22)" --chrNormalize "c(1:22)" --method median '
        f'--centromere {centromere_filepath} '
    )


def ichor_cna(
        ichor_cna_location: str,
        normal_panel_filepath: Optional[str],
        folder: str,
        x: np.ndarray,
        gc_content: np.ndarray,
        mappability: np.ndarray
):
    in_filepath = os.path.join(folder, 'input.wig')
    gc_filepath = os.path.join(folder, 'gc.wig')
    map_filepath = os.path.join(folder, 'map.wig')

    if not os.path.isdir(folder):
        os.makedirs(folder)
    if not os.path.exists(in_filepath):
        create_wig_file(in_filepath, x)
    if not os.path.exists(gc_filepath):
        create_wig_file(gc_filepath, gc_content)
    if not os.path.exists(map_filepath):
        create_wig_file(map_filepath, mappability)
    ichor_cna_script_file = os.path.join(ichor_cna_location, 'scripts', 'runIchorCNA.R')
    centromere_filepath = os.path.join(
        ichor_cna_location, 'inst', 'extdata', 'GRCh38.GCA_000001405.2_centromere_acen.txt')
    cmd = f'Rscript {ichor_cna_script_file} --WIG {in_filepath} --gcWig {gc_filepath} ' \
        f'--mapWig {map_filepath} --outDir {folder} ' \
        f'--chrs "c(1:22)" --chrTrain "c(1:18,20:22)" --chrNormalize "c(1:22)" --estimateNormal TRUE ' \
        f'--maxCN 5 --scStates "c(1, 3)" ' \
        f'--ploidy 2 --estimatePloidy TRUE ' \
        f'--txnE 0.9999 --txnStrength 1e+04 ' \
        f'--estimateScPrevalence TRUE --normal "c(0.75,0.85,0.9,0.95,0.97,0.99)" ' \
        f'--centromere {centromere_filepath} --includeHOMD TRUE '
    if normal_panel_filepath is not None:
        cmd += f'--normalPanel {normal_panel_filepath}'
    os.system(cmd)


def load_ichor_cna_results(folder: str, bin_size: int = 1000000) -> Dict[str, Any]:
    results = {
        'model': HMM(),
        'tumor-fraction': None,
        'tumor-ploidy': None,
        'tumor-cellular-prevalence': None,
        'proportion-subclonal-cnas': None,
        'copy-number': None,
        'log-r': None,
        'success': False
    }

    if not os.path.exists(os.path.join(folder, 'test.params.txt')):
        return results
    with open(os.path.join(folder, 'test.params.txt'), 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if line.startswith('Tumor Fraction:'):
                results['tumor-fraction'] = float(line.split(':')[1])
            elif line.startswith('Ploidy:'):
                results['tumor-ploidy'] = float(line.split(':')[1])
            elif line.startswith('Subclone Fraction:'):
                value = line.split(':')[1].strip()
                results['tumor-cellular-prevalence'] = np.nan if (value == 'NA') else float(value)
            elif line.startswith('Fraction CNA Subclonal:'):
                results['proportion-subclonal-cnas'] = float(line.split(':')[1])
            elif line.startswith('Student\'s t mean:'):
                elements = line.split(':')[1].strip().split(',')
                for i in range(len(elements)):
                    results['model'].mu[i] = float(elements[i])
            elif line.startswith('Student\'s t precision:'):
                elements = line.split(':')[1].strip().split(',')
                for i in range(len(elements)):
                    results['model'].sigma[i] = 1. / float(elements[i])

    if not os.path.exists(os.path.join(folder, 'test.cna.seg')):
        return results
    bounds = ChromosomeBounds.get_1mb()
    profile = [np.full(end - start, -1, dtype=int) for start, end in zip(bounds[:-1], bounds[1:])]
    log_r = [np.full(end - start, 0, dtype=float) for start, end in zip(bounds[:-1], bounds[1:])]
    with open(os.path.join(folder, 'test.cna.seg'), 'r') as f:
        for line in f.readlines()[1:]:
            line = line.rstrip()
            elements = line.split('\t')
            assert len(elements) >= 4
            chr_id = int(elements[0]) - 1
            bin_id = (int(elements[1]) - 1) // bin_size
            status = int(elements[3])
            profile[chr_id][bin_id] = status
            log_r[chr_id][bin_id] = 0 if (elements[5] == 'NA') else float(elements[5])
    results['copy-number'] = np.concatenate(profile, axis=0)
    results['log-r'] = np.concatenate(log_r, axis=0)

    results['success'] = True
    for key in results.keys():
        if key is None:
            results['success'] = False
    return results
