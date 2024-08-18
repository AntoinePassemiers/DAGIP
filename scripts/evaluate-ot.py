import argparse
import collections
import os
import json
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import ot
import ot.da
import numpy as np
import scipy.stats
import pandas as pd
import lightgbm
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

from dagip.core import ot_da, DomainAdapter, ot_emd_with_nuclear_norm_reg
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction.positive import Positive
from dagip.retraction.gip import GIPManifold
from dagip.stats.bounds import compute_theoretical_bounds
from dagip.stats.r2 import r2_coefficient
from dagip.tools.dryclean import run_dryclean


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)


def main(DATASET: str) -> None:

    print(DATASET)

    # Load reference GC content
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc-content-1000kb.csv'))
    gc_content = df['MEAN'].to_numpy()
    bin_chr_names = df['CHR'].to_numpy()
    bin_starts = df['START'].to_numpy()
    bin_ends = df['END'].to_numpy()

    # Load data
    if DATASET == 'OV':
        data = np.load(os.path.join(DATA_FOLDER, 'numpy', 'OV.npz'), allow_pickle=True)
    else:
        data = np.load(os.path.join(DATA_FOLDER, 'numpy', 'NIPT.npz'), allow_pickle=True)
    gc_codes = data['gc_codes']
    gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
    paired_with = data['paired_with']
    X = data['X']
    y = data['y']
    d = data['d']

    # Remove problematic regions
    mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
    X = X[:, mask]
    gc_content = gc_content[mask]
    bin_chr_names = bin_chr_names[mask]
    bin_starts = bin_starts[mask]
    bin_ends =  bin_ends[mask]

    # GC-correction
    X = gc_correction(X, gc_content)

    # Load sample pairs
    idx1, idx2, used = [], [], set()
    for i in range(len(X)):
        if paired_with[i]:
            j = gc_code_dict[paired_with[i]]
            valid = (i not in used)
            if (DATASET == 'OV') and (y[i] != 'OV'):
                valid = False
            if (DATASET == 'NIPT-lib') and (d[i] not in {'D1a', 'D1b'}):
                valid = False
            if (DATASET == 'NIPT-adapter') and (d[i] not in {'D2a', 'D2b'}):
                valid = False
            if (DATASET == 'NIPT-hs2000') and (d[i] not in {'D3a', 'D3b'}):
                valid = False
            if (DATASET == 'NIPT-hs2500') and (d[i] not in {'D4a', 'D4b'}):
                valid = False
            if (DATASET == 'NIPT-hs4000') and (d[i] not in {'D5a', 'D5b'}):
                valid = False
            if (DATASET == 'NIPT-chemistry') and (d[i] not in {'D6a', 'D6b'}):
                valid = False
            if valid:
                idx1.append(i)
                idx2.append(j)
                used.add(i)
                used.add(j)
    idx1 = np.asarray(idx1, dtype=int)
    idx2 = np.asarray(idx2, dtype=int)

    X_ces = np.copy(X)
    target_scaler = RobustScaler()
    target_scaler.fit(X[idx2, :])
    source_scaler = RobustScaler()
    source_scaler.fit(X[idx1, :])
    X_ces[idx1, :] = target_scaler.inverse_transform(source_scaler.transform(X[idx1, :]))

    X_dry = np.load(os.path.join(RESULTS_FOLDER, 'corrected', 'dryclean', f'{DATASET}-corrected.npy'))

    C = np.square(cdist(X[idx1, :], X[idx2, :]))
    C_ces = np.square(cdist(X_ces[idx1, :], X_ces[idx2, :]))
    C_dry = np.square(cdist(X_dry[idx1, :], X_dry[idx2, :]))

    n = C.shape[0]
    m = C.shape[1]
    a = np.full(n, 1. / n)
    b = np.full(m, 1. / m)

    reg_rate = 0.1 * np.mean(C)

    gamma = ot.emd(a, b, C)

    is_correct = np.logical_and(
        np.argmin(C, axis=0) == np.arange(len(C)),
        np.argmin(C, axis=1) == np.arange(len(C)),
    )
    is_correct2 = np.logical_and(
        np.argmin(C_ces, axis=0) == np.arange(len(C)),
        np.argmin(C_ces, axis=1) == np.arange(len(C)),
    )
    is_correct3 = np.logical_and(
        np.argmin(C_dry, axis=0) == np.arange(len(C)),
        np.argmin(C_dry, axis=1) == np.arange(len(C)),
    )
    is_correct4 = np.logical_and(
        np.argmax(gamma, axis=0) == np.arange(len(gamma)),
        np.argmax(gamma, axis=1) == np.arange(len(gamma)),
    )

    if DATASET == 'OV':
        domain0_name = r'$\mathcal{D}_{9}$ protocol'
        domain1_name = r'$\mathcal{D}_{10}$ protocol'
    elif DATASET == 'NIPT-lib':
        domain0_name = r'TruSeq Nano kit ($\mathcal{D}_{1,a}$)'
        domain1_name = r'Kapa HyperPrep kit ($\mathcal{D}_{1,b}$)'
    elif DATASET == 'NIPT-adapter':
        domain0_name = r'IDT indexes ($\mathcal{D}_{2,a}$)'
        domain1_name = r'Kapa dual indexes ($\mathcal{D}_{2,b}$)'
    elif DATASET == 'NIPT-hs2000':
        domain0_name = r'HiSeq 2000 ($\mathcal{D}_{3,a}$)'
        domain1_name = r'NovaSeq ($\mathcal{D}_{3,b}$)'
    elif DATASET == 'NIPT-hs2500':
        domain0_name = r'HiSeq 2500 ($\mathcal{D}_{4,a}$)'
        domain1_name = r'NovaSeq ($\mathcal{D}_{4,b}$)'
    elif DATASET == 'NIPT-hs4000':
        domain0_name = r'HiSeq 4000 ($\mathcal{D}_{5,a}$)'
        domain1_name = r'NovaSeq ($\mathcal{D}_{5,b}$)'
    elif DATASET == 'NIPT-chemistry':
        domain0_name = r'NovaSeq V1 chemistry ($\mathcal{D}_{6,a}$)'
        domain1_name = r'NovaSeq V1.5 chemistry ($\mathcal{D}_{6,b}$)'
    else:
        raise NotImplementedError()

    kwargs = dict(cmap='GnBu', aspect='equal')

    plt.figure(figsize=(10, 8))

    ax = plt.subplot(2, 2, 1)
    im = ax.imshow(C, **kwargs)
    ax.set_title(f'Baseline\nPairwise distances\nAccuracy: {np.sum(is_correct)} / {len(gamma)}')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel(domain0_name)
    ax.set_ylabel(domain1_name)
    ax.figure.colorbar(im, ax=ax)

    ax = plt.subplot(2, 2, 2)
    im = ax.imshow(C_ces, **kwargs)
    ax.set_title(f'Center-and-scale\nPairwise distances\nAccuracy: {np.sum(is_correct2)} / {len(gamma)}')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel(domain0_name)
    ax.set_ylabel(domain1_name)
    ax.figure.colorbar(im, ax=ax)

    ax = plt.subplot(2, 2, 3)
    im = ax.imshow(C_dry, **kwargs)
    ax.set_title(f'dryclean\nPairwise distances\nAccuracy: {np.sum(is_correct3)} / {len(gamma)}')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel(domain0_name)
    ax.set_ylabel(domain1_name)
    ax.figure.colorbar(im, ax=ax)

    ax = plt.subplot(2, 2, 4)
    kwargs['cmap'] = 'BuPu'
    im = ax.imshow(gamma, **kwargs)
    ax.set_title(f'Optimal transport\nTransport plan\nAccuracy: {np.sum(is_correct4)} / {len(gamma)}')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel(domain0_name)
    ax.set_ylabel(domain1_name)
    ax.figure.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f'cdist-{DATASET}.png'), dpi=400)
    plt.cla()
    plt.clf()

    return [
        np.mean(is_correct),
        np.mean(is_correct2),
        np.mean(is_correct3),
        np.mean(is_correct4),
    ]


DATASETS = ['OV', 'NIPT-chemistry', 'NIPT-lib', 'NIPT-adapter', 'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000']
with open(os.path.join(RESULTS_FOLDER, 'ot-acc.csv'), 'w') as f:
    f.write(',Baseline,Center-and-scale,dryclean,Optimal transport\n')
    for dataset in DATASETS:
        res = main(dataset)
        f.write(dataset + ',' + ','.join([str(x) for x in res]) + '\n')
