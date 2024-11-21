import argparse
import collections
import os
import random
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


def plot_explained_variance(DATASET: str, ax) -> None:

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

    if DATASET == 'OV':
        domain0_name = r'$\mathcal{D}_{9}$ protocol'
        domain1_name = r'$\mathcal{D}_{10}$ protocol'
        color = 'peru'
    elif DATASET == 'NIPT-lib':
        domain0_name = r'TruSeq Nano kit ($\mathcal{D}_{1,a}$)'
        domain1_name = r'Kapa HyperPrep kit ($\mathcal{D}_{1,b}$)'
        color = 'firebrick'
    elif DATASET == 'NIPT-adapter':
        domain0_name = r'IDT indexes ($\mathcal{D}_{2,a}$)'
        domain1_name = r'Kapa dual indexes ($\mathcal{D}_{2,b}$)'
        color = 'darkviolet'
    elif DATASET == 'NIPT-hs2000':
        domain0_name = r'HiSeq 2000 ($\mathcal{D}_{3,a}$)'
        domain1_name = r'NovaSeq ($\mathcal{D}_{3,b}$)'
        color = 'teal'
    elif DATASET == 'NIPT-hs2500':
        domain0_name = r'HiSeq 2500 ($\mathcal{D}_{4,a}$)'
        domain1_name = r'NovaSeq ($\mathcal{D}_{4,b}$)'
        color = 'steelblue'
    elif DATASET == 'NIPT-hs4000':
        domain0_name = r'HiSeq 4000 ($\mathcal{D}_{5,a}$)'
        domain1_name = r'NovaSeq ($\mathcal{D}_{5,b}$)'
        color = 'forestgreen'
    elif DATASET == 'NIPT-chemistry':
        domain0_name = r'NovaSeq V1 chemistry ($\mathcal{D}_{6,a}$)'
        domain1_name = r'NovaSeq V1.5 chemistry ($\mathcal{D}_{6,b}$)'
        color = 'darkmagenta'
    else:
        raise NotImplementedError()

    # PCA
    pca = PCA()
    pca.fit(X[idx1, :])
    explained_variance1 = np.cumsum(pca.explained_variance_ / np.sum(pca.explained_variance_), axis=0)
    pca = PCA()
    pca.fit(X[idx2, :])
    explained_variance2 = np.cumsum(pca.explained_variance_ / np.sum(pca.explained_variance_), axis=0)

    xs = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = max(np.where(xs >= 0.95)[0][0] + 1, 5)

    ax.plot(explained_variance1, label=f'{domain0_name}', linestyle='-', color=color)
    ax.plot(explained_variance2, label=f'{domain1_name} (PC={n_pcs})', linestyle='--', color=color)


DATASETS = ['OV', 'NIPT-hs2500', 'NIPT-hs2000', 'NIPT-lib', 'NIPT-hs4000', 'NIPT-adapter', 'NIPT-chemistry']
plt.figure(figsize=(10, 6))
ax = plt.subplot(1, 1, 1)
for k, dataset in enumerate(DATASETS):
    plot_explained_variance(dataset, ax)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
ax.grid(alpha=0.4, color='grey', linestyle='--', linewidth=0.5)
ax.axhline(y=0.95, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Principal components')
ax.set_ylabel('Cumulative explained variance ratio')
ax.legend()
plt.tight_layout()
plt.savefig('explained-variance.png', dpi=300)
plt.show()
