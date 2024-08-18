import argparse
import os
import sys
import json
import tqdm
from typing import Tuple, List, Optional, Dict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))
sys.path.append('/lustre1/project/stg_00019/research/Antoine/dependencies')

import ot.da
import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, ks_2samp
import scipy.spatial
from sklearn.metrics import auc
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, LabelEncoder

from dagip.core import ot_da, DomainAdapter
from dagip.tools.dryclean import run_dryclean
from dagip.correction.gc import gc_correction
from dagip.nipt.binning import ChromosomeBounds
from dagip.retraction import *
from dagip.plot import plot_end_motif_freqs, scatterplot_with_sample_importances, loo_influence_analysis


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
OUT_FOLDER = os.path.join(ROOT, '..', 'figures')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'corrected')

parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'HEMA', 'OV',
        #'NIPT-chemistry', 'NIPT-lib', 'NIPT-adapter', 'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000',
        'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions'
    ],
    help='Dataset name'
)
args = parser.parse_args()
DATASET = args.dataset

# Load reference GC content
df = pd.read_csv(os.path.join(DATA_FOLDER, 'gc-content-1000kb.csv'))
gc_content = df['MEAN'].to_numpy()
bin_chr_names = df['CHR'].to_numpy()
bin_starts = df['START'].to_numpy()
bin_ends = df['END'].to_numpy()


# Load data
if DATASET not in {'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions'}:

    manifold = Positive()

    if DATASET in {'OV-forward', 'OV-backward', 'OV'}:
        filename = 'OV.npz'
    elif DATASET == 'HEMA':
        filename = 'HEMA.npz'
    else:
        raise NotImplementedError()
    data = np.load(os.path.join(ROOT, DATA_FOLDER, 'numpy', filename), allow_pickle=True)
    gc_codes = data['gc_codes']
    gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
    paired_with = data['paired_with']
    X = data['X']
    labels = data['y']
    d = data['d']
    y = (labels != 'Healthy').astype(int)

    print('labels: ', set(labels))

    # Remove problematic regions
    mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
    X = X[:, mask]
    gc_content = gc_content[mask]
    bin_chr_names = bin_chr_names[mask]
    bin_starts = bin_starts[mask]
    bin_ends =  bin_ends[mask]

    if DATASET == 'HEMA':
        idx1 = np.where(np.logical_and(d == 'D7', labels == 'Healthy'))[0]
        idx2 = np.where(np.logical_and(d == 'D8', labels == 'Healthy'))[0]
    elif DATASET == 'OV':
        idx1 = np.where(np.logical_and(d == 'D9', labels == 'Healthy'))[0]
        idx2 = np.where(np.logical_and(d == 'D10', labels == 'Healthy'))[0]
    else:
        idx1, idx2 = [], []
        for i in range(len(X)):
            if paired_with[i]:
                j = gc_code_dict[paired_with[i]]
                idx1.append(i)
                idx2.append(j)
        idx1 = np.asarray(idx1, dtype=int)
        idx2 = np.asarray(idx2, dtype=int)

    # GC correction
    X = gc_correction(X, gc_content)

else:

    with open(os.path.join(DATA_FOLDER, 'D11-D12-batches.json'), 'r') as f:
        batch_dict = json.load(f)

    def load_cohort(folder: str) -> Tuple[np.ndarray, List[str], List[str]]:
        X = []
        ega_ids = []
        batch_ids = []
        for filename in os.listdir(folder):
            ega_id = os.path.splitext(filename)[0]
            if not filename.endswith('.csv'):
                continue
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath, sep=',')
            X.append(df.values[:, -1])
            batch_ids.append(batch_dict[ega_id])
            ega_ids.append(ega_id)
        return np.asarray(X, dtype=float), batch_ids, ega_ids


    X_cases_D11, batch_cases_D11, ega_cases_D11 = load_cohort(os.path.join(DATA_FOLDER, 'D11', 'cases', DATASET))
    X_controls_D11, batch_controls_D11, ega_controls_D11 = load_cohort(os.path.join(DATA_FOLDER, 'D11', 'controls', DATASET))
    X_controls_D12, batch_controls_D12, ega_controls_D12 = load_cohort(os.path.join(DATA_FOLDER, 'D12', 'controls', DATASET))
    gc_codes = np.concatenate([ega_cases_D11, ega_controls_D11, ega_controls_D12], axis=0)

    X = np.concatenate([X_cases_D11, X_controls_D11, X_controls_D12], axis=0)
    y = np.concatenate([np.ones(len(X_cases_D11), dtype=int), np.zeros(len(X_controls_D11) + len(X_controls_D12), dtype=int)])
    d = np.asarray(['D11'] * (len(X_cases_D11) + len(X_controls_D11)) + ['D12'] * len(X_controls_D12), dtype=object)
    groups = LabelEncoder().fit_transform(batch_cases_D11 + batch_controls_D11 + batch_controls_D12)
    labels = np.asarray(['BRCA'] * len(X_cases_D11) + ['Healthy'] * (len(X_controls_D11) + len(X_controls_D12)), dtype=object)

    idx1 = np.where(np.logical_and(y == 0, d == 'D11'))[0]
    idx2 = np.where(np.logical_and(y == 0, d == 'D12'))[0]

    is_constant = np.all(X == X[0, np.newaxis, :], axis=0)
    X = X[:, ~is_constant]

    if DATASET in {'fragment-length-distributions', 'end-motif-frequencies'}:

        if DATASET == 'fragment-length-distributions':
            X = X[:, 40:]

        X = X / np.sum(X, axis=1)[:, np.newaxis]

# Get cancer stage info
df = pd.read_csv(os.path.join(DATA_FOLDER, 'metadata.csv'))
cancer_stage_dict = {gc_code: cancer_stage for gc_code, cancer_stage in zip(df['ID'], df['CancerStage'])}
cancer_stages = np.asarray([cancer_stage_dict[gc_code] for gc_code in gc_codes], dtype=object)
cancer_stages[cancer_stages == 'IA'] = 'I'
cancer_stages[cancer_stages == 'IB'] = 'I'
cancer_stages[cancer_stages == 'IIA'] = 'II'
cancer_stages[cancer_stages == 'IIB'] = 'II'
cancer_stages[cancer_stages == 'IIIA'] = 'III'
cancer_stages[cancer_stages == 'IIIC'] = 'III'
assert len(cancer_stages) == len(X)

# Center-and-scale
scaler = RobustScaler()
scaler.fit(X[idx1])
X_ces = scaler.transform(X)
X_ces[idx2, :] = RobustScaler().fit_transform(X[idx2])

# Dryclean
if DATASET not in {'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions'}:
    if not os.path.exists(os.path.join(RESULTS_FOLDER, 'dryclean', f'{DATASET}-corrected.npy')):
        os.makedirs(os.path.join(RESULTS_FOLDER, 'dryclean'), exist_ok=True)
        X_dry = np.copy(X)
        X_dry[idx1, :] = run_dryclean(bin_chr_names, bin_starts, bin_ends, X[idx1, :], X[idx1, :], 'tmp-dryclean')
        X_dry[idx2, :] = run_dryclean(bin_chr_names, bin_starts, bin_ends, X[idx2, :], X[idx2, :], 'tmp-dryclean')
        np.save(os.path.join(RESULTS_FOLDER, 'dryclean', f'{DATASET}-corrected.npy'), X_dry)
    else:
        X_dry = np.load(os.path.join(RESULTS_FOLDER, 'dryclean', f'{DATASET}-corrected.npy'))
else:
    X_dry = X

# MappingTransport
if not os.path.exists(os.path.join(RESULTS_FOLDER, 'mapping-transport', f'{DATASET}-corrected.npy')):
    os.makedirs(os.path.join(RESULTS_FOLDER, 'mapping-transport'), exist_ok=True)
    X_mt = np.copy(X)
    model = ot.da.MappingTransport()
    model.fit(
        Xs=X[idx1, :],
        ys=np.zeros(len(idx1), dtype=int),
        Xt=X[idx2, :]
    )
    X_mt[idx1, :] = model.transform(X[idx1, :])
    np.save(os.path.join(RESULTS_FOLDER, 'mapping-transport', f'{DATASET}-corrected.npy'), X_mt)
else:
    X_mt = np.load(os.path.join(RESULTS_FOLDER, 'mapping-transport', f'{DATASET}-corrected.npy'))

# Domain adaptation
if not os.path.exists(os.path.join(RESULTS_FOLDER, 'da', f'{DATASET}-corrected.npy')):
    os.makedirs(os.path.join(RESULTS_FOLDER, 'da'), exist_ok=True)
    folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', DATASET)
    X_adapted = np.copy(X)
    adapter = DomainAdapter(folder=folder, manifold=manifold)
    X_adapted[idx1] = adapter.fit_transform(X_adapted[idx1], X_adapted[idx2])
    np.save(os.path.join(RESULTS_FOLDER, 'da', f'{DATASET}-corrected.npy'), X_adapted)
else:
    X_adapted = np.load(os.path.join(RESULTS_FOLDER, 'da', f'{DATASET}-corrected.npy'))

settings = [(0, 0, X, 'Baseline', '#a883ef', '--')]
settings.append((0, 1, X_ces, 'Center-and-scale', '#f35ee2', '-.'))
if DATASET not in {'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions'}:
    settings.append((0, 2, X_dry, 'dryclean', '#f996ab', '-.'))
settings.append((1, 0, X_mt, 'MappingTransport', '#f4c45a', '-'))
settings.append((1, 1, X_adapted, 'DAGIP', '#b1e468', '-'))

alpha = 0.5
size = 5
f, ax = plt.subplots(2, 3, figsize=(18, 12))

if DATASET == 'HEMA':
    d1, d2 = '7', '8'
elif DATASET == 'OV':
    d1, d2 = '9', '10'
elif DATASET in {'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions'}:
    d1, d2 = '11', '12'
else:
    raise NotImplementedError(f'Unknown dataset "{DATASET}"')
for i, j, X_corrected, title, color, linestyle in settings:
    X1 = X_corrected[idx1, :]
    X2 = X_corrected[idx2, :]

    transformer = TSNE(early_exaggeration=80, perplexity=7)
    coords = transformer.fit_transform(X_corrected)

    combined_labels = np.asarray([label + str(domain) for label, domain in zip(labels, d)], dtype=object)
    if DATASET == 'HEMA':
        label_dict = {
            'HLD7': 'HL (TruSeq ChIP)',
            'DLBCLD7': 'DLBCL (TruSeq ChIP)',
            'MMD7': 'MM (TruSeq ChIP)',
            'HealthyD7': 'Healthy (TruSeq ChIP)',
            'HealthyD8': 'Healthy (TruSeq Nano)'
        }
        style_dict = {
            'Healthy (TruSeq Nano)': ('#74c2c9', 's'),
            'Healthy (TruSeq ChIP)': ('#79d991', 'D'),
            'HL (TruSeq ChIP)': ('#df998f', 'o'),
            'DLBCL (TruSeq ChIP)': ('#da83c7', 'o'),
            'MM (TruSeq ChIP)': ('#c686e2', 'o'),
        }  # extra color:#94a7e2
    elif DATASET == 'OV':
        label_dict = {
            'OVD9': r'Ovarian carcinoma ($\mathcal{D}_{9}$)',
            'HealthyD9': r'Healthy ($\mathcal{D}_{9}$)',
            'OVD10': r'Ovarian carcinoma ($\mathcal{D}_{10}$)',
            'HealthyD10': r'Healthy ($\mathcal{D}_{10}$)'
        }
        style_dict = {
            r'Ovarian carcinoma ($\mathcal{D}_{9}$)': ('#da83c7', 'o'),
            r'Healthy ($\mathcal{D}_{9}$)': ('#79d991', 'o'),
            r'Ovarian carcinoma ($\mathcal{D}_{10}$)': ('#df998f', 'D'),
            r'Healthy ($\mathcal{D}_{10}$)': ('#74c2c9', 's'),
        }  # extra color:#94a7e2
    elif DATASET in {'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions'}:
        label_dict = {
            'BRCAD11': r'Breast cancer (NEBNext Enzymatic Methyl-seq)',
            'HealthyD11': r'Healthy (NEBNext Enzymatic Methyl-seq)',
            'HealthyD12': r'Healthy (KAPA HyperPrep)',
        }
        style_dict = {
            'Healthy (KAPA HyperPrep)': ('#74c2c9', 's'),
            'Healthy (NEBNext Enzymatic Methyl-seq)': ('#79d991', 'D'),
            'Breast cancer (NEBNext Enzymatic Methyl-seq)': ('#df998f', 'o'),
        }
    else:
        raise NotImplementedError()

    pretty_labels = np.asarray([label_dict[label] for label in combined_labels], dtype=object)

    add_legend = ((i == 0) and (j == 1))
    scatterplot_with_sample_importances(
        ax[i, j], ax[1, 2], X_corrected, labels != 'Healthy', d, pretty_labels, cancer_stages, style_dict,
        stage0_label='Ductal carcinoma in situ (DCIS)', legend=add_legend
    )
    ax[i, j].set_title(title, fontsize=20)
if len(settings) == 4:
    ax[0, 2].axis('off')
ax[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, f'tsne-{DATASET}.png'), dpi=300)
plt.show()
