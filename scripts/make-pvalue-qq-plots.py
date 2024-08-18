import argparse
import os
import sys
import json
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
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
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

PRETTY_DATASET_NAMES = {
    'HEMA': 'HEMA data set\n',
    'OV': 'OV data set\n',
    'long-fragment-ratio-profiles': 'FRAG data set\nLong fragment ratio profiles',
    'nucleosome-positioning-score-profiles': 'FRAG data set\nNucleosome positioning score profiles', 
    'end-motif-frequencies': 'FRAG data set\nEnd motif histograms',
    'fragment-length-distributions': 'FRAG data set\nFragment length histograms'
}

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
        filename = 'NIPT.npz'
    data = np.load(os.path.join(ROOT, DATA_FOLDER, 'numpy', filename), allow_pickle=True)
    gc_codes = data['gc_codes']
    gc_code_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
    paired_with = data['paired_with']
    X = data['X']
    labels = data['y']
    d = data['d']
    y = (labels != 'Healthy').astype(int)

    print('labels: ', set(labels))

    if DATASET == 'OV':
        d = (d == 'D9').astype(int)
    elif DATASET == 'HEMA':
        d = (d == 'D7').astype(int)
    else:
        raise NotImplementedError()

    # Remove problematic regions
    mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
    X = X[:, mask]
    gc_content = gc_content[mask]
    bin_chr_names = bin_chr_names[mask]
    bin_starts = bin_starts[mask]
    bin_ends =  bin_ends[mask]

    # GC correction
    X = gc_correction(X, gc_content)

else:

    with open(os.path.join(DATA_FOLDER, 'D11-D12-batches.json'), 'r') as f:
        batch_dict = json.load(f)

    def load_cohort(folder: str) -> Tuple[np.ndarray, List[str]]:
        X = []
        batch_ids = []
        for filename in os.listdir(folder):
            ega_id = os.path.splitext(filename)[0]
            if not filename.endswith('.csv'):
                continue
            filepath = os.path.join(folder, filename)
            df = pd.read_csv(filepath, sep=',')
            X.append(df.values[:, -1])
            batch_ids.append(batch_dict[ega_id])
        return np.asarray(X, dtype=float), batch_ids


    X_cases_D11, batch_cases_D11 = load_cohort(os.path.join(DATA_FOLDER, 'D11', 'cases', DATASET))
    X_controls_D11, batch_controls_D11 = load_cohort(os.path.join(DATA_FOLDER, 'D11', 'controls', DATASET))
    X_controls_D12, batch_controls_D12 = load_cohort(os.path.join(DATA_FOLDER, 'D12', 'controls', DATASET))


    X = np.concatenate([X_cases_D11, X_controls_D11, X_controls_D12], axis=0)
    y = np.concatenate([np.ones(len(X_cases_D11), dtype=int), np.zeros(len(X_controls_D11) + len(X_controls_D12), dtype=int)])
    d = np.concatenate([np.zeros(len(X_cases_D11) + len(X_controls_D11), dtype=int), np.ones(len(X_controls_D12), dtype=int)])
    groups = LabelEncoder().fit_transform(batch_cases_D11 + batch_controls_D11 + batch_controls_D12)

    is_constant = np.all(X == X[0, np.newaxis, :], axis=0)
    X = X[:, ~is_constant]

    if DATASET in {'fragment-length-distributions', 'end-motif-frequencies'}:

        if DATASET == 'fragment-length-distributions':
            X = X[:, 40:]

        X = X / np.sum(X, axis=1)[:, np.newaxis]

    if DATASET in {'fragment-length-distributions', 'end-motif-frequencies'}:
        manifold = ProbabilitySimplex()
    else:
        manifold = RatioManifold()

idx1 = np.where(np.logical_and(y == 0, d == 1))[0]
idx2 = np.where(np.logical_and(y == 0, d == 0))[0]

# Center-and-scale
X_ces = np.copy(X)
for label in ([0, 1] if (DATASET == 'OV') else [0]):
    target_scaler = RobustScaler()
    print(set(y), set(d))
    target_scaler.fit(X[np.logical_and(d == 0, y == label)])
    X_ces[np.logical_and(d == 1, y == label), :] = target_scaler.inverse_transform(
        RobustScaler().fit_transform(X[np.logical_and(d == 1, y == label), :])
    )

# Dryclean
if DATASET not in {'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions'}:
    if not os.path.exists(os.path.join(RESULTS_FOLDER, 'dryclean', f'{DATASET}-corrected.npy')):
        os.makedirs(os.path.join(RESULTS_FOLDER, 'dryclean'), exist_ok=True)
        X_dry = np.copy(X)
        X_dry[:, :] = np.nan
        X_dry[d == 1, :] = run_dryclean(bin_chr_names, bin_starts, bin_ends, X[np.logical_and(d == 1, y == 0), :], X[d == 1, :], 'tmp-dryclean')
        X_dry[d == 0, :] = run_dryclean(bin_chr_names, bin_starts, bin_ends, X[np.logical_and(d == 0, y == 0), :], X[d == 0, :], 'tmp-dryclean')
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
        Xs=X[d == 1, :],
        ys=y[d == 1, :],
        Xt=X[d == 0, :]
    )
    X_mt[d == 1, :] = model.transform(X[d == 1, :])
    np.save(os.path.join(RESULTS_FOLDER, 'mapping-transport', f'{DATASET}-corrected.npy'), X_mt)
else:
    X_mt = np.load(os.path.join(RESULTS_FOLDER, 'mapping-transport', f'{DATASET}-corrected.npy'))

# Domain adaptation
if not os.path.exists(os.path.join(RESULTS_FOLDER, 'dagip', f'{DATASET}-corrected.npy')):
    os.makedirs(os.path.join(RESULTS_FOLDER, 'dagip'), exist_ok=True)
    folder = os.path.join(ROOT, 'ichor-cna-results', 'ot-da-tmp', DATASET)
    X_adapted = np.copy(X)
    adapter = DomainAdapter(folder=folder, manifold=manifold)
    if DATASET == 'OV':
        adapter.fit(
            [X_adapted[np.logical_and(d == 1, y == 0), :], X_adapted[np.logical_and(d == 1, y == 1), :]],
            [X_adapted[np.logical_and(d == 0, y == 0), :], X_adapted[np.logical_and(d == 0, y == 1), :]]
        )
    else:
        adapter.fit(X_adapted[np.logical_and(d == 1, y == 0), :], X_adapted[np.logical_and(d == 0, y == 0), :])
    X_adapted[d == 1, :] = adapter.transform(X_adapted[d == 1, :])
    np.save(os.path.join(RESULTS_FOLDER, 'dagip', f'{DATASET}-corrected.npy'), X_adapted)
else:
    X_adapted = np.load(os.path.join(RESULTS_FOLDER, 'dagip', f'{DATASET}-corrected.npy'))


def compute_p_values(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    p_values = []
    for k in range(X1.shape[1]):
        if np.all(X1[:, k] == X1[0, k]) and np.all(X2[:, k] == X1[0, k]):
            continue
        p_values.append(float(scipy.stats.ks_2samp(X1[:, k], X2[:, k]).pvalue))
    return np.asarray(p_values)


theoretical_p_values = []
X_random = np.copy(X)
for _ in range(10):
    idx = np.concatenate((idx1, idx2), axis=0)
    idx_random = np.copy(idx)
    np.random.shuffle(idx_random)
    X_random[idx_random, :] = X[idx, :]
    theoretical_p_values.append(np.sort(compute_p_values(X_random[idx1, :], X_random[idx2, :])))
theoretical_p_values = np.mean(theoretical_p_values, axis=0)


settings = [(0, 0, X, 'Baseline', '#a883ef', '--')]
settings.append((1, 0, X_ces, 'Center-and-scale', '#f35ee2', '-.'))
settings.append((0, 1, X_mt, 'MappingTransport', '#f4c45a', '-'))
if DATASET not in {'long-fragment-ratio-profiles', 'nucleosome-positioning-score-profiles', 'end-motif-frequencies', 'fragment-length-distributions'}:
    settings.append((0, 1, X_dry, 'dryclean', '#f996ab', '-.'))
settings.append((1, 1, X_adapted, 'DAGIP', '#b1e468', '-'))


LEGEND = False
plt.figure(figsize=((6, 4) if LEGEND else (4, 4)))
ax = plt.subplot(1, 1, 1)
for kk, (i, j, X_corrected, title, color, linestyle) in enumerate(settings):

    p_values = compute_p_values(X_corrected[idx1, :], X_corrected[idx2, :])
    p_values = np.sort(p_values)
    xs = np.asarray([0] + list(p_values) + [1])
    ys = np.asarray([0] + list(theoretical_p_values) + [1])
    ax.plot(xs, ys, label=title, color=color, linestyle=linestyle)

    res = cross_validate(SVC(), X_corrected[y == 0, :], d[y == 0])
    print(res)

    print(f'MAE for {title}:', np.mean(np.abs(xs - ys)))

ax.plot([0, 1], [0, 1], color='black', alpha=0.4, linestyle='--')
ax.set_xlabel('Observed p-value quantiles')
ax.set_ylabel('Theoretical p-value quantiles')
ax.set_title(PRETTY_DATASET_NAMES[DATASET])
if LEGEND:
    ax.legend(bbox_to_anchor=(1.1, 1.05), frameon=False)
ax.grid(alpha=0.4, linestyle='--', linewidth=0.7, color='grey')
plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, f'qq-{DATASET}.png'), dpi=400)
plt.show()


import sys; sys.exit(0)

colors = ['darkcyan', 'darkcyan', 'darkcyan', 'darkcyan']
f, ax = plt.subplots(2, 2, figsize=(16, 8))
for kk, (i, j, X_corrected, title) in enumerate(settings):
    ax[i, j].axhline(y=0.5, linestyle='--', color='black')
    X1 = X_corrected[idx1, :]
    X2 = X_corrected[idx2, :]
    pvalues = np.asarray([ks_2samp(X1[:, k], X2[:, k]).pvalue for k in range(X.shape[1])])
    print(title, len(gc_content), len(pvalues))
    ax[i, j].scatter(gc_content, pvalues, alpha=0.1, color=colors[kk])
    ax[i, j].set_yscale('log')
    ax[i, j].set_xlim([0.32, 0.6])
    ax[i, j].set_ylim([[1e-64, 1e-64, 0.05, 1e-6][kk], 1.5 if (kk == 2) else None])
    # ax[i, j].spines[['right', 'top']].set_visible(False)
    ax[i, j].spines['right'].set_visible(False)
    ax[i, j].set_xlabel('GC content')
    if kk % 2 == 0:
        ax[i, j].set_ylabel('KS p-value')
    ax[i, j].set_title(title)

    bins = np.linspace(0.335, 0.6, 50)
    for start, end in zip(bins[:-1], bins[1:]):
        mask = np.logical_and(start <= gc_content, gc_content < end)
        if np.sum(mask) > 0:
            ax[i, j].scatter(
                0.5 * (start + end), np.exp(np.median(np.log(pvalues[mask]))),
                color='black', marker='D'
            )
plt.tight_layout()
if not os.path.isdir(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)
plt.savefig(os.path.join(OUT_FOLDER, f'{DATASET}-gc-bias-by-method.png'), dpi=150, transparent=True)
plt.clf()
plt.close()

import sys; sys.exit(0)

alpha = 0.5
size = 5
f, ax = plt.subplots(2, 2, figsize=(16, 12))

numb_fontsize = 14

if DATASET == 'HEMA':
    d1, d2 = '7', '8'
elif DATASET == 'OV':
    d1, d2 = '9', '10'
elif DATASET == 'NIPT-chemistry':
    d1, d2 = '6,a', '6,b'
elif DATASET == 'NIPT-lib':
    d1, d2 = '1,a', '1,b'
elif DATASET == 'NIPT-adapter':
    d1, d2 = '2,a', '2,b'
elif DATASET == 'NIPT-hs2000':
    d1, d2 = '3,a', '3,b'
elif DATASET == 'NIPT-hs2500':
    d1, d2 = '4,a', '4,b'
elif DATASET == 'NIPT-hs4000':
    d1, d2 = '5,a', '5,b'
else:
    raise NotImplementedError(f'Unknown dataset "{DATASET}"')
for i, j, X_corrected, title in settings:
    X1 = X_corrected[idx1, :]
    X2 = X_corrected[idx2, :]

    transformer = TSNE(early_exaggeration=80, perplexity=7)
    coords = transformer.fit_transform(X_corrected)

    # Get cancer stage info
    df = pd.read_csv(os.path.join(DATA_FOLDER, 'metadata.csv'))
    cancer_stage_dict = {gc_code: cancer_stage for gc_code, cancer_stage in zip(df['ID'], df['CancerStage'])}
    cancer_stages = np.asarray([cancer_stage_dict[gc_code] for gc_code in gc_codes], dtype=object)

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
            r'Ovarian carcinoma ($\mathcal{D}_{9}$)': ('#74c2c9', 's'),
            r'Healthy ($\mathcal{D}_{9}$)': ('#df998f', 'D'),
            r'Ovarian carcinoma ($\mathcal{D}_{10}$)': ('#79d991', 'o'),
            r'Healthy ($\mathcal{D}_{10}$)': ('#da83c7', 'o'),
        }  # extra color:#94a7e2
    else:
        raise NotImplementedError()
    pretty_labels = np.asarray([label_dict[label] for label in combined_labels], dtype=object)

    add_legend = ((i == 0) and (j == 1))
    scatterplot_with_sample_importances(ax[i, j], X_corrected, labels != 'Healthy', d, pretty_labels, cancer_stages, style_dict, legend=add_legend)
    ax[i, j].set_title(title)

plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, f'{DATASET}-tsne.png'), dpi=300)
plt.show()
import sys; sys.exit(0)



import scipy.stats
res = []
titles = []
for i, j, X_corrected, title in settings[::-1]:
    X1 = X_corrected[idx1, :]
    X2 = X_corrected[idx2, :]

    pvalues = np.asarray([scipy.stats.ks_2samp(X1[:, k], X2[:, k]).pvalue for k in range(X.shape[1])])
    print(title, pvalues)

    res.append(pvalues)
    titles.append(title)
r = ax[0, 3].violinplot(res, vert=False, showmeans=True, showextrema=True)
color = 'b'
r['cbars'].set_color(color)
r['cmins'].set_color(color)
r['cmaxes'].set_color(color)
r['cmeans'].set_color(color)
for body in r['bodies']:
    body.set_color(color)
# ax[0, 3].set_xscale('log')
ax[0, 3].spines['right'].set_visible(False)
ax[0, 3].spines['top'].set_visible(False)
ax[0, 3].set_xlabel('Two-sample KS p-value')
print(titles)
ax[0, 3].set_yticks(range(1, len(res) + 1), titles)
ax[0, 3].axvline(x=0.5, color='black', linestyle='--', linewidth=0.5)

r = ax[1, 3].violinplot([np.log(x) for x in res], vert=False, showmeans=True, showextrema=True)
color = 'b'
r['cbars'].set_color(color)
r['cmins'].set_color(color)
r['cmaxes'].set_color(color)
r['cmeans'].set_color(color)
for body in r['bodies']:
    body.set_color(color)
# ax[1, 3].set_xscale('log')
ax[1, 3].spines['right'].set_visible(False)
ax[1, 3].spines['top'].set_visible(False)
ax[1, 3].set_xlabel('Two-sample KS log(p-value)')
ax[1, 3].set_yticks(range(1, len(res) + 1), titles)
ax[1, 3].axvline(x=np.log(0.5), color='black', linestyle='--', linewidth=0.5)

ax[0, 4].remove()
ax[1, 4].remove()

Z1 = RobustScaler().fit_transform(X[idx1, :])
Z1_corrected = RobustScaler().fit_transform(X_gc_corrected[idx1, :])
Z1_adapted = RobustScaler().fit_transform(X_adapted[idx1, :])

zscore_lim = 3
ax[0, 5].plot([-zscore_lim, zscore_lim], [-zscore_lim, zscore_lim], color='black', linestyle='--', linewidth=0.5)
ax[0, 5].scatter(Z1_corrected.flatten()[::20], Z1_adapted.flatten()[::20], s=4, alpha=0.03, color='black')
ax[0, 5].set_xlim(-zscore_lim, zscore_lim)
ax[0, 5].set_ylim(-zscore_lim, zscore_lim)
ax[0, 5].spines['right'].set_visible(False)
ax[0, 5].spines['top'].set_visible(False)
r = pearsonr(Z1_corrected.flatten()[::20], Z1_adapted.flatten()[::20])[0]
ax[0, 5].set_title(f'Pearson r = {r:.3f}')
ax[0, 5].set_xlabel('Z-score (GC-correction)')
ax[0, 5].set_ylabel('Z-score (optimal transport)')
print('pearson GC correction', r)

# Compute optimal transport plan (after inference)
scaler = RobustScaler()
scaler.fit(X_adapted[idx2])
distances = scipy.spatial.distance.cdist(scaler.transform(X_adapted[idx1]), scaler.transform(X_adapted[idx2]))
gamma = transport_plan(distances)
unique, counts = np.unique(np.sum(gamma > 1e-5, axis=0), return_counts=True)
ax[1, 5].bar(
    unique - 0.15, counts, color='darkgoldenrod', width=0.3,
    label=r'$\mathcal{D}_{' + str(d1) + r'}$' + r' $\rightarrow$ ' + r'$\mathcal{D}_{' + str(d2) + r'}$'
)
print({u: c for u, c in zip(unique, counts)})
unique2, counts = np.unique(np.sum(gamma > 1e-5, axis=1), return_counts=True)
ax[1, 5].bar(
    unique2 + 0.15, counts, color='darkcyan', width=0.3,
    label=r'$\mathcal{D}_{' + str(d2) + r'}$' + r' $\rightarrow$ ' + r'$\mathcal{D}_{' + str(d1) + r'}$'
)
print({u: c for u, c in zip(unique2, counts)})
ax[1, 5].legend()
ax[1, 5].set_xticks(range(1, max(max(unique), max(unique2)) + 1), range(1, max(max(unique), max(unique2)) + 1))
ax[1, 5].set_xlabel(r'Number of similar patients in source domain')
ax[1, 5].set_ylabel(r'Patients in target domain')
ax[1, 5].spines['right'].set_visible(False)
ax[1, 5].spines['top'].set_visible(False)

# plt.tight_layout()
plt.savefig(os.path.join(OUT_FOLDER, f'{DATASET}-fig2.png'), dpi=150)
# plt.show()
