import os
import sys
import json
from typing import List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))

import numpy as np
import scipy.stats
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler

from dagip.plot import plot_end_motif_freqs, scatterplot_with_sample_importances
from dagip.correction.gc import gc_correction


DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(FIGURES_FOLDER, exist_ok=True)


def load_folder(filepaths: List[str]) -> np.ndarray:
    X = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        x = df.to_numpy()[:, -1]
        if ('end-motif-frequencies' in filepath) or ('fragment-length-distributions' in filepath):
            x = x / x.sum()
        X.append(x)
    return np.asarray(X)


# Load metadata
df = pd.read_csv(os.path.join(DATA_FOLDER, 'metadata.csv'))

# Plot nucleosome positioning scores
plt.figure(figsize=(25, 5))

plot_params = dict(alpha=1.0, linewidth=2)
ax = plt.subplot(1, 2, 1)
mask = np.logical_and(df['Domain'] == 'D12', df['Category'] == 'Healthy')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D12', 'controls', 'fragment-length-distributions', f'{gc_code}.csv') for gc_code in gc_codes])
ax.plot(np.median(X, axis=0), label=r'Healthy (KAPA HyperPrep)', color='#74c2c9', **plot_params)
mask = np.logical_and(df['Domain'] == 'D11', df['Category'] == 'Healthy')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D11', 'controls', 'fragment-length-distributions', f'{gc_code}.csv') for gc_code in gc_codes])
ax.plot(np.median(X, axis=0), label=r'Healthy (NEBNext Enzymatic Methyl-seq)', color='#79d991', **plot_params)
mask = np.logical_and(df['Domain'] == 'D11', df['Category'] == 'BRCA')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D11', 'cases', 'fragment-length-distributions', f'{gc_code}.csv') for gc_code in gc_codes])
plt.plot(np.median(X, axis=0), label=r'Breast cancer (NEBNext Enzymatic Methyl-seq)', color='#df998f', **plot_params)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
ax.grid(alpha=0.4, color='grey', linestyle='--', linewidth=0.5)
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('Fragment length (bp)', fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
ax.set_title('Fragment lengths', fontsize=20)

ax = plt.subplot(1, 2, 2)
#
mask = np.logical_and(df['Domain'] == 'D12', df['Category'] == 'Healthy')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D12', 'controls', 'nucleosome-positioning-score-profiles', f'{gc_code}.csv') for gc_code in gc_codes])
x = np.median(X, axis=0)
seaborn.kdeplot(x[x > 0], label='Healthy (KAPA HyperPrep)', color='#74c2c9', ax=ax, **plot_params)
#
mask = np.logical_and(df['Domain'] == 'D11', df['Category'] == 'Healthy')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D11', 'controls', 'nucleosome-positioning-score-profiles', f'{gc_code}.csv') for gc_code in gc_codes])
x = np.median(X, axis=0)
seaborn.kdeplot(x[x > 0], label='Healthy (NEBNext Enzymatic Methyl-seq)', color='#79d991', ax=ax, **plot_params)
#
mask = np.logical_and(df['Domain'] == 'D11', df['Category'] == 'BRCA')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D11', 'cases', 'nucleosome-positioning-score-profiles', f'{gc_code}.csv') for gc_code in gc_codes])
x = np.median(X, axis=0)
seaborn.kdeplot(x[x > 0], label='Breast cancer (NEBNext Enzymatic Methyl-seq)', color='#df998f', ax=ax, **plot_params)
#
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
ax.set_title(r'Nucleosome positioning', fontsize=20)
ax.legend()
ax.set_xlabel('Nucleosome positioning score (1 Mb bins)', fontsize=15)
ax.set_ylabel('Density', fontsize=15)
ax.grid(alpha=0.4, color='grey', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'fragmentomics.png'), dpi=300)
plt.show()

# Plot long fragment ratios per bin
plt.figure(figsize=(18, 5))

ax = plt.subplot(1, 1, 1)
#
mask = np.logical_and(df['Domain'] == 'D12', df['Category'] == 'Healthy')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D12', 'controls', 'long-fragment-ratio-profiles', f'{gc_code}.csv') for gc_code in gc_codes])
x = np.median(X, axis=0)
seaborn.kdeplot(x[x > 0], label='Healthy (KAPA HyperPrep)', color='#74c2c9', ax=ax, **plot_params)
#
mask = np.logical_and(df['Domain'] == 'D11', df['Category'] == 'Healthy')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D11', 'controls', 'long-fragment-ratio-profiles', f'{gc_code}.csv') for gc_code in gc_codes])
x = np.median(X, axis=0)
seaborn.kdeplot(x[x > 0], label='Healthy (NEBNext Enzymatic Methyl-seq)', color='#79d991', ax=ax, **plot_params)
#
mask = np.logical_and(df['Domain'] == 'D11', df['Category'] == 'BRCA')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D11', 'cases', 'long-fragment-ratio-profiles', f'{gc_code}.csv') for gc_code in gc_codes])
x = np.median(X, axis=0)
seaborn.kdeplot(x[x > 0], label='Breast cancer (NEBNext Enzymatic Methyl-seq)', color='#df998f', ax=ax, **plot_params)
#
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
ax.set_title(r'Proportion of long fragments (> 166 bp)', fontsize=20)
ax.legend()
ax.set_xlabel('Proportion of long fragments (1 Mb bins)', fontsize=15)
ax.set_ylabel('Density', fontsize=15)
ax.grid(alpha=0.4, color='grey', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'long-ratios.png'), dpi=300)
plt.show()

# Plot end motif frequencies in D11
plt.figure(figsize=(25, 5))
ax = plt.subplot(1, 2, 1)
mask = np.logical_and(df['Domain'] == 'D11', df['Category'] == 'Healthy')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D11', 'controls', 'end-motif-frequencies', f'{gc_code}.csv') for gc_code in gc_codes])
x = np.median(X, axis=0)
plot_end_motif_freqs(ax, x)
ax.set_ylim([None, 0.02])
ax.set_title(r'End motif frequencies (NEBNext Enzymatic Methyl-seq)', fontsize=20)
plt.tight_layout()
# Plot end motif frequencies in D12
ax = plt.subplot(1, 2, 2)
mask = np.logical_and(df['Domain'] == 'D12', df['Category'] == 'Healthy')
gc_codes = df['ID'][mask].to_numpy()
X = load_folder([os.path.join(DATA_FOLDER, 'D12', 'controls', 'end-motif-frequencies', f'{gc_code}.csv') for gc_code in gc_codes])
x = np.median(X, axis=0)
plot_end_motif_freqs(ax, x)
#ax.set_title(r'Median end motif frequencies ($\mathcal{D}_{12}$ controls)', fontsize=20)
ax.set_title(r'End motif frequencies (KAPA HyperPrep)', fontsize=20)
ax.set_ylim([None, 0.02])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'end-motifs.png'), dpi=300)
