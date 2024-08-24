import sys

import argparse
import json
import os
import decimal

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, '..'))

import tqdm
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from statsmodels.nonparametric.smoothers_lowess import lowess

from dagip.tools.loess import loess


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(FIGURES_FOLDER, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset',
    type=str,
    choices=[
        'HL', 'DLBCL', 'MM', 'OV', 'NIPT'
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
filename = 'OV.npz' if (DATASET == 'OV') else 'HEMA.npz'
data = np.load(os.path.join(ROOT, DATA_FOLDER, 'numpy', filename), allow_pickle=True)
gc_codes = data['gc_codes']
gc_dict = {gc_code: i for i, gc_code in enumerate(gc_codes)}
X = data['X']
y = data['y']
d = np.squeeze(LabelEncoder().fit_transform(data['d'][:, np.newaxis]))
t = data['t']
groups = np.squeeze(LabelEncoder().fit_transform(data['groups'][:, np.newaxis]))

# Define labels
if DATASET == 'OV':
    y = (y == 'OV').astype(int)
elif DATASET == 'HEMA':
    y = (y != 'Healthy').astype(int)
else:
    mask = np.asarray([label in {'Healthy', DATASET} for label in y], dtype=bool)
    X, y, d, t, groups, gc_codes = X[mask], y[mask], d[mask], t[mask], groups[mask], gc_codes[mask]
    y = (y == DATASET).astype(int)
assert np.all(d < 2)

# Ensure there is at least one cancer case in the reference domain.
# Otherwise, swap domains.
if not np.any(y[d == 0]):
    d = 1 - d

# Remove problematic regions
mask = np.logical_and(gc_content >= 0, np.all(X >= 0, axis=0))
X = X[:, mask]
gc_content = gc_content[mask]
bin_chr_names = bin_chr_names[mask]
bin_starts = bin_starts[mask]
bin_ends =  bin_ends[mask]

idx = np.argsort(gc_content)
gc_content = gc_content[idx]
X = X[:, idx]

for label in [0, 1]:

    idx = np.where(y == label)[0]
    np.random.shuffle(idx)

    if len(idx) < 19:
        continue

    COLORS = ['darkcyan', 'cyan', 'violet', 'chartreuse']
    LABELS = ['1 Mb bin', 'LOWESS(span=0.3, d=1)', 'LOESS(span=0.75, d=2)', 'Polynomial(d=5)']

    plt.figure(figsize=(15, 12))
    for k, i in tqdm.tqdm(list(enumerate(idx[:19]))):

        ax = plt.subplot(4, 5, k + 1)

        y_pred = lowess(X[i, :], gc_content, frac=0.3, return_sorted=False)
        y_pred2 = loess(X[i, :], gc_content)
        y_pred3 = np.polyval(np.polyfit(gc_content, X[i, :], 5), gc_content)

        pvalue = scipy.stats.pearsonr(y_pred2, X[i, :]).pvalue
        print(scipy.stats.pearsonr(y_pred3, y_pred2))

        plt.scatter(gc_content, X[i, :], alpha=0.4, color=COLORS[0], label=LABELS[0])
        plt.plot(gc_content, y_pred, color=COLORS[1], label=LABELS[1])
        plt.plot(gc_content, y_pred2, color=COLORS[2], label=LABELS[2])
        plt.plot(gc_content, y_pred3, color=COLORS[3], label=LABELS[3])
        plt.title('p=' + '%.3E' % decimal.Decimal(pvalue))
        for side in ['right', 'top']:
            ax.spines[side].set_visible(False)
        ax.set_xlabel('GC content')
        ax.set_ylabel('Coverage')

    ax = plt.subplot(4, 5, 20)
    handles = []
    for color, name in zip(COLORS, LABELS):
        handles.append(mpatches.Patch(color=color, label=name))
    ax.legend(handles=handles)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_FOLDER, f'gc-bias-{DATASET}-{label}.png'), dpi=400)
