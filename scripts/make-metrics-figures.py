import os
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'pairs-raw')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(FIGURES_FOLDER, exist_ok=True)


DATASETS = ['OV-forward', 'NIPT-chemistry', 'NIPT-lib', 'NIPT-adapter', 'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000']
METRICS = ['pca25-accuracy', 'pca25-r2', 'accuracy', 'r2']
METRIC_NAMES = ['Pairing accuracy based on Euclidean distance', r'Coefficient of determination $R^2$']
METHODS = ['mahalanobis', 'canberra', 'cityblock', 'euclidean', 'cosine', 'braycurtis']
METHOD_NAMES = ['Mahalanobis', 'Canberra', 'Manhattan', 'Euclidean', 'Cosine', 'Bray-Curtis']

results = {method: {metric: [] for metric in METRICS} for method in METHODS}
for dataset in DATASETS:

    data = np.load(os.path.join(RESULTS_FOLDER, f'{dataset}-baseline.json.npz'))
    Y_pred = data['ypred']
    Y_target = data['ytarget']
    pca = PCA()
    pca.fit(Y_target)
    xs = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = max(np.where(xs >= 0.95)[0][0] + 1, 5)
    #n_pcs = 25
    pca = PCA(n_components=n_pcs)
    pca.fit(Y_target)
    print(n_pcs)

    Y_target = pca.transform(Y_target)
    Y_pred = pca.transform(Y_pred)

    for method in METHODS:

        D = cdist(Y_pred, Y_target, metric=method)
        correct1 = (np.arange(D.shape[0]) == np.argmin(D, axis=0))
        correct2 = (np.arange(D.shape[0]) == np.argmin(D, axis=1))
        correct = np.logical_and(correct1, correct2)
        results[method]['accuracy'].append(float(np.mean(correct)))

        ss_tot = np.mean(np.square(Y_target - np.mean(Y_target, axis=0)[np.newaxis, :]))
        ss_res = np.mean(np.square(Y_pred - Y_target))
        r2 = 1. - ss_res / ss_tot
        results[method]['r2'].append(float(r2))

plt.figure(figsize=(12, 5))

ax = plt.subplot(1, 1, 1)

data = []
for method in METHODS:
    row = results[method]['accuracy']
    row.append(np.mean(row))
    data.append(row)
data = np.asarray(data)

index = [
    r'$\mathcal{D}_{9}$ protocol / $\mathcal{D}_{10}$ protocol',
    r'NovaSeq V1 chemistry ($\mathcal{D}_{6,a}$) / NovaSeq V1.5 chemistry ($\mathcal{D}_{6,b}$)',
    r'TruSeq Nano kit ($\mathcal{D}_{1,a}$) / Kapa HyperPrep kit ($\mathcal{D}_{1,b}$)',
    r'IDT indexes ($\mathcal{D}_{2,a}$) / Kapa dual indexes ($\mathcal{D}_{2,b}$)',
    r'HiSeq 2000 ($\mathcal{D}_{3,a}$) / NovaSeq ($\mathcal{D}_{3,b}$)',
    r'HiSeq 2500 ($\mathcal{D}_{4,a}$) / NovaSeq ($\mathcal{D}_{4,b}$)',
    r'HiSeq 4000 ($\mathcal{D}_{5,a}$) / NovaSeq ($\mathcal{D}_{5,b}$)',
    'Average'
]

df = pd.DataFrame(data.T, index=index, columns=METHOD_NAMES)
seaborn.heatmap(
    df, annot=True, fmt='.3f',
    cbar=False, linewidths=2, linecolor='white',
    yticklabels=True,
    ax=ax
)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'pairs-heatmaps-per-metric.png'), dpi=400)
plt.show()
