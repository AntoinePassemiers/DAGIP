import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '..', 'data')
RESULTS_FOLDER = os.path.join(ROOT, '..', 'results', 'pairs')
FIGURES_FOLDER = os.path.join(ROOT, '..', 'figures')
os.makedirs(FIGURES_FOLDER, exist_ok=True)


DATASETS = ['OV-forward', 'NIPT-chemistry', 'NIPT-lib', 'NIPT-adapter', 'NIPT-hs2000', 'NIPT-hs2500', 'NIPT-hs4000']
METRICS = ['pca25-accuracy', 'pca25-r2']
METRIC_NAMES = ['Pairing accuracy based on Euclidean distance', r'Coefficient of determination $R^2$']
METHODS = ['baseline', 'centering-scaling', 'mapping-transport', 'dryclean', 'da']
METHOD_NAMES = ['Baseline', 'Center-and-scale', 'MappingTransport', 'dryclean', 'DAGIP']

results = {method: {metric: [] for metric in METRICS} for method in METHODS}
for dataset in DATASETS:
    for method in METHODS:
        with open(os.path.join(RESULTS_FOLDER, f'{dataset}-{method}.json'), 'r') as f:
            data = json.load(f)
        for metric in METRICS:
            results[method][metric].append(data[metric])

plt.figure(figsize=(12, 5))
for k in range(2):

    metric = METRICS[k]

    ax = plt.subplot(1, 2, k + 1)

    data = []
    for method in METHODS:
        row = results[method][metric]
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
        yticklabels=(k == 0),
        ax=ax
    )
    ax.set_title(METRIC_NAMES[k])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_FOLDER, 'pairs-heatmaps.png'), dpi=400)
plt.show()
